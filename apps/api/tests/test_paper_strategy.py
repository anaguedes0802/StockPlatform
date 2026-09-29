from __future__ import annotations

import pytest

from app.services import paper_strategy as ps


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "DB_PATH", tmp_path / "ledger.sqlite")
    monkeypatch.setattr(ps._local, "conn", None, raising=False)
    ps._set_meta("cash", 1000.0)
    yield
    ps._local.conn = None


def _order(oid, symbol, side):
    with ps._db():
        ps._db().execute("INSERT INTO orders (id, client_order_id, rebalance_id, symbol, side, status) "
                         "VALUES (?, ?, 'r1', ?, ?, 'new')", (oid, f"ssel-{oid}", symbol, side))


def test_fills_book_into_holdings_and_cash(ledger, monkeypatch):
    fills = {
        "b1": {"status": "filled", "filled_qty": "4", "filled_avg_price": "50"},
        "s1": {"status": "filled", "filled_qty": "1", "filled_avg_price": "60"},
    }
    monkeypatch.setattr(ps.broker, "get_order", lambda oid: fills[oid])
    _order("b1", "AAA", "buy")
    assert ps.sync_fills() == 1
    assert ps.holdings()["AAA"] == {"qty": 4.0, "cost": 200.0}
    assert ps._meta("cash") == pytest.approx(800.0)

    _order("s1", "AAA", "sell")
    ps.sync_fills()
    h = ps.holdings()["AAA"]
    assert h["qty"] == 3.0 and h["cost"] == pytest.approx(150.0)   # average cost preserved
    assert ps._meta("cash") == pytest.approx(860.0)
    assert ps.sync_fills() == 0                                    # booked orders are not re-booked


def test_open_orders_wait_and_partial_cancel_books_filled_part(ledger, monkeypatch):
    state = {"status": "partially_filled", "filled_qty": "2", "filled_avg_price": "10"}
    monkeypatch.setattr(ps.broker, "get_order", lambda oid: state)
    _order("b2", "BBB", "buy")
    assert ps.sync_fills() == 0 and "BBB" not in ps.holdings()
    state["status"] = "canceled"
    assert ps.sync_fills() == 1
    assert ps.holdings()["BBB"]["qty"] == 2.0
    assert ps._meta("cash") == pytest.approx(980.0)


def test_full_sale_removes_holding(ledger, monkeypatch):
    fills = {"b": {"status": "filled", "filled_qty": "2", "filled_avg_price": "10"},
             "s": {"status": "filled", "filled_qty": "2", "filled_avg_price": "12"}}
    monkeypatch.setattr(ps.broker, "get_order", lambda oid: fills[oid])
    _order("b", "CCC", "buy"); ps.sync_fills()
    _order("s", "CCC", "sell"); ps.sync_fills()
    assert "CCC" not in ps.holdings()
    assert ps._meta("cash") == pytest.approx(1004.0)


def test_refuses_real_money_endpoint(ledger, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "paper_strategy_broker", "alpaca")
    monkeypatch.setattr(ps.broker, "is_configured", lambda: True)
    monkeypatch.setattr(ps.broker, "is_paper", lambda: False)
    with pytest.raises(ps.NotPaperError):
        ps.rebalance(execute=True, force=True)


def test_refuses_to_switch_broker_mid_test(ledger, monkeypatch):
    from app.config import settings
    ps._set_meta("broker", "sim")
    monkeypatch.setattr(settings, "paper_strategy_broker", "alpaca")
    with pytest.raises(ps.NotPaperError):
        ps._require_paper()


def test_sim_fill_books_costs(ledger, monkeypatch):
    monkeypatch.setattr(ps, "_prices", lambda syms: {s: 100.0 for s in syms})
    b = ps._sim_fill("r", "c1", "AAA", "buy", notional=1001.0)
    # 5 bps slippage -> fill 100.05; 1 bp commission comes out of the notional
    assert b["fill_price"] == pytest.approx(100.05)
    h = ps.holdings()["AAA"]
    assert h["qty"] == pytest.approx(1000.9 / 100.05 / 1.0, rel=1e-3)
    assert ps._meta("cash") == pytest.approx(1000.0 - 1001.0)
    s = ps._sim_fill("r", "c2", "AAA", "sell", qty=h["qty"])
    assert s["fill_price"] == pytest.approx(99.95)
    assert "AAA" not in ps.holdings()
    # round trip loses ~ 2x(slippage + commission) = ~12 bps
    assert ps._meta("cash") == pytest.approx(1000.0 - 1001.0 * 0.0012, rel=1e-3)


def test_plan_orders_rules():
    positions = {
        "OUT": {"market_value": 300.0, "price": 10.0},     # no longer a target -> full exit
        "BIG": {"market_value": 600.0, "price": 20.0},     # > 125% of target -> trim
        "OK":  {"market_value": 330.0, "price": 30.0},     # inside the band -> untouched
        "LOW": {"market_value": 100.0, "price": 5.0},      # < 75% of target -> top up
    }
    qty = {"OUT": 30.0, "BIG": 30.0, "OK": 11.0, "LOW": 20.0}
    sells, buys, per = ps.plan_orders(["BIG", "OK", "LOW", "NEW"], positions, qty, cash=70.0, value=1400.0)
    assert per == pytest.approx(1400 * 0.99 / 4)
    assert {x["symbol"]: x["reason"] for x in sells} == {"OUT": "left the portfolio", "BIG": "trim to target"}
    assert next(x for x in sells if x["symbol"] == "OUT")["qty"] == 30.0
    assert {b["symbol"] for b in buys} == {"LOW", "NEW"}
    budget = 70.0 + sum(x["est_value"] for x in sells) - 1400 * 0.01
    assert sum(b["notional"] for b in buys) <= budget + 0.02
