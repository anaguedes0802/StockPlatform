"""Insider bulk loader + cluster-buy study: detection, timing, placebo, portfolio."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from app.ml import insider_study as st
from app.services import insider_bulk as ib


def _tx(rows: list[dict]) -> pd.DataFrame:
    base = {"accession": "a", "code": "P", "acq_disp": "A", "relationship": "Officer", "title": "CFO",
            "planned_10b5_1": False, "price": 10.0, "issuer_symbol": "AAA", "issuer_name": "Aaa Inc",
            "issuer_cik": 1}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["value"] = df["shares"] * df["price"]
    return df


# ---------------------------------------------------------------------------
# bulk extraction
# ---------------------------------------------------------------------------

def _zip(sub: str, own: str, trn: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", sub)
        z.writestr("REPORTINGOWNER.tsv", own)
        z.writestr("NONDERIV_TRANS.tsv", trn)
    return buf.getvalue()


def test_extract_quarter_keeps_original_form4_p_and_s() -> None:
    sub = ("ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tISSUERNAME\tISSUERTRADINGSYMBOL\tAFF10B5ONE\n"
           "A1\t31-MAR-2026\t4\t0000000001\tAaa Inc\taaa\t0\n"
           "A2\t31-MAR-2026\t4/A\t0000000001\tAaa Inc\taaa\t0\n"
           "A3\t01-APR-2026\t4\t0000000001\tAaa Inc\taaa\ttrue\n")
    own = ("ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
           "A1\t0000000009\tJane\tDirector,Officer\tCEO\nA2\t0000000009\tJane\tOfficer\tCEO\n"
           "A3\t0000000008\tBob\tDirector\t\n")
    trn = ("ACCESSION_NUMBER\tSECURITY_TITLE\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\t"
           "TRANS_ACQUIRED_DISP_CD\tSHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\n"
           "A1\tCommon Stock\t27-MAR-2026\tP\t1000\t12.5\tA\t5000\tD\n"
           "A1\tCommon Stock\t27-MAR-2026\tA\t500\t0\tA\t5500\tD\n"
           "A2\tCommon Stock\t27-MAR-2026\tP\t1000\t12.5\tA\t5000\tD\n"
           "A3\tCommon Stock\t30-MAR-2026\tS\t200\t13\tD\t800\tD\n")
    df = ib.extract_quarter(_zip(sub, own, trn))
    assert list(df["accession"]) == ["A1", "A3"]                 # grant and amendment dropped
    a1 = df.iloc[0]
    assert a1["value"] == pytest.approx(12_500) and a1["issuer_symbol"] == "AAA"
    assert a1["filing_date"] == pd.Timestamp("2026-03-31") and a1["trans_date"] == pd.Timestamp("2026-03-27")
    assert bool(df.iloc[1]["planned_10b5_1"]) is True


# ---------------------------------------------------------------------------
# cluster detection
# ---------------------------------------------------------------------------

def test_cluster_needs_two_insiders_and_min_value() -> None:
    tx = _tx([{"owner_cik": 1, "filing_date": "2020-01-02", "shares": 20_000},
              {"owner_cik": 1, "filing_date": "2020-01-05", "shares": 20_000}])     # one insider twice
    assert st.detect_clusters(tx).empty
    tx2 = _tx([{"owner_cik": 1, "filing_date": "2020-01-02", "shares": 1_000},
               {"owner_cik": 2, "filing_date": "2020-01-05", "shares": 1_000}])     # two, but $20k
    assert st.detect_clusters(tx2).empty


def test_cluster_fires_on_completing_filing_date_with_cooldown() -> None:
    tx = _tx([{"owner_cik": 1, "filing_date": "2020-01-02", "shares": 6_000},
              {"owner_cik": 2, "filing_date": "2020-01-20", "shares": 6_000},       # completes cluster
              {"owner_cik": 3, "filing_date": "2020-02-01", "shares": 6_000},       # inside cooldown
              {"owner_cik": 1, "filing_date": "2020-09-01", "shares": 6_000},
              {"owner_cik": 4, "filing_date": "2020-09-10", "shares": 6_000}])      # new event after cooldown
    ev = st.detect_clusters(tx)
    assert list(ev["signal_date"]) == [pd.Timestamp("2020-01-20"), pd.Timestamp("2020-09-10")]
    assert ev.iloc[0]["n_insiders"] == 2 and ev.iloc[0]["ceo_cfo"]


def test_cluster_window_is_30_days() -> None:
    tx = _tx([{"owner_cik": 1, "filing_date": "2020-01-01", "shares": 6_000},
              {"owner_cik": 2, "filing_date": "2020-02-15", "shares": 6_000}])
    assert st.detect_clusters(tx).empty


def test_sales_planned_and_ten_pct_owners_do_not_count() -> None:
    tx = _tx([{"owner_cik": 1, "filing_date": "2020-01-02", "shares": 6_000},
              {"owner_cik": 2, "filing_date": "2020-01-03", "shares": 6_000, "code": "S", "acq_disp": "D"},
              {"owner_cik": 3, "filing_date": "2020-01-04", "shares": 6_000, "planned_10b5_1": True},
              {"owner_cik": 4, "filing_date": "2020-01-05", "shares": 6_000, "relationship": "TenPercentOwner"}])
    assert st.detect_clusters(tx).empty


# ---------------------------------------------------------------------------
# trade timing and placebo
# ---------------------------------------------------------------------------

def _bars(n=800, start="2010-01-01", drift=0.0, seed=0, vol=1e6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 50 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                         "volume": vol}, index=idx)


def test_trade_enters_next_open_and_exits_after_hold() -> None:
    df = _bars()
    s, spy = st._series(df), st._series(_bars(seed=9))
    t = st._trade(s, spy, 100, hold=10, cost=0.0, asof_end=np.datetime64("2030-01-01"))
    assert t["entry_date"] == df.index[101].tz_localize(None)
    assert t["exit_date"] == df.index[111].tz_localize(None)
    assert t["ret"] == pytest.approx(df["open"].iloc[111] / df["open"].iloc[101] - 1)


def test_unmatured_trade_is_skipped_but_dead_stock_exits_at_last_close() -> None:
    df = _bars(n=300)
    s, spy = st._series(df), st._series(_bars(n=900, seed=9))
    recent = np.datetime64(df.index[-1].tz_localize(None).to_datetime64() + np.timedelta64(2, "D"))
    assert st._trade(s, spy, 250, hold=126, asof_end=recent) is None
    later = np.datetime64("2030-01-01")
    t = st._trade(s, spy, 250, hold=126, cost=0.0, asof_end=later)
    assert t["exit_how"] == "data_end" and t["ret"] == pytest.approx(df["close"].iloc[-1] / df["open"].iloc[251] - 1)


def _study_fixture(*, informed: bool):
    """20 up-trending stocks with sparse cluster buys. If `informed`, each
    event is followed by a +15% move over the next six months."""
    rng = np.random.default_rng(42)
    bars, rows, listings = {}, [], []
    for k in range(20):
        tick = f"S{k:02d}"
        df = _bars(n=2400, start="2006-01-02", drift=0.0006, seed=100 + k, vol=2e6)
        dates = sorted(rng.choice(np.arange(300, 2000), size=3, replace=False))
        if informed:
            c = df["close"].to_numpy().copy()
            for i in dates:
                ramp = np.ones(len(c))
                ramp[i + 1:i + 127] = np.linspace(1.0, 1.15, 126)
                ramp[i + 127:] = 1.15
                c = c * ramp
            df["close"] = c
            df["open"] = np.r_[c[0], c[:-1]]
            df["high"], df["low"] = np.maximum(df["open"], df["close"]), np.minimum(df["open"], df["close"])
        bars[tick] = df
        for i in dates:
            d = df.index[i].tz_localize(None)
            for owner in (1, 2):
                rows.append({"issuer_cik": k + 1, "issuer_symbol": tick, "owner_cik": owner,
                             "filing_date": d, "shares": 10_000})
        listings.append({"issuer_cik": k + 1, "ticker": tick, "exchange": "NYSE"})
    spy = _bars(n=2400, start="2006-01-02", drift=0.0002, seed=3, vol=1e8)
    return _tx(rows), bars, spy, pd.DataFrame(listings)


def test_study_placebo_removes_stock_drift() -> None:
    # Up-trending stocks make raw event returns look good; the same stocks on
    # random dates do as well, so there is no timing edge to report.
    tx, bars, spy, listings = _study_fixture(informed=False)
    out = st.run_study(tx, bars, spy, listings, asof=pd.Timestamp("2016-01-01"))
    assert out["counts"]["studied"] >= 50
    assert out["verdict"] != "edge"


def test_study_detects_informed_insiders() -> None:
    tx, bars, spy, listings = _study_fixture(informed=True)
    out = st.run_study(tx, bars, spy, listings, asof=pd.Timestamp("2016-01-01"))
    assert out["verdict"] == "edge"
    assert out["primary"]["mean_pct"] > 8


def test_unlisted_and_otc_events_are_counted_not_studied() -> None:
    tx = _tx([{"issuer_cik": 1, "owner_cik": 1, "filing_date": "2010-01-04", "shares": 10_000},
              {"issuer_cik": 1, "owner_cik": 2, "filing_date": "2010-01-05", "shares": 10_000},
              {"issuer_cik": 2, "owner_cik": 1, "filing_date": "2010-01-04", "shares": 10_000},
              {"issuer_cik": 2, "owner_cik": 2, "filing_date": "2010-01-05", "shares": 10_000}])
    listings = pd.DataFrame({"issuer_cik": [2], "ticker": ["OTCX"], "exchange": ["OTC"]})
    out = st.run_study(tx, {}, _bars(), listings)
    assert out["counts"]["no_current_listing"] == 1 and out["counts"]["not_major_exchange"] == 1


# ---------------------------------------------------------------------------
# portfolio
# ---------------------------------------------------------------------------

def test_portfolio_respects_slots_and_books_returns() -> None:
    spy = _bars(n=60, start="2020-01-01", seed=3)
    a = spy.copy()
    a["open"] = a["close"] = 10.0
    a.loc[a.index[20]:, ["open", "close"]] = 11.0               # +10% while held
    bars = {"AAA": a, "BBB": a.copy()}
    ev = pd.DataFrame({"ticker": ["AAA", "BBB"], "entry_date": [spy.index[5].tz_localize(None)] * 2,
                       "exit_date": [spy.index[30].tz_localize(None)] * 2, "total_value": [2.0, 1.0]})
    eq = st.simulate_portfolio(ev, bars, spy, max_positions=1, cost=0.0, initial=1000.0, start="2020-01-01")
    assert eq.iloc[-1] == pytest.approx(1000.0 + 1000.0 * 0.10)   # only one slot, one +10% trade
