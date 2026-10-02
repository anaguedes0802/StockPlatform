"""Transaction costs for US equities through Alpaca (cash account).

Per fill, in dollars, on RAW prices and RAW share counts (splits change both):

* half the bid-ask spread: max(half a cent, 1 bp) — large caps quote a
  1-cent spread, so the floor binds for prices under $50 and the bp floor
  above;
* slippage by order type: market at a bar's open 2 bps, stop (a market order
  once triggered, usually into a fast tape) 5 bps, closing auction 1 bp;
  take-profit limits pay neither but only fill if the price trades through;
* sells pay the SEC Section 31 fee and FINRA's TAF; Alpaca charges no
  commission.

`price` is the reference price (bar open, stop level, auction price); the
function returns the adjusted fill price and the fee, so slippage moves the
fill against the trader rather than hiding in a separate bucket.
"""
from __future__ import annotations

from typing import Any

KINDS = ("market", "stop", "moc", "limit")


def fill(price_raw: float, side: int, kind: str, shares_raw: float, c: dict[str, Any]) -> tuple[float, float]:
    """(raw fill price, fees in $). side = +1 buy, −1 sell."""
    if kind not in KINDS:
        raise ValueError(kind)
    if kind == "limit":
        px = price_raw
    else:
        half = max(c["half_spread_cents"] / 100.0, price_raw * c["half_spread_bps_min"] / 1e4)
        slip = {"market": c["slippage_market_bps"], "stop": c["slippage_stop_bps"],
                "moc": c["slippage_moc_bps"]}[kind] * price_raw / 1e4
        if kind == "moc":
            half = 0.0          # the auction prints one price for both sides
        px = price_raw + side * (half + slip)
    fees = c["commission_per_share"] * shares_raw
    if side < 0:
        fees += px * shares_raw * c["sec_fee_per_million"] / 1e6
        fees += min(c["finra_taf_per_share"] * shares_raw, c["finra_taf_max"])
    return px, fees
