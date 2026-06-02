"""Price action / Smart Money Concepts engine.

Deterministic detectors for:

  - Swing points (local highs/lows with N-bar confirmation)
  - Market structure: BOS (Break of Structure) and CHOCH (Change of Character)
  - Fair Value Gaps (FVG) — 3-bar imbalances, tracked until filled
  - Order Blocks (OB) — last opposite candle before a strong impulse
  - Demand / Supply zones (volume-weighted swing point clusters)
  - Liquidity sweeps (wicks taking out a prior swing then reversing)
  - Confluence score for the latest bar

All detectors operate on a single OHLCV dataframe indexed by timestamp.
No ML, no training — pure rules-based, fast (<50ms on a 1y daily df).

References:
  - Glenn Neely / Wyckoff: supply/demand & accumulation/distribution
  - Michael Huddleston (ICT): order blocks, FVG, liquidity, BOS/CHOCH
  - Sam Seiden: institutional demand/supply zones

Implementation aims to match the common modern interpretation; specific
parameter defaults are conservative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["bullish", "bearish"]


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class SwingPoint:
    ts: pd.Timestamp
    price: float
    kind: Literal["high", "low"]
    bar_index: int


@dataclass
class FVG:
    """Fair Value Gap (3-bar imbalance)."""
    direction: Direction
    start_ts: pd.Timestamp          # ts of the middle bar (impulse bar)
    bar_index: int
    upper: float                    # top of gap
    lower: float                    # bottom of gap
    mid: float
    filled: bool = False
    fill_ts: pd.Timestamp | None = None
    strength: float = 0.0           # impulse-bar body / atr at creation


@dataclass
class OrderBlock:
    """Last opposite-color candle before a strong impulse move."""
    direction: Direction            # "bullish" = bullish OB (buyers stepped in)
    start_ts: pd.Timestamp
    bar_index: int
    upper: float
    lower: float
    mid: float
    impulse_strength: float = 0.0   # size of impulse / atr
    mitigated: bool = False         # has price returned to the zone since?
    mitigation_ts: pd.Timestamp | None = None


@dataclass
class StructureEvent:
    """BOS or CHOCH detected at a specific bar."""
    kind: Literal["BOS", "CHOCH"]
    direction: Direction
    ts: pd.Timestamp
    bar_index: int
    broken_level: float             # the swing high/low that was taken out


@dataclass
class LiquiditySweep:
    """A wick that takes out a prior swing then reverses within N bars."""
    direction: Direction            # bullish sweep = swept lows, then up; bearish = swept highs, then down
    ts: pd.Timestamp
    bar_index: int
    swept_level: float
    reversal_close: float


@dataclass
class DealingRange:
    """ICT premium/discount array over the current dealing range.

    The range is bounded by the most recent significant swing high and low.
    Equilibrium is the 50% midpoint: above it is *premium* (institutional sell
    zone), below it is *discount* (buy zone). The Optimal Trade Entry (OTE) is
    the 62%-79% retracement of the range — the deep-discount pocket for longs
    and the deep-premium pocket for shorts.
    """
    high: float
    low: float
    equilibrium: float
    # OTE zone for longs (deep discount) and shorts (deep premium).
    bull_ote_lower: float
    bull_ote_upper: float
    bear_ote_lower: float
    bear_ote_upper: float
    # Where the latest price sits: "premium" | "discount" | "equilibrium".
    position: Literal["premium", "discount", "equilibrium"]
    pct_of_range: float            # 0.0 at range low, 1.0 at range high
    in_bull_ote: bool              # price inside the long OTE pocket
    in_bear_ote: bool              # price inside the short OTE pocket


@dataclass
class PriceActionState:
    swing_points: list[SwingPoint] = field(default_factory=list)
    fvgs: list[FVG] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    structure_events: list[StructureEvent] = field(default_factory=list)
    liquidity_sweeps: list[LiquiditySweep] = field(default_factory=list)
    current_trend: Literal["up", "down", "range"] = "range"
    last_atr: float = 0.0
    dealing_range: "DealingRange | None" = None


# -----------------------------------------------------------------------------
# Detectors
# -----------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift()
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[SwingPoint]:
    """Confirmed swing point: the bar's high (low) is strictly highest (lowest)
    over `left` bars before AND `right` bars after.
    """
    highs = df["high"].values
    lows = df["low"].values
    idx = df.index
    out: list[SwingPoint] = []
    n = len(df)
    for i in range(left, n - right):
        win_h = highs[i - left : i + right + 1]
        win_l = lows[i - left : i + right + 1]
        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            out.append(SwingPoint(ts=idx[i], price=float(highs[i]), kind="high", bar_index=i))
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            out.append(SwingPoint(ts=idx[i], price=float(lows[i]), kind="low", bar_index=i))
    out.sort(key=lambda s: s.bar_index)
    return out


def detect_fvgs(df: pd.DataFrame, atr: pd.Series, min_strength: float = 0.4) -> list[FVG]:
    """Three-bar FVG. Bullish: bar[i-1].high < bar[i+1].low (gap above bar[i-1]).
       Bearish: bar[i-1].low > bar[i+1].high (gap below bar[i-1]).
    `min_strength` = (impulse body / ATR) threshold to avoid noise.
    Mark filled if any subsequent bar's high/low traverses the gap.
    """
    out: list[FVG] = []
    if len(df) < 3:
        return out
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values
    idx = df.index
    atr_v = atr.values

    for i in range(1, len(df) - 1):
        # bullish FVG
        if highs[i - 1] < lows[i + 1]:
            impulse_body = abs(closes[i] - opens[i])
            strength = float(impulse_body / max(atr_v[i], 1e-9))
            if strength < min_strength:
                continue
            upper = float(lows[i + 1])
            lower = float(highs[i - 1])
            fvg = FVG(
                direction="bullish",
                start_ts=idx[i],
                bar_index=i,
                upper=upper,
                lower=lower,
                mid=(upper + lower) / 2,
                strength=round(strength, 2),
            )
            # check fill
            for j in range(i + 2, len(df)):
                if lows[j] <= lower:
                    fvg.filled = True
                    fvg.fill_ts = idx[j]
                    break
            out.append(fvg)
        # bearish FVG
        elif lows[i - 1] > highs[i + 1]:
            impulse_body = abs(closes[i] - opens[i])
            strength = float(impulse_body / max(atr_v[i], 1e-9))
            if strength < min_strength:
                continue
            upper = float(lows[i - 1])
            lower = float(highs[i + 1])
            fvg = FVG(
                direction="bearish",
                start_ts=idx[i],
                bar_index=i,
                upper=upper,
                lower=lower,
                mid=(upper + lower) / 2,
                strength=round(strength, 2),
            )
            for j in range(i + 2, len(df)):
                if highs[j] >= upper:
                    fvg.filled = True
                    fvg.fill_ts = idx[j]
                    break
            out.append(fvg)
    return out


def _impulse_has_displacement(
    highs: np.ndarray,
    lows: np.ndarray,
    start: int,
    end: int,
    direction: Direction,
) -> bool:
    """True if the impulse leg [start, end] contains a Fair Value Gap in its
    direction — i.e. it moved with *displacement*, leaving a 3-bar imbalance.

    Canonical ICT requires an order block's impulse to displace (gap) price,
    not merely drift. A bullish leg displaces when some bar's prior high is
    below a later bar's low (gap up); bearish is the mirror. Without this gate
    every down-candle-before-a-drift gets flagged, over-generating OBs.
    """
    n = len(highs)
    for k in range(start + 1, min(end, n - 1)):
        if direction == "bullish" and highs[k - 1] < lows[k + 1]:
            return True
        if direction == "bearish" and lows[k - 1] > highs[k + 1]:
            return True
    return False


def detect_order_blocks(
    df: pd.DataFrame,
    atr: pd.Series,
    impulse_bars: int = 3,
    min_impulse: float = 1.2,
    require_displacement: bool = True,
) -> list[OrderBlock]:
    """Order block = last opposite-direction candle before a strong impulse move.

    For a bullish OB: find a down candle that is the LAST down candle before an
    upward impulse (the next bar closes up) whose total move over `impulse_bars`
    is >= `min_impulse` * ATR. When `require_displacement` is True (canonical
    ICT), the impulse must also leave a Fair Value Gap in its direction —
    otherwise the move is just drift, not institutional displacement, and the
    "order block" is noise. Returns the OB price zone (low..high of the trigger
    candle).
    """
    out: list[OrderBlock] = []
    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    idx = df.index
    atr_v = atr.values
    n = len(df)

    for i in range(1, n - impulse_bars):
        bear_candle = closes[i] < opens[i]
        bull_candle = closes[i] > opens[i]
        impulse_end = i + impulse_bars
        move = closes[impulse_end] - closes[i]
        atr_now = max(atr_v[i], 1e-9)
        impulse = move / atr_now
        # "Last opposite candle": the impulse must begin on the very next bar,
        # so candle i is the final down (up) candle before an up (down) thrust.
        starts_up = closes[i + 1] > closes[i]
        starts_down = closes[i + 1] < closes[i]

        # Bullish OB: bearish trigger candle, then strong upward move.
        if (
            bear_candle and impulse >= min_impulse and starts_up
            and (not require_displacement
                 or _impulse_has_displacement(highs, lows, i, impulse_end, "bullish"))
        ):
            ob = OrderBlock(
                direction="bullish",
                start_ts=idx[i],
                bar_index=i,
                upper=float(highs[i]),
                lower=float(lows[i]),
                mid=(float(highs[i]) + float(lows[i])) / 2,
                impulse_strength=round(float(impulse), 2),
            )
            # mark mitigated if price later returned into the zone
            for j in range(impulse_end + 1, n):
                if lows[j] <= ob.upper:
                    ob.mitigated = True
                    ob.mitigation_ts = idx[j]
                    break
            out.append(ob)

        # Bearish OB: bullish trigger candle, then strong downward move.
        elif (
            bull_candle and -impulse >= min_impulse and starts_down
            and (not require_displacement
                 or _impulse_has_displacement(highs, lows, i, impulse_end, "bearish"))
        ):
            ob = OrderBlock(
                direction="bearish",
                start_ts=idx[i],
                bar_index=i,
                upper=float(highs[i]),
                lower=float(lows[i]),
                mid=(float(highs[i]) + float(lows[i])) / 2,
                impulse_strength=round(float(-impulse), 2),
            )
            for j in range(impulse_end + 1, n):
                if highs[j] >= ob.lower:
                    ob.mitigated = True
                    ob.mitigation_ts = idx[j]
                    break
            out.append(ob)

    return out


def detect_structure_events(
    df: pd.DataFrame,
    swings: list[SwingPoint],
) -> tuple[list[StructureEvent], Literal["up", "down", "range"]]:
    """Walk the swing sequence chronologically and classify each new break.

    State machine:
      - In an uptrend, a higher-high break = BOS (continuation).
        First lower-low after a HH = CHOCH (trend change → down).
      - Symmetric for downtrends.
    Returns (events, current_trend_at_end).
    """
    events: list[StructureEvent] = []
    if len(swings) < 4:
        return events, "range"

    # Track running highest-high and lowest-low among swings, plus current state.
    state: Literal["up", "down", "range"] = "range"
    last_hh: SwingPoint | None = None
    last_ll: SwingPoint | None = None

    for sp in swings:
        if sp.kind == "high":
            if last_hh and sp.price > last_hh.price:
                # broke a prior high
                if state == "down":
                    events.append(StructureEvent(
                        kind="CHOCH", direction="bullish",
                        ts=sp.ts, bar_index=sp.bar_index, broken_level=last_hh.price,
                    ))
                    state = "up"
                else:
                    events.append(StructureEvent(
                        kind="BOS", direction="bullish",
                        ts=sp.ts, bar_index=sp.bar_index, broken_level=last_hh.price,
                    ))
                    state = "up"
                last_hh = sp
            elif not last_hh or sp.price > last_hh.price:
                last_hh = sp
        else:  # low
            if last_ll and sp.price < last_ll.price:
                if state == "up":
                    events.append(StructureEvent(
                        kind="CHOCH", direction="bearish",
                        ts=sp.ts, bar_index=sp.bar_index, broken_level=last_ll.price,
                    ))
                    state = "down"
                else:
                    events.append(StructureEvent(
                        kind="BOS", direction="bearish",
                        ts=sp.ts, bar_index=sp.bar_index, broken_level=last_ll.price,
                    ))
                    state = "down"
                last_ll = sp
            elif not last_ll or sp.price < last_ll.price:
                last_ll = sp

    return events, state


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    reversal_bars: int = 3,
    sweep_threshold_atr: float = 0.2,
    atr: pd.Series | None = None,
) -> list[LiquiditySweep]:
    """A wick takes out a prior swing point by at least `sweep_threshold_atr * ATR`,
    then the close within the next `reversal_bars` reclaims back the other side.
    """
    if atr is None:
        atr = _atr(df)
    out: list[LiquiditySweep] = []
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    idx = df.index
    atr_v = atr.values
    n = len(df)

    swing_highs = [s for s in swings if s.kind == "high"]
    swing_lows = [s for s in swings if s.kind == "low"]

    for i in range(1, n - reversal_bars):
        atr_i = max(atr_v[i], 1e-9)
        # bullish sweep: wick below a recent swing low, then close back above
        recent_lows = [s for s in swing_lows if s.bar_index < i - 1]
        if recent_lows:
            sl = recent_lows[-1]
            if lows[i] < sl.price - sweep_threshold_atr * atr_i:
                # check reversal
                window_close = closes[i + 1 : i + 1 + reversal_bars]
                if len(window_close) and window_close.max() > sl.price:
                    out.append(LiquiditySweep(
                        direction="bullish",
                        ts=idx[i],
                        bar_index=i,
                        swept_level=sl.price,
                        reversal_close=float(window_close.max()),
                    ))
        # bearish sweep: wick above a recent swing high, then close back below
        recent_highs = [s for s in swing_highs if s.bar_index < i - 1]
        if recent_highs:
            sh = recent_highs[-1]
            if highs[i] > sh.price + sweep_threshold_atr * atr_i:
                window_close = closes[i + 1 : i + 1 + reversal_bars]
                if len(window_close) and window_close.min() < sh.price:
                    out.append(LiquiditySweep(
                        direction="bearish",
                        ts=idx[i],
                        bar_index=i,
                        swept_level=sh.price,
                        reversal_close=float(window_close.min()),
                    ))
    return out


def compute_dealing_range(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    last_price: float,
    lookback: int = 90,
    eq_band: float = 0.05,
    ote_lower: float = 0.62,
    ote_upper: float = 0.79,
) -> DealingRange | None:
    """Build the premium/discount array for the current dealing range.

    The range high/low come from the most recent confirmed swing high and swing
    low within `lookback` bars (falling back to the rolling price extremes when
    swings are sparse). Equilibrium is the 50% level; `eq_band` is the
    half-width (as a fraction of range) of the neutral equilibrium zone so a
    price hovering at the midpoint isn't flip-flopped between premium/discount.

    OTE pockets use the classic 0.62-0.79 retracement: for a long, that's the
    deep-discount band measured down from the range high; for a short, the
    deep-premium band measured up from the range low.
    """
    if df.empty or len(df) < 10 or last_price <= 0:
        return None

    n = len(df)
    window = df.iloc[max(0, n - lookback):]
    cutoff_idx = n - len(window)
    recent_highs = [s.price for s in swings if s.kind == "high" and s.bar_index >= cutoff_idx]
    recent_lows = [s.price for s in swings if s.kind == "low" and s.bar_index >= cutoff_idx]

    hi = max(recent_highs) if recent_highs else float(window["high"].max())
    lo = min(recent_lows) if recent_lows else float(window["low"].min())
    if hi <= lo:
        return None

    rng = hi - lo
    eq = lo + 0.5 * rng

    # Retracement bands. Bullish OTE = deep discount (price retraced 62-79% down
    # from the high); bearish OTE = deep premium (62-79% up from the low).
    bull_ote_upper = hi - ote_lower * rng   # = lo + 0.38*rng
    bull_ote_lower = hi - ote_upper * rng   # = lo + 0.21*rng
    bear_ote_lower = lo + ote_lower * rng   # = lo + 0.62*rng
    bear_ote_upper = lo + ote_upper * rng   # = lo + 0.79*rng

    pct = (last_price - lo) / rng
    if abs(last_price - eq) <= eq_band * rng:
        position: Literal["premium", "discount", "equilibrium"] = "equilibrium"
    elif last_price > eq:
        position = "premium"
    else:
        position = "discount"

    return DealingRange(
        high=float(hi),
        low=float(lo),
        equilibrium=float(eq),
        bull_ote_lower=float(bull_ote_lower),
        bull_ote_upper=float(bull_ote_upper),
        bear_ote_lower=float(bear_ote_lower),
        bear_ote_upper=float(bear_ote_upper),
        position=position,
        pct_of_range=float(np.clip(pct, 0.0, 1.0)),
        in_bull_ote=bool(bull_ote_lower <= last_price <= bull_ote_upper),
        in_bear_ote=bool(bear_ote_lower <= last_price <= bear_ote_upper),
    )


# -----------------------------------------------------------------------------
# Demand / Supply zones — clusters of swings + order blocks
# -----------------------------------------------------------------------------

@dataclass
class Zone:
    direction: Direction       # bullish=demand, bearish=supply
    upper: float
    lower: float
    mid: float
    strength: float            # how many overlapping components (swing/OB/FVG)
    last_touch_ts: pd.Timestamp | None
    components: list[str]


def build_zones(
    swings: list[SwingPoint],
    obs: list[OrderBlock],
    fvgs: list[FVG],
    atr_level: float,
    df: pd.DataFrame,
) -> list[Zone]:
    """Merge nearby same-direction structures into actionable demand/supply zones."""
    bull_components: list[tuple[float, float, str, pd.Timestamp | None]] = []
    bear_components: list[tuple[float, float, str, pd.Timestamp | None]] = []

    for ob in obs:
        if ob.direction == "bullish":
            bull_components.append((ob.lower, ob.upper, "OB", ob.start_ts))
        else:
            bear_components.append((ob.lower, ob.upper, "OB", ob.start_ts))
    for f in fvgs:
        if f.filled:
            continue
        if f.direction == "bullish":
            bull_components.append((f.lower, f.upper, "FVG", f.start_ts))
        else:
            bear_components.append((f.lower, f.upper, "FVG", f.start_ts))

    # Recent swing lows seed demand zones; recent swing highs seed supply zones.
    cutoff_bars = max(0, len(df) - 252)  # last ~1y of daily bars
    for s in swings:
        if s.bar_index < cutoff_bars:
            continue
        half = atr_level * 0.5
        if s.kind == "low":
            bull_components.append((s.price - half, s.price + half, "swing_low", s.ts))
        else:
            bear_components.append((s.price - half, s.price + half, "swing_high", s.ts))

    return _cluster(bull_components, "bullish", atr_level) + _cluster(bear_components, "bearish", atr_level)


def _cluster(
    parts: list[tuple[float, float, str, pd.Timestamp | None]],
    direction: Direction,
    atr_level: float,
) -> list[Zone]:
    """Cluster nearby components into zones. Two clustering disciplines a real
    desk analyst applies (and we previously didn't):

    1. **Width cap**: the resulting zone must be ≤ 2× ATR wide. A "demand zone"
       that spans 12 % of price is not a zone — it's a region, useless for
       entries. When merging another part would push the cluster past 2× ATR,
       we close the current cluster and start a new one.
    2. **Centroid gap**: even within 1× ATR centroid distance, we don't merge
       if the resulting span is over-wide.

    Each cluster's `strength` counts merged components (1 = isolated pivot;
    2-3 = corroborated; 5+ = strong S/R level). Time stamp tracks freshness.
    """
    if not parts:
        return []
    parts = sorted(parts, key=lambda t: (t[0] + t[1]) / 2)
    zones: list[Zone] = []
    cluster: list[tuple[float, float, str, pd.Timestamp | None]] = [parts[0]]

    MAX_WIDTH = 2.0 * atr_level   # zone-width discipline

    def flush(buf: list[tuple]) -> Zone | None:
        if not buf:
            return None
        lo = min(b[0] for b in buf)
        hi = max(b[1] for b in buf)
        comps = [b[2] for b in buf]
        last_ts = max([b[3] for b in buf if b[3] is not None], default=None)
        return Zone(
            direction=direction,
            upper=float(hi),
            lower=float(lo),
            mid=float((hi + lo) / 2),
            strength=float(len(buf)),
            last_touch_ts=last_ts,
            components=comps,
        )

    for cur in parts[1:]:
        prev_mid = (cluster[-1][0] + cluster[-1][1]) / 2
        cur_mid = (cur[0] + cur[1]) / 2
        # Hypothetical merged width if we added `cur` to the current cluster.
        hyp_lo = min(min(b[0] for b in cluster), cur[0])
        hyp_hi = max(max(b[1] for b in cluster), cur[1])
        too_wide = (hyp_hi - hyp_lo) > MAX_WIDTH
        close_enough = abs(cur_mid - prev_mid) <= atr_level
        if close_enough and not too_wide:
            cluster.append(cur)
        else:
            z = flush(cluster)
            if z: zones.append(z)
            cluster = [cur]
    z = flush(cluster)
    if z: zones.append(z)
    return zones


def classify_zones_by_polarity(
    zones: list[Zone],
    last_price: float,
) -> tuple[list[Zone], list[Zone], list[Zone], list[Zone]]:
    """Apply the Wyckoff/ICT polarity principle.

       Demand (bullish) zones:
         - Still active as long as price is *above or inside* the zone (price
           hasn't broken below the lower bound).
         - "Flipped to resistance" once price closes BELOW `lower` by a
           cushion — trapped longs become future sellers on the retest.

       Supply (bearish) zones:
         - Still active as long as price is *below or inside* the zone (price
           hasn't broken above the upper bound).
         - "Flipped to support" once price closes ABOVE `upper` by a cushion —
           trapped shorts become future buyers on the retest.

    Cushion = half the zone's height (so a small probe into the zone doesn't
    declare a polarity flip — the zone has to be conclusively breached).

    Returns (active_demand, active_supply, flipped_resistance, flipped_support).
    The opinion engine uses ACTIVE lists for entry/stop planning. FLIPPED
    lists are useful context ("former resistance, now support if retested")
    but not entry levels in their flipped role.
    """
    active_demand: list[Zone] = []
    active_supply: list[Zone] = []
    flipped_res: list[Zone] = []   # broken demand zones (former support, now resistance)
    flipped_sup: list[Zone] = []   # broken supply zones (former resistance, now support)
    if last_price <= 0:
        return ([z for z in zones if z.direction == "bullish"],
                [z for z in zones if z.direction == "bearish"], [], [])
    for z in zones:
        half_h = max((z.upper - z.lower) / 2, 1e-6)
        if z.direction == "bullish":
            # Demand. Broken when price has closed below the zone's lower
            # bound by more than half its height.
            if last_price < z.lower - half_h:
                flipped_res.append(z)
            else:
                active_demand.append(z)
        else:
            # Supply. Broken when price has closed above the zone's upper
            # bound by more than half its height.
            if last_price > z.upper + half_h:
                flipped_sup.append(z)
            else:
                active_supply.append(z)
    return active_demand, active_supply, flipped_res, flipped_sup


# -----------------------------------------------------------------------------
# Confluence score for the latest bar
# -----------------------------------------------------------------------------

def confluence(
    df: pd.DataFrame,
    state: PriceActionState,
    zones: list[Zone],
) -> dict:
    """Score the *current* setup. Range: [-1, +1]. Bullish > 0, bearish < 0.

    Components (each ∈ [-1,1], weights sum to 1):
      - structure: BOS/CHOCH in the last 10 bars
      - zone_proximity: is price inside or within 0.5*ATR of an unfilled demand/supply zone?
      - recent_fvg: did an FVG form in the last 5 bars?
      - liquidity_sweep: did a sweep print within the last 3 bars?
      - premium_discount: where price sits in the dealing range (discount/OTE = bullish)
      - trend_alignment: current trend (from structure walker)
    """
    if df.empty:
        return {"score": 0.0, "label": "neutral", "components": {}, "drivers": []}

    last_close = float(df["close"].iloc[-1])
    last_bar = len(df) - 1
    atr_now = state.last_atr

    drivers: list[str] = []
    comp: dict[str, float] = {"structure": 0.0, "zone_proximity": 0.0, "recent_fvg": 0.0,
                              "liquidity_sweep": 0.0, "premium_discount": 0.0,
                              "trend_alignment": 0.0}

    # structure
    recent_events = [e for e in state.structure_events if last_bar - e.bar_index <= 10]
    if recent_events:
        ev = recent_events[-1]
        weight = 1.0 if ev.kind == "CHOCH" else 0.6
        comp["structure"] = weight if ev.direction == "bullish" else -weight
        drivers.append(f"{ev.kind} {ev.direction} {last_bar - ev.bar_index} bars ago")

    # zone proximity (closest active zone within 0.5 * ATR)
    if atr_now > 0:
        closest: tuple[float, Zone | None] = (float("inf"), None)
        for z in zones:
            if z.lower <= last_close <= z.upper:
                closest = (0.0, z); break
            d = min(abs(last_close - z.lower), abs(last_close - z.upper))
            if d < closest[0]:
                closest = (d, z)
        d, z = closest
        if z and d <= 0.5 * atr_now:
            score = 1.0 - (d / max(0.5 * atr_now, 1e-9))
            comp["zone_proximity"] = score if z.direction == "bullish" else -score
            kind = "demand" if z.direction == "bullish" else "supply"
            from collections import Counter
            c = Counter(z.components)
            comp_summary = ", ".join(f"{n}×{k}" if n > 1 else k for k, n in c.most_common(3))
            drivers.append(f"At {kind} zone (strength {int(z.strength)}; {comp_summary})")

    # recent FVG
    recent_fvgs = [f for f in state.fvgs if last_bar - f.bar_index <= 5 and not f.filled]
    if recent_fvgs:
        f = recent_fvgs[-1]
        comp["recent_fvg"] = 0.6 if f.direction == "bullish" else -0.6
        drivers.append(f"Unfilled {f.direction} FVG {last_bar - f.bar_index} bars ago (strength {f.strength})")

    # liquidity sweep
    recent_sweeps = [s for s in state.liquidity_sweeps if last_bar - s.bar_index <= 3]
    if recent_sweeps:
        s = recent_sweeps[-1]
        comp["liquidity_sweep"] = 0.8 if s.direction == "bullish" else -0.8
        drivers.append(f"{s.direction.capitalize()} liquidity sweep {last_bar - s.bar_index} bars ago")

    # premium / discount (ICT array): buying in discount / OTE is bullish-
    # favourable, selling in premium / short-OTE is bearish-favourable. The OTE
    # pockets are the strongest read; plain premium/discount is a softer bias.
    dr = state.dealing_range
    if dr is not None:
        if dr.in_bull_ote:
            comp["premium_discount"] = 1.0
            drivers.append(f"Price in bullish OTE (discount {dr.pct_of_range:.0%} of range)")
        elif dr.in_bear_ote:
            comp["premium_discount"] = -1.0
            drivers.append(f"Price in bearish OTE (premium {dr.pct_of_range:.0%} of range)")
        elif dr.position == "discount":
            comp["premium_discount"] = 0.4
            drivers.append(f"Discount ({dr.pct_of_range:.0%} of range)")
        elif dr.position == "premium":
            comp["premium_discount"] = -0.4
            drivers.append(f"Premium ({dr.pct_of_range:.0%} of range)")

    # trend alignment
    if state.current_trend == "up":
        comp["trend_alignment"] = 0.4; drivers.append("Trend: up")
    elif state.current_trend == "down":
        comp["trend_alignment"] = -0.4; drivers.append("Trend: down")

    # weighted blend (intentional — structure and sweeps are higher signal than
    # trend; premium/discount locates the entry within the range). Sums to 1.0.
    weights = {"structure": 0.25, "zone_proximity": 0.20, "recent_fvg": 0.12,
               "liquidity_sweep": 0.18, "premium_discount": 0.15, "trend_alignment": 0.10}
    score = sum(comp[k] * weights[k] for k in weights)
    score = max(-1.0, min(1.0, score))

    if score > 0.4: label = "strong_bullish"
    elif score > 0.15: label = "bullish"
    elif score < -0.4: label = "strong_bearish"
    elif score < -0.15: label = "bearish"
    else: label = "neutral"

    return {
        "score": round(float(score), 4),
        "label": label,
        "components": {k: round(v, 3) for k, v in comp.items()},
        "drivers": drivers,
    }


# -----------------------------------------------------------------------------
# Top-level analyze()
# -----------------------------------------------------------------------------

def analyze(df: pd.DataFrame, swing_window: int = 3) -> dict:
    """Compute all structures + confluence on a single OHLCV dataframe."""
    if df.empty or len(df) < 30:
        return {
            "ok": False,
            "error": "insufficient data",
            "n_bars": len(df),
        }

    atr_series = _atr(df, 14)
    state = PriceActionState(last_atr=float(atr_series.iloc[-1]))

    state.swing_points = detect_swings(df, left=swing_window, right=swing_window)
    state.fvgs = detect_fvgs(df, atr_series)
    state.order_blocks = detect_order_blocks(df, atr_series)
    events, trend = detect_structure_events(df, state.swing_points)
    state.structure_events = events
    state.current_trend = trend
    state.liquidity_sweeps = detect_liquidity_sweeps(df, state.swing_points, atr=atr_series)

    last_price = float(df["close"].iloc[-1])
    state.dealing_range = compute_dealing_range(df, state.swing_points, last_price)

    zones = build_zones(state.swing_points, state.order_blocks, state.fvgs, state.last_atr, df)

    # Polarity classification: separate truly-active S/R from levels that price
    # has already broken (former demand now resistance candidate, etc.). A pro
    # would never call a level "supply" if price has cleared it months ago —
    # that's now a flipped support level, totally different read.
    active_demand, active_supply, flipped_res, flipped_sup = classify_zones_by_polarity(zones, last_price)

    # Order each list by what a trader actually cares about: closest to
    # current price first. Demand is below price (price would fall to it);
    # supply is above (price would rise to it). Sort each by absolute
    # distance from current price.
    active_demand.sort(key=lambda z: abs(z.mid - last_price))
    active_supply.sort(key=lambda z: abs(z.mid - last_price))
    flipped_res.sort(key=lambda z: (z.last_touch_ts or pd.Timestamp.min), reverse=True)
    flipped_sup.sort(key=lambda z: (z.last_touch_ts or pd.Timestamp.min), reverse=True)

    conf = confluence(df, state, zones)

    # "Price discovery": no active overhead supply ABOVE current price.
    # An active supply zone whose mid is above price is what defines overhead
    # resistance; if every active_supply has mid <= last_price then it's all
    # already cleared and we're in blue-sky territory.
    overhead_supply = [z for z in active_supply if z.mid > last_price]
    price_in_discovery = len(overhead_supply) == 0 and state.current_trend == "up"

    # Nearest demand strictly BELOW current price (only those are actionable
    # as buy-the-dip targets). Same for supply strictly ABOVE current price.
    dem_below = [z for z in active_demand if z.mid < last_price]
    sup_above = [z for z in active_supply if z.mid > last_price]
    nearest_demand_pct = ((dem_below[0].mid - last_price) / last_price * 100
                          if dem_below else None)
    nearest_supply_pct = ((sup_above[0].mid - last_price) / last_price * 100
                          if sup_above else None)

    return {
        "ok": True,
        "n_bars": len(df),
        "atr": round(state.last_atr, 4),
        "atr_pct_of_price": round(state.last_atr / last_price * 100, 2) if last_price else None,
        "last_price": last_price,
        "current_trend": state.current_trend,
        "price_in_discovery": price_in_discovery,  # blue-sky breakout
        "nearest_demand_pct_below": round(nearest_demand_pct, 2) if nearest_demand_pct is not None else None,
        "nearest_supply_pct_above": round(nearest_supply_pct, 2) if nearest_supply_pct is not None else None,
        "swing_points": [_swing_to_dict(s) for s in state.swing_points[-40:]],
        "fvgs": [_fvg_to_dict(f) for f in state.fvgs[-40:]],
        "order_blocks": [_ob_to_dict(o) for o in state.order_blocks[-30:]],
        "structure_events": [_se_to_dict(e) for e in state.structure_events[-15:]],
        "liquidity_sweeps": [_ls_to_dict(s) for s in state.liquidity_sweeps[-10:]],
        # Legacy `zones` kept for back-compat — full list of every cluster.
        "zones": [_zone_to_dict(z) for z in zones],
        # New: polarity-aware classification — this is what the opinion engine
        # and the smart-money UI panel should consume.
        "active_demand": [_zone_to_dict(z) for z in active_demand],
        "active_supply": [_zone_to_dict(z) for z in active_supply],
        "flipped_resistance": [_zone_to_dict(z) for z in flipped_res],  # broken demand
        "flipped_support":    [_zone_to_dict(z) for z in flipped_sup],  # broken supply
        # New: premium/discount array + Optimal Trade Entry zones (ICT).
        "dealing_range": _dr_to_dict(state.dealing_range) if state.dealing_range else None,
        "confluence": conf,
    }


# -----------------------------------------------------------------------------
# Serializers
# -----------------------------------------------------------------------------

def _swing_to_dict(s: SwingPoint) -> dict:
    return {"ts": s.ts.isoformat(), "price": round(s.price, 4), "kind": s.kind}


def _fvg_to_dict(f: FVG) -> dict:
    return {
        "ts": f.start_ts.isoformat(),
        "direction": f.direction,
        "upper": round(f.upper, 4),
        "lower": round(f.lower, 4),
        "filled": f.filled,
        "fill_ts": f.fill_ts.isoformat() if f.fill_ts else None,
        "strength": f.strength,
    }


def _ob_to_dict(o: OrderBlock) -> dict:
    return {
        "ts": o.start_ts.isoformat(),
        "direction": o.direction,
        "upper": round(o.upper, 4),
        "lower": round(o.lower, 4),
        "impulse_strength": o.impulse_strength,
        "mitigated": o.mitigated,
        "mitigation_ts": o.mitigation_ts.isoformat() if o.mitigation_ts else None,
    }


def _se_to_dict(e: StructureEvent) -> dict:
    return {
        "kind": e.kind,
        "direction": e.direction,
        "ts": e.ts.isoformat(),
        "broken_level": round(e.broken_level, 4),
    }


def _ls_to_dict(s: LiquiditySweep) -> dict:
    return {
        "ts": s.ts.isoformat(),
        "direction": s.direction,
        "swept_level": round(s.swept_level, 4),
        "reversal_close": round(s.reversal_close, 4),
    }


def _dr_to_dict(dr: DealingRange) -> dict:
    return {
        "high": round(dr.high, 4),
        "low": round(dr.low, 4),
        "equilibrium": round(dr.equilibrium, 4),
        "bull_ote_lower": round(dr.bull_ote_lower, 4),
        "bull_ote_upper": round(dr.bull_ote_upper, 4),
        "bear_ote_lower": round(dr.bear_ote_lower, 4),
        "bear_ote_upper": round(dr.bear_ote_upper, 4),
        "position": dr.position,
        "pct_of_range": round(dr.pct_of_range, 4),
        "in_bull_ote": dr.in_bull_ote,
        "in_bear_ote": dr.in_bear_ote,
    }


def _zone_to_dict(z: Zone) -> dict:
    return {
        "direction": z.direction,
        "upper": round(z.upper, 4),
        "lower": round(z.lower, 4),
        "mid": round(z.mid, 4),
        "strength": z.strength,
        "components": z.components,
        "last_touch_ts": z.last_touch_ts.isoformat() if z.last_touch_ts else None,
    }
