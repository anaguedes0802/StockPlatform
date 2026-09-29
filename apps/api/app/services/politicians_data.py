"""Hand-written DEMO sample of Congress trades — NOT real disclosures.

Members are fictional placeholders ("Demo Member NN") so invented trades are never
attributed to real people.

Real data comes from the official House Clerk filings (app/services/congress_ptr.py,
refreshed by scripts/refresh_congress.py) and, for the Senate, Finnhub when
FINNHUB_API_KEY is set. This fixture is only served when neither source has data,
and politicians.py always tags those rows `demo: true`.
"""
from __future__ import annotations

# fields: politician, chamber, party, traded_at, disclosed_at, symbol, side, amount_range
BUNDLED_TRADES: list[dict] = [
    {"politician": "Demo Member 01", "chamber": "House",  "party": "—",
     "traded_at": "2025-03-12", "disclosed_at": "2025-04-04",
     "symbol": "NVDA", "side": "buy",  "amount_range": "$1M – $5M",
     "note": "Call options on NVDA"},
    {"politician": "Demo Member 01", "chamber": "House",  "party": "—",
     "traded_at": "2025-02-21", "disclosed_at": "2025-03-15",
     "symbol": "TEM",  "side": "buy",  "amount_range": "$50K – $100K",
     "note": "Initial position in Tempus AI"},
    {"politician": "Demo Member 02",  "chamber": "House",  "party": "—",
     "traded_at": "2025-04-02", "disclosed_at": "2025-04-22",
     "symbol": "PLTR", "side": "buy",  "amount_range": "$15K – $50K"},
    {"politician": "Demo Member 03","chamber": "House",  "party": "—",
     "traded_at": "2025-04-15", "disclosed_at": "2025-05-01",
     "symbol": "MSFT", "side": "buy",  "amount_range": "$15K – $50K"},
    {"politician": "Demo Member 03","chamber": "House",  "party": "—",
     "traded_at": "2025-04-15", "disclosed_at": "2025-05-01",
     "symbol": "GOOGL","side": "buy",  "amount_range": "$15K – $50K"},
    {"politician": "Demo Member 04","chamber": "Senate","party": "—",
     "traded_at": "2025-03-28", "disclosed_at": "2025-04-15",
     "symbol": "AVGO", "side": "buy",  "amount_range": "$50K – $100K"},
    {"politician": "Demo Member 05","chamber": "Senate","party": "—",
     "traded_at": "2025-03-04", "disclosed_at": "2025-03-25",
     "symbol": "META", "side": "buy",  "amount_range": "$15K – $50K"},
    {"politician": "Demo Member 06",     "chamber": "House", "party": "—",
     "traded_at": "2025-02-14", "disclosed_at": "2025-03-07",
     "symbol": "VST",  "side": "buy",  "amount_range": "$100K – $250K",
     "note": "Vistra (nuclear-AI thematic)"},
    {"politician": "Demo Member 07", "chamber": "House", "party": "—",
     "traded_at": "2025-01-22", "disclosed_at": "2025-02-12",
     "symbol": "NVDA", "side": "buy",  "amount_range": "$50K – $100K"},
    {"politician": "Demo Member 08",      "chamber": "House", "party": "—",
     "traded_at": "2025-02-07", "disclosed_at": "2025-03-01",
     "symbol": "AAPL", "side": "buy",  "amount_range": "$1K – $15K"},
    {"politician": "Demo Member 01",   "chamber": "House", "party": "—",
     "traded_at": "2024-12-20", "disclosed_at": "2025-01-13",
     "symbol": "GOOG", "side": "buy",  "amount_range": "$500K – $1M",
     "note": "Call options on Alphabet C"},
    {"politician": "Demo Member 01",   "chamber": "House", "party": "—",
     "traded_at": "2024-12-20", "disclosed_at": "2025-01-13",
     "symbol": "AVGO", "side": "buy",  "amount_range": "$500K – $1M",
     "note": "Call options on Broadcom"},
    {"politician": "Demo Member 09",  "chamber": "House", "party": "—",
     "traded_at": "2025-01-30", "disclosed_at": "2025-02-21",
     "symbol": "AMZN", "side": "buy",  "amount_range": "$15K – $50K"},
    {"politician": "Demo Member 10",   "chamber": "Senate","party": "—",
     "traded_at": "2025-02-25", "disclosed_at": "2025-03-19",
     "symbol": "JPM",  "side": "buy",  "amount_range": "$50K – $100K"},
    {"politician": "Demo Member 11","chamber":"Senate","party": "—",
     "traded_at": "2025-03-05", "disclosed_at": "2025-03-27",
     "symbol": "BRK-B","side": "buy",  "amount_range": "$15K – $50K"},
    {"politician": "Demo Member 01",   "chamber": "House", "party": "—",
     "traded_at": "2024-11-10", "disclosed_at": "2024-12-03",
     "symbol": "TEM",  "side": "buy",  "amount_range": "$100K – $250K",
     "note": "First public health-AI position"},
    {"politician": "Demo Member 12","chamber":"House","party": "—",
     "traded_at": "2025-04-08", "disclosed_at": "2025-04-29",
     "symbol": "QQQ",  "side": "buy",  "amount_range": "$15K – $50K"},
    {"politician": "Demo Member 13",     "chamber": "Senate","party": "—",
     "traded_at": "2025-03-19", "disclosed_at": "2025-04-09",
     "symbol": "DIS",  "side": "sell", "amount_range": "$50K – $100K"},
    {"politician": "Demo Member 14","chamber":"House","party": "—",
     "traded_at": "2025-02-17", "disclosed_at": "2025-03-10",
     "symbol": "AAPL", "side": "buy",  "amount_range": "$1K – $15K"},
    {"politician": "Demo Member 15", "chamber": "House", "party": "—",
     "traded_at": "2025-03-24", "disclosed_at": "2025-04-14",
     "symbol": "MSFT", "side": "buy",  "amount_range": "$15K – $50K"},
]


# Mid-point estimate (in USD) for each disclosed range. Used to weight signals.
RANGE_MIDPOINTS: dict[str, float] = {
    "$1K – $15K":     8_000,
    "$15K – $50K":    32_500,
    "$50K – $100K":   75_000,
    "$100K – $250K":  175_000,
    "$250K – $500K":  375_000,
    "$500K – $1M":    750_000,
    "$1M – $5M":      3_000_000,
    "$5M – $25M":     15_000_000,
}
