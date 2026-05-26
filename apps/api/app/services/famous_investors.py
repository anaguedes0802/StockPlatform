"""Curated snapshot of famous investors' top public 13F holdings.

Source: public 13F-HR filings on SEC EDGAR (e.g.
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001067983&type=13F-HR
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001336528&type=13F-HR
).

This is a **snapshot** intentionally bundled with the repo so the famous-investor
feature works without paid 13F API access. Refresh by editing this file or
implementing scripts/refresh_13f.py against EDGAR (PRs welcome).

Each entry has:
  - name: human-readable
  - fund: legal name of the filer
  - cik: SEC CIK number (so a refresh script can hit EDGAR directly)
  - philosophy: one-line investment style
  - filing_period: the quarter the holdings reflect
  - holdings: list of {symbol, weight_pct, change, note}

Weights are approximate as percent of the fund's reported 13F portfolio (not AUM).
"""
from __future__ import annotations

FAMOUS_INVESTORS: list[dict] = [
    {
        "name": "Warren Buffett",
        "fund": "Berkshire Hathaway",
        "cik": "0001067983",
        "philosophy": "Long-duration concentrated quality — wide moats, durable earnings, owner-operator alignment.",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "AAPL", "weight_pct": 24.0, "change": "trimmed", "note": "Largest position despite reductions"},
            {"symbol": "AXP",  "weight_pct": 15.4, "change": "held",    "note": "Long-term card franchise hold"},
            {"symbol": "BAC",  "weight_pct": 9.5,  "change": "trimmed", "note": "Reduced through 2024"},
            {"symbol": "KO",   "weight_pct": 9.0,  "change": "held",    "note": "Iconic perpetual hold"},
            {"symbol": "CVX",  "weight_pct": 6.4,  "change": "held",    "note": ""},
            {"symbol": "OXY",  "weight_pct": 5.0,  "change": "added",   "note": "Continued accumulation"},
            {"symbol": "MCO",  "weight_pct": 4.5,  "change": "held",    "note": ""},
            {"symbol": "KHC",  "weight_pct": 3.8,  "change": "held",    "note": ""},
            {"symbol": "CB",   "weight_pct": 2.7,  "change": "held",    "note": ""},
            {"symbol": "DVA",  "weight_pct": 1.9,  "change": "held",    "note": ""},
        ],
    },
    {
        "name": "Bill Ackman",
        "fund": "Pershing Square Capital",
        "cik": "0001336528",
        "philosophy": "Highly concentrated, activist long-only — 8-12 large positions.",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "CMG",  "weight_pct": 15.0, "change": "held",    "note": ""},
            {"symbol": "BN",   "weight_pct": 13.0, "change": "added",   "note": "Brookfield Corp."},
            {"symbol": "QSR",  "weight_pct": 11.0, "change": "held",    "note": ""},
            {"symbol": "HHH",  "weight_pct": 10.5, "change": "added",   "note": "Howard Hughes Holdings"},
            {"symbol": "HLT",  "weight_pct": 10.0, "change": "held",    "note": ""},
            {"symbol": "GOOG", "weight_pct": 8.5,  "change": "added",   "note": "Alphabet — added on AI overhang"},
            {"symbol": "GOOGL","weight_pct": 5.5,  "change": "added",   "note": ""},
            {"symbol": "CP",   "weight_pct": 8.5,  "change": "held",    "note": "Canadian Pacific Kansas City"},
            {"symbol": "NKE",  "weight_pct": 7.0,  "change": "added",   "note": "Reinitiated"},
            {"symbol": "BNRE", "weight_pct": 6.5,  "change": "added",   "note": "Brookfield Reinsurance"},
        ],
    },
    {
        "name": "Michael Burry",
        "fund": "Scion Asset Management",
        "cik": "0001649339",
        "philosophy": "Deep-value contrarian; small concentrated book; tactical macro hedges.",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "BABA", "weight_pct": 16.4, "change": "added",  "note": "China internet exposure"},
            {"symbol": "JD",   "weight_pct": 13.5, "change": "held",   "note": ""},
            {"symbol": "BIDU", "weight_pct": 9.0,  "change": "held",   "note": ""},
            {"symbol": "PDD",  "weight_pct": 8.5,  "change": "added",  "note": ""},
            {"symbol": "ACIC", "weight_pct": 7.0,  "change": "held",   "note": "American Coastal"},
            {"symbol": "MOH",  "weight_pct": 6.5,  "change": "held",   "note": "Molina Healthcare"},
            {"symbol": "OSCR", "weight_pct": 5.0,  "change": "added",  "note": ""},
            {"symbol": "BRKR", "weight_pct": 4.8,  "change": "held",   "note": ""},
            {"symbol": "HCA",  "weight_pct": 4.2,  "change": "held",   "note": ""},
        ],
    },
    {
        "name": "Stanley Druckenmiller",
        "fund": "Duquesne Family Office",
        "cik": "0001536411",
        "philosophy": "Top-down macro + concentrated equities; rapid rotation.",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "TEVA", "weight_pct": 9.0, "change": "added",  "note": "Major position"},
            {"symbol": "NTRA", "weight_pct": 8.0, "change": "added",  "note": "Natera"},
            {"symbol": "WDAY", "weight_pct": 6.5, "change": "added",  "note": ""},
            {"symbol": "MSFT", "weight_pct": 5.5, "change": "held",   "note": ""},
            {"symbol": "KMI",  "weight_pct": 5.0, "change": "added",  "note": "Kinder Morgan"},
            {"symbol": "WMB",  "weight_pct": 4.5, "change": "added",  "note": "Williams Companies"},
            {"symbol": "AMZN", "weight_pct": 4.0, "change": "held",   "note": ""},
            {"symbol": "PHIN", "weight_pct": 3.6, "change": "held",   "note": ""},
            {"symbol": "FLUT", "weight_pct": 3.2, "change": "added",  "note": "Flutter Entertainment"},
        ],
    },
    {
        "name": "Cathie Wood",
        "fund": "ARK Investment Management",
        "cik": "0001697748",
        "philosophy": "Disruptive innovation thematic — high-beta growth (genomics, AI, robotics, fintech).",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "TSLA", "weight_pct": 10.5, "change": "held",   "note": "Top thematic position"},
            {"symbol": "COIN", "weight_pct": 9.0,  "change": "added",  "note": ""},
            {"symbol": "ROKU", "weight_pct": 6.5,  "change": "added",  "note": ""},
            {"symbol": "PLTR", "weight_pct": 5.5,  "change": "trimmed","note": ""},
            {"symbol": "HOOD", "weight_pct": 5.0,  "change": "added",  "note": ""},
            {"symbol": "RBLX", "weight_pct": 4.5,  "change": "held",   "note": ""},
            {"symbol": "CRSP", "weight_pct": 4.0,  "change": "held",   "note": ""},
            {"symbol": "DKNG", "weight_pct": 3.5,  "change": "held",   "note": ""},
            {"symbol": "NTLA", "weight_pct": 3.0,  "change": "added",  "note": ""},
            {"symbol": "PATH", "weight_pct": 2.8,  "change": "held",   "note": ""},
        ],
    },
    {
        "name": "David Einhorn",
        "fund": "Greenlight Capital",
        "cik": "0001079114",
        "philosophy": "Value-oriented long/short; emphasis on capital allocation and inflation hedges.",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "GRBK", "weight_pct": 10.0, "change": "held",  "note": "Green Brick Partners"},
            {"symbol": "BLDR", "weight_pct": 8.5,  "change": "held",  "note": "Builders FirstSource"},
            {"symbol": "CNC",  "weight_pct": 7.5,  "change": "added", "note": "Centene"},
            {"symbol": "HPE",  "weight_pct": 7.0,  "change": "added", "note": ""},
            {"symbol": "PENN", "weight_pct": 5.5,  "change": "added", "note": ""},
            {"symbol": "PFGC", "weight_pct": 5.0,  "change": "held",  "note": "Performance Food"},
            {"symbol": "KD",   "weight_pct": 4.5,  "change": "held",  "note": "Kyndryl"},
            {"symbol": "VTRS", "weight_pct": 4.0,  "change": "held",  "note": ""},
        ],
    },
    {
        "name": "Seth Klarman",
        "fund": "Baupost Group",
        "cik": "0001061165",
        "philosophy": "Absolute-return value with margin of safety; willing to hold cash and special situations.",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "LBRDA", "weight_pct": 8.0, "change": "held",  "note": "Liberty Broadband"},
            {"symbol": "LBRDK", "weight_pct": 9.5, "change": "held",  "note": ""},
            {"symbol": "WBD",   "weight_pct": 6.5, "change": "added", "note": "Warner Bros. Discovery"},
            {"symbol": "VRT",   "weight_pct": 4.8, "change": "held",  "note": "Vertiv Holdings"},
            {"symbol": "META",  "weight_pct": 4.5, "change": "held",  "note": ""},
            {"symbol": "GOOGL", "weight_pct": 4.0, "change": "held",  "note": ""},
            {"symbol": "AMZN",  "weight_pct": 3.7, "change": "held",  "note": ""},
            {"symbol": "FTAI",  "weight_pct": 3.4, "change": "added", "note": "FTAI Aviation"},
        ],
    },
    {
        "name": "Howard Marks",
        "fund": "Oaktree Capital",
        "cik": "0001284812",
        "philosophy": "Distressed credit + opportunistic equity; cycle-aware contrarian.",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "TORM", "weight_pct": 9.0, "change": "held",  "note": "Tanker shipping"},
            {"symbol": "VST",  "weight_pct": 6.5, "change": "added", "note": "Vistra Energy"},
            {"symbol": "CHKP", "weight_pct": 4.5, "change": "held",  "note": ""},
            {"symbol": "FNV",  "weight_pct": 4.0, "change": "held",  "note": "Franco-Nevada"},
            {"symbol": "WPM",  "weight_pct": 3.8, "change": "added", "note": "Wheaton Precious Metals"},
        ],
    },
    {
        "name": "Chase Coleman (Tiger Global)",
        "fund": "Tiger Global Management",
        "cik": "0001167483",
        "philosophy": "Internet/tech growth crossover, public + private.",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "META", "weight_pct": 12.0, "change": "added",  "note": ""},
            {"symbol": "MSFT", "weight_pct": 9.0,  "change": "added",  "note": ""},
            {"symbol": "GOOGL","weight_pct": 8.0,  "change": "held",   "note": ""},
            {"symbol": "AMZN", "weight_pct": 7.5,  "change": "added",  "note": ""},
            {"symbol": "NVDA", "weight_pct": 7.0,  "change": "trimmed","note": ""},
            {"symbol": "FLUT", "weight_pct": 5.5,  "change": "added",  "note": ""},
            {"symbol": "TCOM", "weight_pct": 5.0,  "change": "held",   "note": "Trip.com"},
            {"symbol": "SE",   "weight_pct": 4.5,  "change": "added",  "note": "Sea Limited"},
            {"symbol": "MELI", "weight_pct": 4.0,  "change": "held",   "note": "MercadoLibre"},
        ],
    },
    {
        "name": "Ray Dalio",
        "fund": "Bridgewater Associates",
        "cik": "0001350694",
        "philosophy": "Risk-parity macro; broad diversification across asset classes via ETFs and large caps.",
        "filing_period": "2024-Q4 snapshot",
        "holdings": [
            {"symbol": "IEMG", "weight_pct": 8.0, "change": "added", "note": "EM equity ETF"},
            {"symbol": "SPY",  "weight_pct": 7.5, "change": "added", "note": ""},
            {"symbol": "IVV",  "weight_pct": 6.0, "change": "held",  "note": ""},
            {"symbol": "GLD",  "weight_pct": 4.5, "change": "added", "note": ""},
            {"symbol": "PG",   "weight_pct": 3.5, "change": "held",  "note": ""},
            {"symbol": "WMT",  "weight_pct": 3.4, "change": "held",  "note": ""},
            {"symbol": "JNJ",  "weight_pct": 3.2, "change": "held",  "note": ""},
            {"symbol": "KO",   "weight_pct": 3.0, "change": "held",  "note": ""},
        ],
    },
]
