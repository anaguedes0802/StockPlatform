from __future__ import annotations

from app.services import congress_ptr as cp

# Shape of pypdf "plain" text for an electronically filed House PTR, including
# the NUL bytes pypdf emits for the "Filing Status" / "Description" labels.
SAMPLE = """Filing ID #20035143
P        T           R
Clerk of the House of Representatives • Legislative Resource Center • B81 Cannon Building • Washington, DC 20515
F     I
Name: Hon. Nancy Pelosi
Status: Member
State/District: CA11
T
ID Owner Asset Transaction
Type
Date Notification
Date
Amount Cap.
Gains >
$200?
SP Bloom Energy Corporation Class A
Common Stock (BE) [ST]
P 07/24/2026 07/24/2026 $1,000,001 -
$5,000,000
F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00\x00: New
D\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00: Purchased 10,000 shares.
SP Bloom Energy Corporation Class A
Common Stock (BE) [OP]
P 07/24/2026 07/24/2026 $1,000,001 -
$5,000,000
F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00\x00: New
D\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00: Purchased 100 call options with a strike price of $100 and an expiration date of 6/17/27.
JT Apple Inc. (AAPL) [ST] S (partial) 06/30/2026 06/30/2026 $5,000,001 - $25,000,000
D\x00\x00\x00: Sold 31,600 shares.
SP REOF XXV, LLC [AB] P 07/27/2026 07/27/2026 $500,001 -
$1,000,000
D\x00\x00\x00: Additional investment in LLC which is acquiring a hotel property.
* For the complete list of asset type abbreviations, please visit https://fd.house.gov/reference/asset-type-codes.aspx.
I CERTIFY that the statements I have made on the attached Periodic Transaction Report are true.
"""


def test_parse_ptr_rows():
    rows = cp.parse_ptr_text(SAMPLE)
    assert [r["symbol"] for r in rows] == ["BE", "BE", "AAPL", None]
    be_stock, be_opt, aapl, reof = rows
    assert be_stock["owner"] == "SP" and be_stock["asset_type"] == "ST" and be_stock["side"] == "buy"
    assert be_stock["asset_name"] == "Bloom Energy Corporation Class A Common Stock"
    assert (be_stock["amount_min"], be_stock["amount_max"]) == (1_000_001, 5_000_000)
    assert be_stock["description"] == "Purchased 10,000 shares."
    assert be_opt["asset_type"] == "OP" and "call options" in be_opt["description"]
    assert aapl["owner"] == "JT" and aapl["side"] == "sell_partial"
    assert (aapl["amount_min"], aapl["amount_max"]) == (5_000_001, 25_000_000)
    assert aapl["traded_at"] == "2026-06-30" and aapl["notified_at"] == "2026-06-30"
    assert reof["asset_type"] == "AB" and reof["asset_name"] == "REOF XXV, LLC"
    assert reof["description"].startswith("Additional investment")


def test_parse_ptr_empty_text():
    assert cp.parse_ptr_text("") == []
    assert cp.parse_ptr_text("Scanned image, no text layer") == []


# Early-2020 electronic layout as pypdf extracts it: scrambled letter case,
# the type letter and the dates on separate lines, labels run together.
OLD_LAYOUT = (
    "P\nerioDic  t ranSaction  r ePort Clerk of the House of Re\npresentatives\n$200?\nsP\n"
    "alphabet Inc. - Cl ass a (googl)[oP]\nP\n02/27/2020 02/27/2020 $500,001 - $1,000,000\ng\nfedcF\n"
    "IlINg  s TaTus : New D\nEsCRIPTIoN : Purchased 40 call options with a strike price of $1200.sP\n"
    "amazon.com, Inc. (aMZN) [sT] s 01/16/2020 01/16/2020 $250,001 - $500,000\ng\nfedcF\n"
    "IlINg  s TaTus : New D\nEsCRIPTIoN : sold 20 call options.\n"
)


def test_parse_old_layout():
    rows = cp.parse_ptr_text(OLD_LAYOUT)
    assert [(r["symbol"], r["side"], r["owner"]) for r in rows] == [("GOOGL", "buy", "SP"), ("AMZN", "sell", "SP")]
    assert rows[0]["asset_type"] == "OP"
    assert rows[0]["description"].startswith("Purchased 40 call options")
    assert rows[1]["asset_name"] == "amazon.com, Inc."


HERN_LIKE = """$200?
JT Boston Scientific Corporation
Common Stock (BSX) [ST]
S (partial) 09/04/2026 09/15/2026 $50,001 -
$100,000
F S: New
S O: Hern Family Revocable Trust
CenterPoint Energy, Inc (CNP) [ST] S 09/15/2026 09/15/2026 $15,001 -
$50,000
F S: New
S O: Kevin Hern Traditional IRA
"""


def test_subholding_lines_do_not_leak_into_next_asset():
    rows = cp.parse_ptr_text(HERN_LIKE)
    assert [r["symbol"] for r in rows] == ["BSX", "CNP"]
    assert rows[0]["subholding"] == "Hern Family Revocable Trust"
    assert rows[1]["asset_name"] == "CenterPoint Energy, Inc" and rows[1]["owner"] == ""
    assert rows[1]["subholding"] == "Kevin Hern Traditional IRA"
