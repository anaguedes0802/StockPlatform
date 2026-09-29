"""Full-history insider transactions from the SEC's quarterly bulk data sets.

The platform's older Form 4 adapter (`edgar_form4`) reads EDGAR full-text
search, which returns at most the 100 most recent filings per company: about
8 months of history for JPMorgan and 15 for Pfizer. Anything earlier was
zero, so no past test ever saw real insider data.

The SEC's "Insider Transactions Data Sets" (Forms 3/4/5, 2006 onward,
published quarterly) contain every filing. This module downloads each
quarter once, keeps only open-market purchases (code P) and sales (code S) of
non-derivative securities from original Form 4s, and stores a compact
per-quarter gzip table under `artifacts/insider_bulk/` (gitignored). The
~12 MB zip is deleted after extraction.

Point-in-time: every row carries FILING_DATE, the day the trade became
public. Anything that trades on insider data must key off that date, never
TRANS_DATE.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from app.core.logging import log

_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "insider_bulk"
_INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
_SEC = "https://www.sec.gov"
_UA = "StockPlatform/0.1 contact@example.com"

_SUB_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME",
             "ISSUERTRADINGSYMBOL", "AFF10B5ONE"]
_OWN_COLS = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]
_TRN_COLS = ["ACCESSION_NUMBER", "SECURITY_TITLE", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
             "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS",
             "DIRECT_INDIRECT_OWNERSHIP"]


def _get(url: str, *, timeout: float = 120.0, tries: int = 4) -> httpx.Response:
    last: Exception | None = None
    for i in range(tries):
        try:
            r = httpx.get(url, headers={"User-Agent": _UA}, timeout=timeout, follow_redirects=True)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except httpx.HTTPError as e:
            last = e
        time.sleep(1.5 * (i + 1))  # SEC fair-access: be gentle on retries
    raise RuntimeError(str(last))


def quarter_links() -> dict[str, str]:
    """{"2026q2": absolute_url, ...} scraped from the SEC data-set page."""
    html = _get(_INDEX_URL, timeout=30).text
    out = {}
    for path in re.findall(r'href="([^"]*?(\d{4}q[1-4])_form345\.zip)"', html):
        out[path[1]] = _SEC + path[0] if path[0].startswith("/") else path[0]
    return dict(sorted(out.items()))


def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    with z.open(name) as fh:
        # QUOTE_NONE: free-text fields contain stray quotes; tabs are the only delimiter.
        return pd.read_csv(fh, sep="\t", dtype=str, usecols=lambda c: c in cols, quoting=3,
                           on_bad_lines="skip", encoding_errors="replace")


def extract_quarter(raw: bytes) -> pd.DataFrame:
    """Parse one quarterly zip into P/S transaction rows (one row per owner)."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    trn = _read(z, "NONDERIV_TRANS.tsv", _TRN_COLS)
    trn = trn[trn["TRANS_CODE"].isin(["P", "S"])]
    sub = _read(z, "SUBMISSION.tsv", _SUB_COLS)
    sub = sub[sub["DOCUMENT_TYPE"] == "4"]          # originals only: amendments would double count
    own = _read(z, "REPORTINGOWNER.tsv", _OWN_COLS)
    df = trn.merge(sub, on="ACCESSION_NUMBER", how="inner").merge(own, on="ACCESSION_NUMBER", how="left")
    out = pd.DataFrame({
        "accession": df["ACCESSION_NUMBER"],
        "filing_date": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
        "trans_date": pd.to_datetime(df["TRANS_DATE"], format="%d-%b-%Y", errors="coerce"),
        "issuer_cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce").astype("Int64"),
        "issuer_name": df["ISSUERNAME"],
        "issuer_symbol": df["ISSUERTRADINGSYMBOL"].str.strip().str.upper(),
        "owner_cik": pd.to_numeric(df["RPTOWNERCIK"], errors="coerce").astype("Int64"),
        "owner_name": df["RPTOWNERNAME"],
        "relationship": df["RPTOWNER_RELATIONSHIP"].fillna(""),
        "title": df["RPTOWNER_TITLE"].fillna(""),
        "code": df["TRANS_CODE"],
        "acq_disp": df["TRANS_ACQUIRED_DISP_CD"],
        "security": df["SECURITY_TITLE"].fillna(""),
        "shares": pd.to_numeric(df["TRANS_SHARES"], errors="coerce"),
        "price": pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce"),
        "owned_after": pd.to_numeric(df["SHRS_OWND_FOLWNG_TRANS"], errors="coerce"),
        "direct": df["DIRECT_INDIRECT_OWNERSHIP"],
        # The 10b5-1 checkbox exists only on filings from 2023 on.
        "planned_10b5_1": (df["AFF10B5ONE"].str.lower().isin(["1", "true"]) if "AFF10B5ONE" in df
                           else pd.Series(False, index=df.index)),
    })
    out["value"] = out["shares"] * out["price"]
    out = out.dropna(subset=["filing_date", "issuer_cik"])
    return out.reset_index(drop=True)


def _fetch_quarter(q: str, url: str, log_fn=None) -> tuple[str, int, float]:
    """Stream one quarterly zip to a temp file, extract, store, delete the zip."""
    tmp = _DIR / f"{q}.zip.part"
    last: Exception | None = None
    for attempt in range(4):
        try:
            with httpx.stream("GET", url, headers={"User-Agent": _UA}, timeout=60.0,
                              follow_redirects=True) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            raw = tmp.read_bytes()
            df = extract_quarter(raw)
            df.to_pickle(_DIR / f"{q}.pkl.gz", compression="gzip")
            tmp.unlink(missing_ok=True)
            if log_fn:
                log_fn(f"{q}: {len(df):,} P/S rows ({len(raw) / 1e6:.1f} MB)")
            return q, len(df), len(raw) / 1e6
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"{q}: {last}")


def sync(*, quarters: list[str] | None = None, verbose: bool = False, workers: int = 3,
         log_path: str | None = None) -> dict[str, Any]:
    """Download and extract any quarters not yet stored. Idempotent.

    A few parallel streams (well under the SEC's 10 requests/second fair-access
    limit); progress lines go to stdout and/or `log_path` as each quarter lands.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _DIR.mkdir(parents=True, exist_ok=True)
    links = quarter_links()
    todo = [q for q in (quarters or links) if q in links and not (_DIR / f"{q}.pkl.gz").exists()]

    def log_fn(msg: str) -> None:
        if verbose:
            print(msg, flush=True)
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(msg + "\n")

    done, failed, rows, mb = [], {}, 0, 0.0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_fetch_quarter, q, links[q], log_fn): q for q in todo}
        for f in as_completed(futs):
            q = futs[f]
            try:
                _, n, size = f.result()
                done.append(q)
                rows += n
                mb += size
            except Exception as e:  # noqa: BLE001 — the next sync retries it
                failed[q] = str(e)[:200]
                log_fn(f"{q}: FAILED {str(e)[:120]}")
                log.warning("insider_bulk_quarter_failed", quarter=q, err=str(e)[:200])
    return {"available": len(links), "downloaded": sorted(done), "failed": failed,
            "new_rows": rows, "downloaded_mb": round(mb, 1),
            "stored": sorted(p.name.split(".")[0] for p in _DIR.glob("*.pkl.gz"))}


def load_transactions(*, start: str | None = None) -> pd.DataFrame:
    """All stored P/S rows, oldest first. Empty frame if nothing synced yet."""
    files = sorted(_DIR.glob("*.pkl.gz"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_pickle(f, compression="gzip") for f in files], ignore_index=True)
    # A filing can appear in two quarterly files around a boundary.
    df = df.drop_duplicates(subset=["accession", "owner_cik", "trans_date", "code", "shares", "price"])
    if start:
        df = df[df["filing_date"] >= pd.Timestamp(start)]
    return df.sort_values(["filing_date", "issuer_cik"]).reset_index(drop=True)


def is_officer_or_director(relationship: pd.Series) -> pd.Series:
    r = relationship.fillna("")
    return r.str.contains("Director") | r.str.contains("Officer")
