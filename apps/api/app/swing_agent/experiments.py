"""Experiment registry: every variant ever evaluated, one JSON line each.

Phase 5 deflates Sharpe ratios by the number of variants tried, so this file
must hold *all* of them, including the ones that looked bad. Each row records
the component, the variant's parameters, the data segment it saw, its
metrics, the config hash and a hash of the swing_agent code that ran it.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.swing_agent import config

_PKG = Path(__file__).resolve().parent


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(_PKG.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def path(root: Path | None = None) -> Path:
    return (root or config.root()) / "experiments.jsonl"


def log(phase: int, component: str, variant: str, params: dict[str, Any], segment: str,
        metrics: dict[str, Any], *, root: Path | None = None, notes: str = "") -> dict[str, Any]:
    rec = {"ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "phase": phase,
           "component": component, "variant": variant, "segment": segment, "params": params,
           "metrics": metrics, "config_hash": config.config_hash(), "code_hash": code_hash(),
           "notes": notes}
    p = path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def read(root: Path | None = None) -> list[dict[str, Any]]:
    p = path(root)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def count(component: str | None = None, root: Path | None = None) -> int:
    """Distinct (component, variant, params) tried — the multiple-testing N."""
    seen = {(r["component"], r["variant"], json.dumps(r["params"], sort_keys=True, default=str))
            for r in read(root) if component is None or r["component"] == component}
    return len(seen)
