"""Load `apps/api/swing_agent.toml` into a plain dict plus a few typed helpers.

The file is the single source of truth. `config_hash()` fingerprints it so
every experiment-log row and every artifact records exactly which settings
produced it.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

API_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = API_DIR / "swing_agent.toml"


@lru_cache(maxsize=4)
def load(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    with p.open("rb") as f:
        return tomllib.load(f)


def config_hash(cfg: dict[str, Any] | None = None) -> str:
    blob = json.dumps(cfg or load(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def root(cfg: dict[str, Any] | None = None) -> Path:
    r = Path((cfg or load())["data"]["root"])
    return r if r.is_absolute() else API_DIR / r


def segment(name: str, cfg: dict[str, Any] | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) dates of a split: warmup | train | validation | test | research."""
    s = (cfg or load())["splits"]
    if name == "research":
        return pd.Timestamp(s["train"][0]), pd.Timestamp(s["validation"][1])
    lo, hi = s[name]
    return pd.Timestamp(lo), pd.Timestamp(hi)
