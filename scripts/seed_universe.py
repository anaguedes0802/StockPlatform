"""Seed the `instruments` table from the bundled universe.

Usage (from apps/api with the venv active):
    python ../../scripts/seed_universe.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "apps/api")

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import Instrument  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services.universe_data import all_instruments  # noqa: E402


def main() -> None:
    rows = [
        {
            "symbol": sym,
            "name": name,
            "exchange": exchange,
            "asset_class": asset_class,
            "sector": sector,
            "is_active": True,
        }
        for sym, name, exchange, asset_class, sector in all_instruments()
    ]
    with SessionLocal() as db:
        stmt = pg_insert(Instrument).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "name": stmt.excluded.name,
                "exchange": stmt.excluded.exchange,
                "asset_class": stmt.excluded.asset_class,
                "sector": stmt.excluded.sector,
                "is_active": True,
            },
        )
        db.execute(stmt)
        db.commit()
        print(f"upserted {len(rows)} instruments")


if __name__ == "__main__":
    main()
