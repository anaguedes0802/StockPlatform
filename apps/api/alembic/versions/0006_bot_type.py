"""bot_type for trader vs investor bots

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-01
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("trading_bots",
                  sa.Column("bot_type", sa.String(16), nullable=False, server_default="trader"))


def downgrade() -> None:
    op.drop_column("trading_bots", "bot_type")
