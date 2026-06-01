"""trading bot live-execution fields

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-01
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("trading_bots",
                  sa.Column("mode", sa.String(8), nullable=False, server_default="paper"))
    op.add_column("trading_bots",
                  sa.Column("armed", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("trading_bots",
                  sa.Column("execution", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")))
    op.add_column("trading_bots",
                  sa.Column("run_state", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")))


def downgrade() -> None:
    op.drop_column("trading_bots", "run_state")
    op.drop_column("trading_bots", "execution")
    op.drop_column("trading_bots", "armed")
    op.drop_column("trading_bots", "mode")
