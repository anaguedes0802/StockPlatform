"""ai_gate_decisions: forward-test log of the LLM risk-gate

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-29
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The app also creates this table on first write (gate_log.ensure_table),
    # so skip if it already exists.
    if "ai_gate_decisions" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "ai_gate_decisions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("as_of", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("bot_id", sa.String(64)),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("size_multiplier", sa.Numeric(8, 4)),
        sa.Column("conviction", sa.Numeric(8, 4)),
        sa.Column("provider", sa.String(64)),
        sa.Column("rationale", sa.Text()),
        sa.Column("key_risks", postgresql.JSONB()),
        sa.Column("signal_price", sa.Numeric(20, 8)),
        sa.Column("stop_dist", sa.Numeric(20, 8)),
        sa.Column("entry_price", sa.Numeric(20, 8)),
        sa.Column("ret_5", sa.Numeric(12, 6)),
        sa.Column("ret_10", sa.Numeric(12, 6)),
        sa.Column("ret_20", sa.Numeric(12, 6)),
        sa.Column("trade_r", sa.Numeric(12, 6)),
        sa.Column("trade_return_pct", sa.Numeric(12, 6)),
        sa.Column("trade_exit_reason", sa.String(32)),
        sa.Column("trade_bars", sa.Integer()),
        sa.Column("scored_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_ai_gate_decisions_created_at", "ai_gate_decisions", ["created_at"])
    op.create_index("ix_ai_gate_decisions_symbol", "ai_gate_decisions", ["symbol"])
    op.create_index("ix_ai_gate_decisions_decision", "ai_gate_decisions", ["decision"])


def downgrade() -> None:
    op.drop_table("ai_gate_decisions")
