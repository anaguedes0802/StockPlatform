"""users.totp_enabled: 2FA is enforced only after the first verified code

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-05
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("totp_enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    # Until now any stored secret was enforced at login, so those users keep 2FA on.
    users = sa.table("users", sa.column("totp_secret", sa.String), sa.column("totp_enabled", sa.Boolean))
    op.execute(users.update().where(users.c.totp_secret.isnot(None)).values(totp_enabled=True))


def downgrade() -> None:
    op.drop_column("users", "totp_enabled")
