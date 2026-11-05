"""idle unload delay on user_settings: how long an unused model stays loaded

Revision ID: c4d9e2a7f318
Revises: 717291e2cad1
Create Date: 2026-11-05 18:00:00.000000

Issue #619: the Settings card "Unload the model when idle" offers 5, 15, 30
and 60 minutes and Never. ``idle_unload_minutes`` holds the choice in minutes,
0 for Never, so the column is NOT NULL with a server default of 5: every
install, including one updated from a version without the setting, keeps the
five-minute delay it always had.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c4d9e2a7f318"
down_revision: Union[str, Sequence[str], None] = "717291e2cad1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_settings",
        sa.Column("idle_unload_minutes", sa.SmallInteger(), nullable=False, server_default="5"),
    )


def downgrade() -> None:
    op.drop_column("user_settings", "idle_unload_minutes")
