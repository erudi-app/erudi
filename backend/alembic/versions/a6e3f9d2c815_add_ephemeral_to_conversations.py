"""ephemeral flag on conversations: a conversation deleted when the user leaves it

Revision ID: a6e3f9d2c815
Revises: c4d9e2a7f318
Create Date: 2026-11-17 12:00:00.000000

Issue #428: a conversation started with the ephemeral toggle is never listed
or searched and is deleted when the user leaves it, when the app stops, or at
the next start after a crash. ``ephemeral`` marks it; the column is NOT NULL
with a server default of false, so every conversation that predates the
feature stays a regular one.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a6e3f9d2c815"
down_revision: Union[str, Sequence[str], None] = "c4d9e2a7f318"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("ephemeral", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("conversations", "ephemeral")
