"""compaction threshold setting and conversation level

Revision ID: a3e8c6f1d742
Revises: b7f1c4e9a25d
Create Date: 2026-10-12 10:00:00.000000

Issue #607: the share of the working window at which a conversation compacts
becomes a user-facing percentage, 50 to 90 in steps of 5.

1. ``user_settings.default_compaction_threshold`` - the GLOBAL default, 80:
   the threshold compaction fired at before the setting existed.
2. ``conversations.compaction_threshold`` - the per-conversation threshold.
   The application copies the global default at conversation creation;
   afterwards the conversation owns it (a later global change never
   retro-affects existing conversations), mirroring ``reasoning_effort``
   (b7f1c4e9a25d).

Both land as SMALLINT NOT NULL with a server default, so every existing row
backfills to 80 and keeps compacting exactly where it did. The range is
validated by the schemas and the entities, not by a database constraint,
like every other conversation setting.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a3e8c6f1d742"
down_revision: Union[str, Sequence[str], None] = "b7f1c4e9a25d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the global default and the per-conversation compaction threshold."""
    op.add_column(
        "user_settings",
        sa.Column(
            "default_compaction_threshold",
            sa.SmallInteger(),
            nullable=False,
            server_default="80",
        ),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "compaction_threshold",
            sa.SmallInteger(),
            nullable=False,
            server_default="80",
        ),
    )


def downgrade() -> None:
    """Drop both columns."""
    op.drop_column("conversations", "compaction_threshold")
    op.drop_column("user_settings", "default_compaction_threshold")
