"""reasoning effort setting and conversation level

Revision ID: b7f1c4e9a25d
Revises: c1d5b83f9a24
Create Date: 2026-09-15 09:00:00.000000

Release 1.1.2: how much a model may deliberate before answering becomes a
user-facing level with five values (none / low / medium / high / xhigh).

1. ``user_settings.default_reasoning_effort`` - the GLOBAL default, "medium":
   the level at which a model reasoning naturally behaves exactly as it did
   before the setting existed.
2. ``conversations.reasoning_effort`` - the per-conversation level. The
   application copies the global default at conversation creation; afterwards
   the conversation owns its level (a later global change never retro-affects
   existing conversations), mirroring ``web_search_enabled`` (a7c5e2d9f4b1).

Both land NOT NULL with a server default, so every existing row backfills to
"medium" and the API never serves a null level.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7f1c4e9a25d"
down_revision: Union[str, Sequence[str], None] = "c1d5b83f9a24"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the global default and the per-conversation reasoning effort."""
    op.add_column(
        "user_settings",
        sa.Column(
            "default_reasoning_effort",
            sa.String(length=8),
            nullable=False,
            server_default="medium",
        ),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "reasoning_effort",
            sa.String(length=8),
            nullable=False,
            server_default="medium",
        ),
    )


def downgrade() -> None:
    """Drop both columns."""
    op.drop_column("conversations", "reasoning_effort")
    op.drop_column("user_settings", "default_reasoning_effort")
