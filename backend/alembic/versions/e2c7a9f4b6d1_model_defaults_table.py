"""model_defaults table: the conversation defaults saved for a model

Revision ID: e2c7a9f4b6d1
Revises: a3e8c6f1d742
Create Date: 2026-11-05 10:00:00.000000

Issue #609: ``/config global`` saves a conversation's values as the defaults
of its model. One row per model at most, keyed by ``llm_id`` (primary key and
foreign key to ``llms.id`` with ON DELETE CASCADE, so deleting the model
deletes its row). Every value column is nullable and NULL means "not saved":
the value then comes from the user settings or from the model's own
recommendation. The value rules are enforced by the schemas and the entity,
not by database constraints, like every other conversation setting.

The table starts empty: an existing install has nothing saved and every new
conversation keeps opening exactly as before.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e2c7a9f4b6d1"
down_revision: Union[str, Sequence[str], None] = "a3e8c6f1d742"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the model_defaults table."""
    op.create_table(
        "model_defaults",
        sa.Column("llm_id", sa.Integer(), nullable=False),
        sa.Column("reasoning_effort", sa.String(length=8), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("top_p", sa.Float(), nullable=True),
        sa.Column("compaction_threshold", sa.SmallInteger(), nullable=True),
        sa.Column("web_search_enabled", sa.Boolean(), nullable=True),
        sa.Column("custom_prompt", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["llm_id"], ["llms.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("llm_id"),
    )


def downgrade() -> None:
    """Drop the table (every saved default is lost)."""
    op.drop_table("model_defaults")
