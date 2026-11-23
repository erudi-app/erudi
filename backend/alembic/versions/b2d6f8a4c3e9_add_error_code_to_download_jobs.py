"""add error_code to download_jobs

Revision ID: b2d6f8a4c3e9
Revises: a6e3f9d2c815
Create Date: 2026-11-23 12:00:00.000000

Issue #780: a failed download stores a stable code beside its English
``error_message`` (``offline``, ``disk_full``, ``gated``, ...), which the
interface turns into a sentence in its own language. The code is derived from
the exception where the failure is recorded, and that exception is gone once
the row is written, so it is a column rather than something read back. It is
nullable with no default: a job that failed before this revision has no code,
and the interface shows its generic sentence for it.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b2d6f8a4c3e9"
down_revision: Union[str, Sequence[str], None] = "a6e3f9d2c815"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("download_jobs", sa.Column("error_code", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("download_jobs", "error_code")
