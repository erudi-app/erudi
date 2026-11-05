"""update opt-ins on user_settings: download and install are separate choices

Revision ID: 717291e2cad1
Revises: e2c7a9f4b6d1
Create Date: 2026-11-05 16:30:00.000000

Issue #563: the update switch splits into three. ``auto_update_enabled`` stays
and is the "Check for updates" switch (notify only), with the value it already
holds. ``auto_download_updates`` ("Download updates automatically") and
``auto_install_updates`` ("Install downloaded updates when I quit") are new,
NOT NULL with a server default of false: every install, including one updated
from a version where automatic updates downloaded and installed on their own,
starts with both off, so downloading and installing are explicit consent.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "717291e2cad1"
down_revision: Union[str, Sequence[str], None] = "e2c7a9f4b6d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_settings",
        sa.Column("auto_download_updates", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "user_settings",
        sa.Column("auto_install_updates", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("user_settings", "auto_install_updates")
    op.drop_column("user_settings", "auto_download_updates")
