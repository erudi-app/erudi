"""add inference_backend to user_settings

The inference backend is an app-wide setting persisted next to the web-search
default, the interface language and the automatic-update preference. It holds
"auto" (hardware detection picks the engine) or "cpu" (run models on the CPU
build even when an NVIDIA GPU is present). The column is NOT NULL with a server
default of 'auto' so an install that predates this revision keeps selecting its
engine from the hardware exactly as it did.

Revision ID: c1d5b83f9a24
Revises: a9f4c1b7e206
Create Date: 2026-09-06

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1d5b83f9a24"
down_revision: Union[str, Sequence[str], None] = "a9f4c1b7e206"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_settings",
        sa.Column(
            "inference_backend",
            sa.String(length=8),
            nullable=False,
            server_default="auto",
        ),
    )


def downgrade() -> None:
    op.drop_column("user_settings", "inference_backend")
