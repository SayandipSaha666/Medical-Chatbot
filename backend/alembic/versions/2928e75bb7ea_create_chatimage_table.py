"""Create ChatImage table

Revision ID: 2928e75bb7ea
Revises: 
Create Date: 2025-12-28 20:22:12.172389

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2928e75bb7ea'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "chat_images",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column(
            "user_id",
            sa.Integer,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chat_id",
            sa.Integer,
            sa.ForeignKey("chats.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "message_id",
            sa.Integer,
            sa.ForeignKey("messages.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("imagekit_file_id", sa.String, nullable=False),
        sa.Column("image_url", sa.String, nullable=False),
        sa.Column("thumbnail_url", sa.String, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_chat_images_user_id",
        "chat_images",
        ["user_id"],
    )

    op.create_index(
        "ix_chat_images_chat_id",
        "chat_images",
        ["chat_id"],
    )

    op.create_index(
        "ix_chat_images_message_id",
        "chat_images",
        ["message_id"],
    )

    pass


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_chat_images_message_id", table_name="chat_images")
    op.drop_index("ix_chat_images_chat_id", table_name="chat_images")
    op.drop_index("ix_chat_images_user_id", table_name="chat_images")
    op.drop_table("chat_images")
    pass
