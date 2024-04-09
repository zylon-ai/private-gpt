"""create chat history and items

Revision ID: 9957402017dc
Revises: 
Create Date: 2024-04-04 11:31:53.261330

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9957402017dc'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('chat_history',
    sa.Column('conversation_id', sa.UUID(), nullable=False),
    sa.Column('title', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('conversation_id')
    )
    op.create_table('chat_items',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sender', sa.String(length=225), nullable=False),
    sa.Column('content', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('like', sa.Boolean(), nullable=True),
    sa.Column('conversation_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['conversation_id'], ['chat_history.conversation_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # op.create_unique_constraint('unique_user_role', 'user_roles', ['user_id', 'role_id', 'company_id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # op.drop_constraint('unique_user_role', 'user_roles', type_='unique')
    op.drop_table('chat_items')
    op.drop_table('chat_history')
    # ### end Alembic commands ###
