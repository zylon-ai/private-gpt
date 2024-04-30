#!/bin/sh


# Initialize alembic ini
# echo "Apply makemigrations "
# alembic init alembic

# Apply database migrations
echo "Apply makemigrations "
alembic revision --autogenerate

# Apply database migrations
echo "Apply database migrations"
alembic upgrade head

# Start server
echo "Starting server"
python -m private_gpt