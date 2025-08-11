#!/bin/bash
set -e

# Wait for database to be ready
echo "Waiting for database to be ready..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "Database is ready!"

# Run database migrations
echo "Running database migrations..."
cd /app
python -m alembic upgrade head

# Initialize knowledge base if needed
echo "Initializing knowledge base..."
python -c "
from src.que_agents.utils.data_populator import DataPopulator
try:
    populator = DataPopulator()
    populator.populate_all()
    print('Knowledge base initialized successfully')
except Exception as e:
    print(f'Knowledge base initialization failed: {e}')
"

# Start the application
echo "Starting Que Agents application..."
exec "$@"