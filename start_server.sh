#!/bin/bash
# Agent RAG API Server Startup Script

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8005}
WORKERS=${WORKERS:-1}

echo "=================================================="
echo "  Agent RAG API Server"
echo "=================================================="
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "=================================================="
echo ""

# Start the server
if [ "$WORKERS" -gt 1 ]; then
    echo "Starting with $WORKERS workers..."
    uvicorn agent_rag.api.main:app --host $HOST --port $PORT --workers $WORKERS
else
    echo "Starting in development mode (reload enabled)..."
    uvicorn agent_rag.api.main:app --host $HOST --port $PORT --reload
fi
