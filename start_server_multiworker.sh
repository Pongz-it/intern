#!/bin/bash
# Agent RAG API Server - Multi-Worker Startup Script
# Usage: HOST=0.0.0.0 PORT=8005 WORKERS=8 ./start_server_multiworker.sh

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8005}
WORKERS=${WORKERS:-4}

echo "=================================================="
echo "  Agent RAG API Server (Production Mode)"
echo "=================================================="
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  PID File: /tmp/agent_rag_server.pid"
echo "=================================================="
echo ""

# Start with gunicorn for production
if command -v gunicorn &> /dev/null; then
    echo "Starting with Gunicorn..."
    gunicorn agent_rag.api.main:app \
        --bind $HOST:$PORT \
        --workers $WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout 120 \
        --keep-alive 5 \
        --access-logfile - \
        --error-logfile - \
        --pid /tmp/agent_rag_server.pid
else
    echo "Gunicorn not found, falling back to uvicorn..."
    uvicorn agent_rag.api.main:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS
fi
