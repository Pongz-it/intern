#!/usr/bin/env python3
"""Hatchet worker for Agent RAG ingestion workflow.

Run this script to start a worker that processes ingestion workflows.

Usage:
    python run_worker.py
"""

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Start the Hatchet worker."""
    logger.info("Starting Hatchet worker for Agent RAG ingestion...")

    # Import Hatchet client
    from hatchet_client import hatchet

    # Import the ingestion workflow to register it
    from agent_rag.ingestion.workflow.ingestion_workflow import ingestion_workflow

    logger.info(f"Registered workflow: {ingestion_workflow}")

    # Create and start worker
    worker = hatchet.worker(
        "agent-rag-ingestion-worker",
        workflows=[ingestion_workflow],
    )

    logger.info("Worker starting... Press Ctrl+C to stop")

    # Start the worker (blocking call)
    worker.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
