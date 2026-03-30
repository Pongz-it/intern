"""Hatchet client initialization for Agent RAG workflows.

This module initializes the Hatchet SDK client for workflow orchestration.
"""

import os
import logging

from dotenv import load_dotenv
from hatchet_sdk import Hatchet

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Hatchet client
# The SDK automatically reads configuration from environment variables:
# - HATCHET_SERVER_URL: Hatchet server URL (default: http://localhost:7077)
# - HATCHET_CLIENT_TLS_STRATEGY: TLS strategy (none, tls, mtls)
# - HATCHET_CLIENT_TOKEN: Optional authentication token

try:
    hatchet = Hatchet(
        debug=os.getenv("HATCHET_DEBUG", "false").lower() == "true",
    )
    logger.info(f"Hatchet client initialized: server={os.getenv('HATCHET_SERVER_URL', 'http://localhost:7077')}")
except Exception as e:
    logger.error(f"Failed to initialize Hatchet client: {e}")
    raise


# Export the client
__all__ = ["hatchet"]
