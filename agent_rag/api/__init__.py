"""Agent RAG API - FastAPI-based REST API for Agent RAG services."""

from agent_rag.api.main import app, create_app

__all__ = ["app", "create_app"]
