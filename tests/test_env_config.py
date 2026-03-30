"""Tests for environment configuration loading."""

import os
import pytest
from pathlib import Path
import tempfile

from agent_rag.core.env_config import (
    load_dotenv,
    get_config_from_env,
    get_llm_config_from_env,
    get_embedding_config_from_env,
    get_document_index_config_from_env,
    get_vespa_schema_config_from_env,
    get_search_config_from_env,
    get_agent_config_from_env,
    _get_env_bool,
    _get_env_int,
    _get_env_float,
    _get_env_list,
)


class TestEnvHelpers:
    """Tests for environment variable helper functions."""

    def test_get_env_bool_true_values(self):
        os.environ["TEST_BOOL"] = "true"
        assert _get_env_bool("TEST_BOOL") is True

        os.environ["TEST_BOOL"] = "1"
        assert _get_env_bool("TEST_BOOL") is True

        os.environ["TEST_BOOL"] = "yes"
        assert _get_env_bool("TEST_BOOL") is True

        os.environ["TEST_BOOL"] = "on"
        assert _get_env_bool("TEST_BOOL") is True

        del os.environ["TEST_BOOL"]

    def test_get_env_bool_false_values(self):
        os.environ["TEST_BOOL"] = "false"
        assert _get_env_bool("TEST_BOOL") is False

        os.environ["TEST_BOOL"] = "0"
        assert _get_env_bool("TEST_BOOL") is False

        os.environ["TEST_BOOL"] = "no"
        assert _get_env_bool("TEST_BOOL") is False

        del os.environ["TEST_BOOL"]

    def test_get_env_bool_default(self):
        assert _get_env_bool("NONEXISTENT_BOOL", False) is False
        assert _get_env_bool("NONEXISTENT_BOOL", True) is True

    def test_get_env_int(self):
        os.environ["TEST_INT"] = "42"
        assert _get_env_int("TEST_INT", 0) == 42
        del os.environ["TEST_INT"]

    def test_get_env_int_invalid(self):
        os.environ["TEST_INT"] = "not_a_number"
        assert _get_env_int("TEST_INT", 100) == 100
        del os.environ["TEST_INT"]

    def test_get_env_int_default(self):
        assert _get_env_int("NONEXISTENT_INT", 99) == 99

    def test_get_env_float(self):
        os.environ["TEST_FLOAT"] = "3.14"
        assert _get_env_float("TEST_FLOAT", 0.0) == 3.14
        del os.environ["TEST_FLOAT"]

    def test_get_env_float_invalid(self):
        os.environ["TEST_FLOAT"] = "not_a_float"
        assert _get_env_float("TEST_FLOAT", 2.5) == 2.5
        del os.environ["TEST_FLOAT"]

    def test_get_env_list(self):
        os.environ["TEST_LIST"] = "a,b,c"
        assert _get_env_list("TEST_LIST") == ["a", "b", "c"]
        del os.environ["TEST_LIST"]

    def test_get_env_list_with_spaces(self):
        os.environ["TEST_LIST"] = "a, b, c"
        assert _get_env_list("TEST_LIST") == ["a", "b", "c"]
        del os.environ["TEST_LIST"]

    def test_get_env_list_default(self):
        assert _get_env_list("NONEXISTENT_LIST", ["x", "y"]) == ["x", "y"]


class TestLoadDotenv:
    """Tests for .env file loading."""

    def test_load_dotenv_simple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("TEST_VAR=hello\n")

            # Clear if exists
            if "TEST_VAR" in os.environ:
                del os.environ["TEST_VAR"]

            load_dotenv(env_path)
            assert os.environ.get("TEST_VAR") == "hello"

            del os.environ["TEST_VAR"]

    def test_load_dotenv_with_quotes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text('QUOTED_VAR="hello world"\n')

            if "QUOTED_VAR" in os.environ:
                del os.environ["QUOTED_VAR"]

            load_dotenv(env_path)
            assert os.environ.get("QUOTED_VAR") == "hello world"

            del os.environ["QUOTED_VAR"]

    def test_load_dotenv_skips_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("# This is a comment\nVALID_VAR=value\n")

            if "VALID_VAR" in os.environ:
                del os.environ["VALID_VAR"]

            load_dotenv(env_path)
            assert os.environ.get("VALID_VAR") == "value"

            del os.environ["VALID_VAR"]

    def test_load_dotenv_does_not_override(self):
        """Existing env vars should not be overwritten."""
        os.environ["EXISTING_VAR"] = "original"

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("EXISTING_VAR=from_file\n")

            load_dotenv(env_path)
            assert os.environ.get("EXISTING_VAR") == "original"

            del os.environ["EXISTING_VAR"]


class TestLLMConfig:
    """Tests for LLM configuration from environment."""

    def test_llm_config_defaults(self):
        # Clear relevant env vars
        for key in list(os.environ.keys()):
            if key.startswith("AGENT_RAG_LLM_"):
                del os.environ[key]

        config = get_llm_config_from_env()
        assert config.model == "gpt-4o"
        assert config.provider == "litellm"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0

    def test_llm_config_from_env(self):
        os.environ["AGENT_RAG_LLM_MODEL"] = "claude-3-opus"
        os.environ["AGENT_RAG_LLM_PROVIDER"] = "anthropic"
        os.environ["AGENT_RAG_LLM_MAX_TOKENS"] = "8192"
        os.environ["AGENT_RAG_LLM_TEMPERATURE"] = "0.7"

        config = get_llm_config_from_env()
        assert config.model == "claude-3-opus"
        assert config.provider == "anthropic"
        assert config.max_tokens == 8192
        assert config.temperature == 0.7

        # Cleanup
        for key in ["AGENT_RAG_LLM_MODEL", "AGENT_RAG_LLM_PROVIDER",
                    "AGENT_RAG_LLM_MAX_TOKENS", "AGENT_RAG_LLM_TEMPERATURE"]:
            del os.environ[key]


class TestEmbeddingConfig:
    """Tests for embedding configuration from environment."""

    def test_embedding_config_defaults(self):
        for key in list(os.environ.keys()):
            if key.startswith("AGENT_RAG_EMBEDDING_"):
                del os.environ[key]

        config = get_embedding_config_from_env()
        assert config.model == "text-embedding-3-small"
        assert config.dimension == 1536
        assert config.batch_size == 32

    def test_embedding_config_from_env(self):
        os.environ["AGENT_RAG_EMBEDDING_MODEL"] = "text-embedding-ada-002"
        os.environ["AGENT_RAG_EMBEDDING_DIMENSION"] = "768"

        config = get_embedding_config_from_env()
        assert config.model == "text-embedding-ada-002"
        assert config.dimension == 768

        # Cleanup
        del os.environ["AGENT_RAG_EMBEDDING_MODEL"]
        del os.environ["AGENT_RAG_EMBEDDING_DIMENSION"]


class TestVespaSchemaConfig:
    """Tests for Vespa schema configuration from environment."""

    def test_vespa_schema_config_defaults(self):
        for key in list(os.environ.keys()):
            if key.startswith("AGENT_RAG_VESPA_"):
                del os.environ[key]

        config = get_vespa_schema_config_from_env()
        assert config.schema_name == "agent_rag_chunk"
        assert config.dim == 1536
        assert config.enable_title_embedding is True
        assert config.enable_knowledge_graph is True

    def test_vespa_schema_config_from_env(self):
        os.environ["AGENT_RAG_VESPA_SCHEMA_NAME"] = "custom_schema"
        os.environ["AGENT_RAG_VESPA_SCHEMA_DIM"] = "768"
        os.environ["AGENT_RAG_VESPA_ENABLE_KNOWLEDGE_GRAPH"] = "false"
        os.environ["AGENT_RAG_VESPA_MULTI_TENANT"] = "true"

        config = get_vespa_schema_config_from_env()
        assert config.schema_name == "custom_schema"
        assert config.dim == 768
        assert config.enable_knowledge_graph is False
        assert config.multi_tenant is True

        # Cleanup
        for key in ["AGENT_RAG_VESPA_SCHEMA_NAME", "AGENT_RAG_VESPA_SCHEMA_DIM",
                    "AGENT_RAG_VESPA_ENABLE_KNOWLEDGE_GRAPH", "AGENT_RAG_VESPA_MULTI_TENANT"]:
            del os.environ[key]


class TestSearchConfig:
    """Tests for search configuration from environment."""

    def test_search_config_defaults(self):
        for key in list(os.environ.keys()):
            if key.startswith("AGENT_RAG_SEARCH_"):
                del os.environ[key]

        config = get_search_config_from_env()
        assert config.default_hybrid_alpha == 0.5
        assert config.num_results == 10
        assert config.enable_query_expansion is True

    def test_search_config_from_env(self):
        os.environ["AGENT_RAG_SEARCH_DEFAULT_HYBRID_ALPHA"] = "0.8"
        os.environ["AGENT_RAG_SEARCH_NUM_RESULTS"] = "20"
        os.environ["AGENT_RAG_SEARCH_ENABLE_QUERY_EXPANSION"] = "false"

        config = get_search_config_from_env()
        assert config.default_hybrid_alpha == 0.8
        assert config.num_results == 20
        assert config.enable_query_expansion is False

        # Cleanup
        for key in ["AGENT_RAG_SEARCH_DEFAULT_HYBRID_ALPHA",
                    "AGENT_RAG_SEARCH_NUM_RESULTS",
                    "AGENT_RAG_SEARCH_ENABLE_QUERY_EXPANSION"]:
            del os.environ[key]


class TestAgentConfig:
    """Tests for agent configuration from environment."""

    def test_agent_config_defaults(self):
        for key in list(os.environ.keys()):
            if key.startswith("AGENT_RAG_AGENT_") or key.startswith("AGENT_RAG_DR_"):
                del os.environ[key]

        config = get_agent_config_from_env()
        from agent_rag.core.config import AgentMode
        assert config.mode == AgentMode.CHAT
        assert config.max_cycles == 6
        assert config.enable_citations is True

    def test_agent_config_deep_research_mode(self):
        os.environ["AGENT_RAG_AGENT_MODE"] = "deep_research"
        os.environ["AGENT_RAG_AGENT_MAX_CYCLES"] = "10"

        config = get_agent_config_from_env()
        from agent_rag.core.config import AgentMode
        assert config.mode == AgentMode.DEEP_RESEARCH
        assert config.max_cycles == 10

        # Cleanup
        del os.environ["AGENT_RAG_AGENT_MODE"]
        del os.environ["AGENT_RAG_AGENT_MAX_CYCLES"]


class TestFullConfig:
    """Tests for complete configuration from environment."""

    def test_get_config_from_env(self):
        # Set some env vars
        os.environ["AGENT_RAG_LLM_MODEL"] = "test-model"
        os.environ["AGENT_RAG_INDEX_TYPE"] = "vespa"

        config = get_config_from_env(load_env_file=False)

        assert config.llm.model == "test-model"
        assert config.document_index.type == "vespa"
        assert config.embedding is not None
        assert config.agent is not None

        # Cleanup
        del os.environ["AGENT_RAG_LLM_MODEL"]
        del os.environ["AGENT_RAG_INDEX_TYPE"]
