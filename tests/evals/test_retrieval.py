import json
import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.models import LiteLLMModel
from deepeval.test_case import LLMTestCase

load_dotenv()

BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000")
DATA_FILE = (
        Path(__file__).resolve().parents[2] /"tests"/ "evals" / "data" / "retrieval_goldens.json"
)


def load_goldens() -> list[dict]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def build_judge_model() -> LiteLLMModel:
    model = os.getenv("DEEPEVAL_JUDGE_MODEL") or os.getenv("AGENT_RAG_LLM_MODEL")
    api_key = os.getenv("DEEPEVAL_JUDGE_API_KEY") or os.getenv("AGENT_RAG_LLM_API_KEY")
    api_base = os.getenv("DEEPEVAL_JUDGE_API_BASE") or os.getenv("AGENT_RAG_LLM_API_BASE")

    if not model:
        raise RuntimeError(
            "Missing judge model. Set DEEPEVAL_JUDGE_MODEL or AGENT_RAG_LLM_MODEL."
        )

    if not api_key:
        raise RuntimeError(
            "Missing judge API key. Set DEEPEVAL_JUDGE_API_KEY or AGENT_RAG_LLM_API_KEY."
        )

    return LiteLLMModel(
        model=model,
        api_key=api_key,
        base_url=api_base,
        temperature=0,
    )


def call_search(query: str, limit: int = 5) -> list[dict]:
    response = httpx.post(
        f"{BASE_URL}/api/search",
        json={
            "query": query,
            "limit": limit,
            "hybrid_alpha": 0.5,
            "generate_answer": False,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def build_retrieval_context(results: list[dict]) -> list[str]:
    context = []
    for item in results:
        title = item.get("title") or "Untitled"
        content = item.get("content") or ""
        context.append(f"{title}\n{content}")
    return context


@pytest.mark.parametrize("golden", load_goldens(), ids=lambda g: g["name"])
def test_retrieval_quality(golden: dict) -> None:
    results = call_search(golden["input"], limit=golden.get("limit", 5))
    retrieval_context = build_retrieval_context(results)
    judge_model = build_judge_model()

    test_case = LLMTestCase(
        input=golden["input"],
        expected_output=golden["expected_output"],
        retrieval_context=retrieval_context,
    )

    metrics = [
        ContextualPrecisionMetric(threshold=0.7, model=judge_model),
        ContextualRecallMetric(threshold=0.7, model=judge_model),
    ]
    assert_test(test_case, metrics)
