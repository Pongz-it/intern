import json
import os
from pathlib import Path
from pydoc import text

import httpx
import pytest
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import LiteLLMModel
from deepeval.test_case import LLMTestCase

load_dotenv()

BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000")
DATA_FILE = (
        Path(__file__).resolve().parents[2] / "tests" / "evals" / "data" / "answer_goldens.json"
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


def clean_answer(text: str) -> str:
    text = str(text or "")
    lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[Searching"):
            continue
        if stripped.startswith("正在搜索"):
            continue
        if stripped.startswith("思考中"):
            continue
        lines.append(stripped)

    return "\n".join(lines).strip()


def call_search_stream(query: str) -> str:
    answer_chunks: list[str] = []
    final_answer: str | None = None

    try:
        with httpx.stream(
                "POST",
                f"{BASE_URL}/api/search/stream",
                json={"query": query},
                timeout=120.0,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                payload = json.loads(line[6:])
                event_type = payload.get("type")

                if event_type == "answer_chunk":
                    chunk = payload.get("content", "")
                    if chunk:
                        answer_chunks.append(chunk)

                elif event_type == "answer_end":
                    final_answer = payload.get("full_content")

                elif event_type == "error":
                    raise RuntimeError(payload.get("message", "unknown api error"))

    except (httpx.RemoteProtocolError, httpx.ReadError):
        if answer_chunks:
            return clean_answer("".join(answer_chunks))
        raise

    if final_answer:
        return clean_answer(final_answer)

    return clean_answer("".join(answer_chunks))


@pytest.mark.parametrize("golden", load_goldens(), ids=lambda g: g["name"])
def test_answer_relevancy(golden: dict) -> None:
    actual_output = call_search_stream(golden["input"])
    judge_model = build_judge_model()

    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=judge_model,
    )
    assert_test(test_case, [metric])
