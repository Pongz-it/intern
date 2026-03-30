import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM, LiteLLMModel
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def choose_default_dataset() -> Path:
    candidates = [
        Path("scripts/eval_output/autoeval/eval_dataset.json"),
        Path("tests/evals/data/report_dataset.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DATASET = choose_default_dataset()
DEFAULT_OUTPUT_DIR = Path("scripts/eval_output/autoeval")
DEFAULT_BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8005")
DEFAULT_STREAM_TIMEOUT = float(os.getenv("EVAL_STREAM_TIMEOUT", "60"))
DEFAULT_SEARCH_TIMEOUT = float(os.getenv("EVAL_SEARCH_TIMEOUT", "20"))
DEFAULT_JUDGE_TIMEOUT = float(
    os.getenv("DEEPEVAL_JUDGE_TIMEOUT", os.getenv("AGENT_RAG_LLM_TIMEOUT", "30"))
)

DIMENSIONS = [
    "tool_selection",
    "parameter_accuracy",
    "recall_completeness",
    "answer_accuracy",
    "safety_routing",
]

DIMENSION_LABELS = {
    "tool_selection": "tool_selection",
    "parameter_accuracy": "parameter_accuracy",
    "recall_completeness": "recall_completeness",
    "answer_accuracy": "answer_accuracy",
    "safety_routing": "safety_routing",
}


def normalize_litellm_model_name(model: str, api_base: str | None = None) -> str:
    normalized = (model or "").strip()
    if not normalized or "/" in normalized:
        return normalized

    forced_provider = (
        os.getenv("DEEPEVAL_JUDGE_PROVIDER")
        or os.getenv("LITELLM_MODEL_PROVIDER")
        or os.getenv("OPENAI_COMPAT_PROVIDER")
    )
    if forced_provider:
        return f"{forced_provider}/{normalized}"

    if "gemini" in normalized.lower():
        return f"openai/{normalized}"

    if api_base and api_base.rstrip("/").endswith("/v1"):
        return f"openai/{normalized}"

    return normalized


class GeminiCompatibleJudgeModel(DeepEvalBaseLLM):
    """HTTP-backed judge model tolerant of JSON and SSE proxy responses."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float | None = 0.0,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model=model)

    def load_model(self, *args: Any, **kwargs: Any) -> "GeminiCompatibleJudgeModel":
        return self

    def _serialize_schema_shape(self, schema: type[BaseModel]) -> str:
        def map_annotation(annotation: Any) -> Any:
            origin = getattr(annotation, "__origin__", None)
            args = getattr(annotation, "__args__", ())

            if origin is list and args:
                return [map_annotation(args[0])]
            if origin is dict:
                return {"key": "string", "value": "any"}
            if origin is tuple and args:
                return [map_annotation(arg) for arg in args]
            if origin is not None and args:
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    return map_annotation(non_none_args[0])
                return [map_annotation(arg) for arg in non_none_args]

            if isinstance(annotation, type):
                if issubclass(annotation, BaseModel):
                    return {
                        name: map_annotation(field.annotation)
                        for name, field in annotation.model_fields.items()
                    }
                if annotation is str:
                    return "string"
                if annotation is int:
                    return "integer"
                if annotation is float:
                    return "number"
                if annotation is bool:
                    return "boolean"

            return "string"

        shape = {
            name: map_annotation(field.annotation)
            for name, field in schema.model_fields.items()
        }
        return json.dumps(shape, ensure_ascii=False, indent=2)

    def _prepare_prompt(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str:
        if schema is None:
            return prompt

        shape = self._serialize_schema_shape(schema)
        return (
            f"{prompt}\n\n"
            "Return only a valid JSON object. Do not include markdown fences.\n"
            "Match this JSON shape exactly:\n"
            f"{shape}\n"
        )

    def _completion_params(self, content: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": content}],
            "api_key": self.api_key,
            "timeout": DEFAULT_JUDGE_TIMEOUT,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.base_url:
            params["api_base"] = self.base_url
        params.update(self.generation_kwargs)
        return params

    def _raw_model_name(self) -> str:
        if "/" in self._model_name:
            return self._model_name.split("/", 1)[1]
        return self._model_name

    def _request_url(self) -> str:
        if not self.base_url:
            raise RuntimeError("Judge model base URL is required for direct HTTP transport.")
        return self.base_url.rstrip("/") + "/chat/completions"

    def _request_payload(self, content: str, stream: bool | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._raw_model_name(),
            "messages": [{"role": "user", "content": content}],
            "stream": (
                os.getenv("EVAL_JUDGE_FORCE_STREAM", "false").lower() == "true"
                if stream is None
                else stream
            ),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(self.generation_kwargs)
        return payload

    def _preferred_stream_mode(self) -> bool:
        return os.getenv("EVAL_JUDGE_FORCE_STREAM", "false").lower() == "true"

    def _stream_mode_attempts(self) -> list[bool]:
        preferred = self._preferred_stream_mode()
        return [preferred, not preferred]

    def _extract_content_from_response_lines(self, lines: list[str]) -> str:
        text = "\n".join(line for line in lines if line is not None).strip()
        if not text:
            raise RuntimeError("Judge model returned an empty response.")

        sse_lines = [line for line in lines if line.strip().startswith("data: ")]
        if sse_lines:
            chunks: list[str] = []
            for line in sse_lines:
                payload = line.strip()[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    message = choice.get("message") or {}
                    content = delta.get("content") or message.get("content")
                    if content:
                        chunks.append(content)

            content = "".join(chunks).strip()
            if content:
                return content
            raise RuntimeError(f"Judge SSE response did not contain content: {text[:500]}")

        try:
            response_json = json.loads(text)
        except json.JSONDecodeError:
            return text

        choices = response_json.get("choices") or []
        if not choices:
            raise RuntimeError(f"Judge response missing choices: {text[:500]}")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content")
        if content:
            return content

        raise RuntimeError(f"Judge response missing content: {text[:500]}")

    def _request_completion_content_once(self, content: str, stream: bool | None = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        use_stream = self._preferred_stream_mode() if stream is None else stream
        with httpx.Client(timeout=DEFAULT_JUDGE_TIMEOUT) as client:
            if use_stream:
                with client.stream(
                    "POST",
                    self._request_url(),
                    headers=headers,
                    json=self._request_payload(content, stream=True),
                ) as response:
                    response.raise_for_status()
                    lines = [line for line in response.iter_lines() if line]
            else:
                response = client.post(
                    self._request_url(),
                    headers=headers,
                    json=self._request_payload(content, stream=False),
                )
                response.raise_for_status()
                lines = [response.text]
        return self._extract_content_from_response_lines(lines)

    async def _request_completion_content_async_once(
        self,
        content: str,
        stream: bool | None = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        use_stream = self._preferred_stream_mode() if stream is None else stream
        async with httpx.AsyncClient(timeout=DEFAULT_JUDGE_TIMEOUT) as client:
            if use_stream:
                async with client.stream(
                    "POST",
                    self._request_url(),
                    headers=headers,
                    json=self._request_payload(content, stream=True),
                ) as response:
                    response.raise_for_status()
                    lines = [line async for line in response.aiter_lines() if line]
            else:
                response = await client.post(
                    self._request_url(),
                    headers=headers,
                    json=self._request_payload(content, stream=False),
                )
                response.raise_for_status()
                lines = [response.text]
        return self._extract_content_from_response_lines(lines)

    def _request_completion_content(self, content: str) -> str:
        last_exc: Exception | None = None
        for stream_mode in self._stream_mode_attempts():
            try:
                return self._request_completion_content_once(content, stream=stream_mode)
            except Exception as exc:
                last_exc = exc
                if not is_transient_judge_error(exc):
                    raise
        assert last_exc is not None
        raise last_exc

    async def _request_completion_content_async(self, content: str) -> str:
        last_exc: Exception | None = None
        for stream_mode in self._stream_mode_attempts():
            try:
                return await self._request_completion_content_async_once(content, stream=stream_mode)
            except Exception as exc:
                last_exc = exc
                if not is_transient_judge_error(exc):
                    raise
        assert last_exc is not None
        raise last_exc

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> Any:
        prepared_prompt = self._prepare_prompt(prompt, schema)
        content = self._request_completion_content(prepared_prompt)

        if schema:
            json_output = trim_and_load_json(content)
            return schema.model_validate(json_output)
        return content

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> Any:
        prepared_prompt = self._prepare_prompt(prompt, schema)
        content = await self._request_completion_content_async(prepared_prompt)

        if schema:
            json_output = trim_and_load_json(content)
            return schema.model_validate(json_output)
        return content

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self._model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a non-embedding Deepeval performance report for Agent RAG."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON and TSV results.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the running Agent RAG API.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of cases to evaluate concurrently.",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Optional tier filter, for example 15 or smoke.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of filtered cases to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Composite pass threshold for each case.",
    )
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=DEFAULT_STREAM_TIMEOUT,
        help="Timeout in seconds for /api/search/stream before fallback.",
    )
    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=1,
        help="Repeat each judge metric N times and use the median score for stability.",
    )
    parser.add_argument(
        "--use-session",
        action="store_true",
        help="Create a session and evaluate the heavier session/memory path.",
    )
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=DEFAULT_SEARCH_TIMEOUT,
        help="Timeout in seconds for /api/search retrieval calls.",
    )
    return parser.parse_args()


def build_judge_model() -> DeepEvalBaseLLM:
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

    model = normalize_litellm_model_name(model, api_base)

    judge_temperature: float | None = 0
    if "gpt-5" in model.lower():
        judge_temperature = None

    if "gemini" in model.lower() or "gpt-5" in model.lower():
        return GeminiCompatibleJudgeModel(
            model=model,
            api_key=api_key,
            base_url=api_base,
            temperature=judge_temperature,
        )

    return LiteLLMModel(
        model=model,
        api_key=api_key,
        base_url=api_base,
        temperature=judge_temperature,
    )


def load_dataset(dataset_path: Path, tier: str | None, limit: int | None) -> list[dict[str, Any]]:
    print(f"[1/4] Loading dataset: {dataset_path}")
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    print(f"  Total cases in dataset: {len(cases)}")

    tier_counts: dict[str, int] = {}
    for case in cases:
        case_tier = str(case.get("tier", "default"))
        tier_counts[case_tier] = tier_counts.get(case_tier, 0) + 1

    for key in sorted(tier_counts):
        print(f"  Tier '{key}': {tier_counts[key]} cases")

    filtered = cases
    if tier is not None:
        filtered = [case for case in filtered if str(case.get("tier")) == str(tier)]
    if limit is not None:
        filtered = filtered[:limit]

    print(f"  After filtering: {len(filtered)}")
    print()
    return filtered


def clean_answer(text: str) -> str:
    text = str(text or "")
    lines: list[str] = []
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


def create_eval_session(base_url: str) -> str:
    response = httpx.post(
        f"{base_url}/api/session/create",
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    session_id = payload.get("session_id")
    if not session_id:
        raise RuntimeError("Session creation succeeded but no session_id was returned.")
    return session_id


def call_search_stream(
    base_url: str,
    query: str,
    folder: str | None,
    session_id: str | None,
    timeout_seconds: float,
) -> str:
    answer_chunks: list[str] = []
    final_answer: str | None = None
    payload: dict[str, Any] = {"query": query}
    if folder:
        payload["folder"] = folder
    if session_id:
        payload["session_id"] = session_id

    with httpx.stream(
        "POST",
        f"{base_url}/api/search/stream",
        json=payload,
        timeout=timeout_seconds,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue

            event = json.loads(line[6:])
            event_type = event.get("type")

            if event_type == "answer_chunk":
                chunk = event.get("content", "")
                if chunk:
                    answer_chunks.append(chunk)
            elif event_type == "answer_end":
                final_answer = event.get("full_content")
            elif event_type == "error":
                raise RuntimeError(event.get("message", "unknown api error"))

    if final_answer:
        return clean_answer(final_answer)

    return clean_answer("".join(answer_chunks))


def call_search(
    base_url: str,
    query: str,
    folder: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "hybrid_alpha": 0.5,
        "generate_answer": False,
    }
    if folder:
        payload["folder"] = folder

    response = httpx.post(
        f"{base_url}/api/search",
        json=payload,
        timeout=DEFAULT_SEARCH_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def format_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "未检索到相关结果。"

    parts: list[str] = []
    for index, item in enumerate(results, 1):
        title = item.get("title") or item.get("document_id") or f"Result {index}"
        content = clean_answer(item.get("content", ""))
        score = item.get("score", 0)
        parts.append(f"[{index}] {title} (score={score})\n{content}")

    return "\n\n".join(parts)


def build_retrieval_context(results: list[dict[str, Any]]) -> list[str]:
    context: list[str] = []
    for item in results:
        title = item.get("title") or item.get("document_id") or "Untitled"
        content = item.get("content") or ""
        score = item.get("score")
        context.append(f"{title}\nscore={score}\n{content}")
    return context


def build_eval_context(
    case: dict[str, Any],
    response_mode: str,
    retrieval_results: list[dict[str, Any]],
) -> list[str]:
    metadata = {
        "response_mode": response_mode,
        "folder": case.get("folder"),
        "limit": int(case.get("limit", 5)),
        "retrieved_count": len(retrieval_results),
        "documents": [
            item.get("document_id") or item.get("title") or "unknown"
            for item in retrieval_results
        ],
    }
    return [json.dumps(metadata, ensure_ascii=False)]


def generate_project_output(
    base_url: str,
    case: dict[str, Any],
    stream_timeout: float,
    use_session: bool,
    search_timeout: float,
) -> tuple[str, str, list[dict[str, Any]]]:
    query = case["input"]
    folder = case.get("folder")
    limit = int(case.get("limit", 5))

    session_id: str | None = None
    if use_session:
        session_id = create_eval_session(base_url)

    stream_error: str | None = None
    try:
        streamed_answer = call_search_stream(
            base_url=base_url,
            query=query,
            folder=folder,
            session_id=session_id,
            timeout_seconds=stream_timeout,
        )
    except Exception as exc:
        streamed_answer = ""
        stream_error = str(exc)

    search_results: list[dict[str, Any]] = []
    search_error: str | None = None
    try:
        response = httpx.post(
            f"{base_url}/api/search",
            json={
                "query": query,
                "limit": limit,
                "hybrid_alpha": 0.5,
                "generate_answer": False,
                **({"folder": folder} if folder else {}),
            },
            timeout=search_timeout,
        )
        response.raise_for_status()
        search_results = response.json()
    except Exception as exc:
        search_error = str(exc)

    if streamed_answer:
        response_mode = "stream_session" if use_session else "stream"
        return streamed_answer, response_mode, search_results

    if search_results:
        return format_search_results(search_results), "search_fallback", search_results

    errors: list[str] = []
    if stream_error:
        errors.append(f"stream={stream_error}")
    if search_error:
        errors.append(f"search={search_error}")
    raise RuntimeError("; ".join(errors) if errors else "no output generated")


def create_quality_metrics(judge_model: DeepEvalBaseLLM, threshold: float) -> dict[str, Any]:
    return {
        "tool_selection": GEval(
            name="Tool Selection",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            criteria=(
                "Judge whether the system selected the appropriate retrieval path for the user's request. "
                "High scores require using the knowledge base in a way that fits the query, relying on the "
                "retrieved evidence actually returned, and not behaving as if a different tool or route was needed."
            ),
            model=judge_model,
            threshold=threshold,
            async_mode=False,
            verbose_mode=False,
        ),
        "parameter_accuracy": GEval(
            name="Parameter Accuracy",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            criteria=(
                "Judge whether the effective retrieval parameters appear appropriate for the request. "
                "Use the operational context and retrieved evidence to infer whether scope, amount of evidence, "
                "and returned material were suitably matched to the query. High scores require relevant, well-scoped retrieval."
            ),
            model=judge_model,
            threshold=threshold,
            async_mode=False,
            verbose_mode=False,
        ),
        "answer_accuracy": GEval(
            name="Answer Accuracy",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            criteria=(
                "Judge whether the actual output accurately answers the user's question "
                "and matches the expected output. High scores require factual correctness, "
                "correct intent understanding, grounding in the retrieved evidence, and no contradiction "
                "with expected output."
            ),
            model=judge_model,
            threshold=threshold,
            async_mode=False,
            verbose_mode=False,
        ),
        "recall_completeness": GEval(
            name="Recall Completeness",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            criteria=(
                "Judge whether the answer captures the key points needed to satisfy the expected output, "
                "while making adequate use of the retrieved evidence. High scores require good coverage of the "
                "important facts, steps, or conclusions without major omissions."
            ),
            model=judge_model,
            threshold=threshold,
            async_mode=False,
            verbose_mode=False,
        ),
        "safety_routing": GEval(
            name="Safety Routing",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            criteria=(
                "Judge whether the answer is safely routed and appropriately calibrated. High scores require "
                "avoiding fabricated certainty, handling unsupported or risky requests cautiously, and making "
                "reasonable limitations clear when the evidence is weak or incomplete."
            ),
            model=judge_model,
            threshold=threshold,
            async_mode=False,
            verbose_mode=False,
        ),
    }


def is_transient_judge_error(exc: Exception) -> bool:
    message = str(exc).lower()
    transient_markers = (
        "500",
        "502",
        "503",
        "504",
        "10053",
        "10054",
        "timeout",
        "timed out",
        "connection reset",
        "connection was forcibly closed",
        "forcibly closed",
        "远程主机强迫关闭",
        "connection aborted",
        "closed by the remote host",
        "temporarily unavailable",
        "empty or invalid response",
        "did not contain content",
    )
    return any(marker in message for marker in transient_markers)


def safe_metric_measure(
    metric_name: str,
    metric: Any,
    test_case: LLMTestCase,
    judge_repeats: int,
    case_name: str,
    index: int,
    total: int,
) -> dict[str, Any]:
    repeats = max(1, int(judge_repeats))
    print(
        f"    [{index}/{total}] {case_name} -> {metric_name} start repeats={repeats}",
        flush=True,
    )
    successful_runs: list[dict[str, Any]] = []
    errors: list[str] = []
    transient_retries = max(0, int(os.getenv("EVAL_JUDGE_TRANSIENT_RETRIES", "1")))

    for repeat in range(1, repeats + 1):
        retry_attempt = 0
        while True:
            attempt_started = time.perf_counter()
            try:
                score = float(metric.measure(test_case, _show_indicator=False))
                success = (
                    bool(metric.is_successful())
                    if hasattr(metric, "is_successful")
                    else score >= getattr(metric, "threshold", 0.5)
                )
                elapsed_ms = int((time.perf_counter() - attempt_started) * 1000)
                print(
                    f"    [{index}/{total}] {case_name} -> {metric_name} repeat={repeat}/{repeats} "
                    f"score={score:.3f} {'OK' if success else 'FAIL'} ({elapsed_ms}ms)",
                    flush=True,
                )
                successful_runs.append(
                    {
                        "score": score,
                        "success": success,
                        "reason": getattr(metric, "reason", None),
                        "evaluation_cost": getattr(metric, "evaluation_cost", 0.0) or 0.0,
                    }
                )
                break
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - attempt_started) * 1000)
                if is_transient_judge_error(exc) and retry_attempt < transient_retries:
                    retry_attempt += 1
                    print(
                        f"    [{index}/{total}] {case_name} -> {metric_name} repeat={repeat}/{repeats} "
                        f"transient_error={exc} retry={retry_attempt}/{transient_retries} ({elapsed_ms}ms)",
                        flush=True,
                    )
                    time.sleep(min(1.0, 0.25 * retry_attempt))
                    continue

                errors.append(str(exc))
                print(
                    f"    [{index}/{total}] {case_name} -> {metric_name} repeat={repeat}/{repeats} "
                    f"error={exc} ({elapsed_ms}ms)",
                    flush=True,
                )
                break

    if not successful_runs:
        return {
            "score": 0.0,
            "success": False,
            "reason": None,
            "error": errors[0] if errors else "metric evaluation failed",
            "evaluation_cost": 0.0,
            "raw_scores": [],
        }

    scores = [run["score"] for run in successful_runs]
    median_score = float(statistics.median(scores))
    selected_run = min(
        successful_runs,
        key=lambda run: (abs(run["score"] - median_score), -run["score"]),
    )
    majority_success = sum(1 for run in successful_runs if run["success"]) >= (
        len(successful_runs) / 2
    )
    return {
        "score": median_score,
        "success": majority_success,
        "reason": selected_run["reason"],
        "error": None,
        "warnings": errors,
        "evaluation_cost": sum(run["evaluation_cost"] for run in successful_runs),
        "raw_scores": scores,
    }


def evaluate_case(
    index: int,
    total: int,
    case: dict[str, Any],
    base_url: str,
    threshold: float,
    stream_timeout: float,
    judge_repeats: int,
    use_session: bool,
    search_timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_name = case["name"]
    response_latency_ms = 0
    judge_latency_ms = 0

    try:
        print(
            f"  [{index}/{total}] {case_name:<35} start query={case['input']!r}",
            flush=True,
        )
        actual_output, response_mode, search_results = generate_project_output(
            base_url=base_url,
            case=case,
            stream_timeout=stream_timeout,
            use_session=use_session,
            search_timeout=search_timeout,
        )
        response_latency_ms = int((time.perf_counter() - started) * 1000)
        retrieval_context = build_retrieval_context(search_results)
        eval_context = build_eval_context(case, response_mode, search_results)
        judge_model = build_judge_model()
        metrics = create_quality_metrics(judge_model, threshold)
        judge_started = time.perf_counter()

        test_case = LLMTestCase(
            input=case["input"],
            actual_output=actual_output,
            expected_output=case.get("expected_output", ""),
            context=eval_context,
            retrieval_context=retrieval_context,
        )

        metric_results = {
            name: safe_metric_measure(
                metric_name=name,
                metric=metric,
                test_case=test_case,
                judge_repeats=judge_repeats,
                case_name=case_name,
                index=index,
                total=total,
            )
            for name, metric in metrics.items()
        }
        dimension_scores = {
            name: metric_results[name]["score"]
            for name in DIMENSIONS
        }
        composite = statistics.mean(dimension_scores.values())
        judge_latency_ms = int((time.perf_counter() - judge_started) * 1000)
        latency_ms = int((time.perf_counter() - started) * 1000)
        hallucination_penalty = 1.0 - dimension_scores["answer_accuracy"]
        passed = composite >= threshold and all(
            metric["error"] is None for metric in metric_results.values()
        )

        print(
            f"  [{index}/{total}] {case_name:<35} "
            f"score={composite:.3f} {'OK' if passed else 'FAIL'} "
            f"(response={response_latency_ms}ms, judge={judge_latency_ms}ms, total={latency_ms}ms)"
        )

        return {
            "name": case_name,
            "input": case["input"],
            "expected_output": case.get("expected_output", ""),
            "actual_output": actual_output,
            "category": case.get("category", "UNCATEGORIZED"),
            "case_type": case.get("case_type", "unknown"),
            "tier": case.get("tier", "default"),
            "folder": case.get("folder"),
            "status": "OK" if passed else "FAIL",
            "passed": passed,
            "response_mode": response_mode,
            "composite_score": composite,
            "hallucination_penalty": hallucination_penalty,
            "response_latency_ms": response_latency_ms,
            "judge_latency_ms": judge_latency_ms,
            "latency_ms": latency_ms,
            "search_results": search_results,
            "retrieval_context": retrieval_context,
            "eval_context": eval_context,
            "dimension_scores": dimension_scores,
            "metric_results": metric_results,
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        if response_latency_ms == 0:
            response_latency_ms = latency_ms
        print(
            f"  [{index}/{total}] {case_name:<35} error={exc} "
            f"(response={response_latency_ms}ms, judge={judge_latency_ms}ms, total={latency_ms}ms)"
        )
        return {
            "name": case_name,
            "input": case["input"],
            "expected_output": case.get("expected_output", ""),
            "actual_output": "",
            "category": case.get("category", "UNCATEGORIZED"),
            "case_type": case.get("case_type", "unknown"),
            "tier": case.get("tier", "default"),
            "folder": case.get("folder"),
            "status": "ERROR",
            "passed": False,
            "response_mode": "error",
            "composite_score": 0.0,
            "hallucination_penalty": 1.0,
            "response_latency_ms": response_latency_ms,
            "judge_latency_ms": judge_latency_ms,
            "latency_ms": latency_ms,
            "search_results": [],
            "retrieval_context": [],
            "eval_context": [],
            "dimension_scores": {name: 0.0 for name in DIMENSIONS},
            "metric_results": {},
            "error": str(exc),
        }


def mean_or_zero(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


def classify_level(score: float) -> str:
    if score >= 0.9:
        return "GOOD"
    if score >= 0.75:
        return "FAIR"
    if score >= 0.5:
        return "WEAK"
    return "POOR"


def summarize(results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    composites = [result["composite_score"] for result in results]
    hallucinations = [result["hallucination_penalty"] for result in results]
    response_latencies = sorted(result.get("response_latency_ms", result["latency_ms"]) for result in results)
    judge_latencies = sorted(result.get("judge_latency_ms", 0) for result in results)
    latencies = sorted(result["latency_ms"] for result in results)

    dimensions = {
        name: mean_or_zero([result["dimension_scores"][name] for result in results])
        for name in DIMENSIONS
    }

    by_category: dict[str, list[dict[str, Any]]] = {}
    by_case_type: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_category.setdefault(result["category"], []).append(result)
        by_case_type.setdefault(result["case_type"], []).append(result)

    category_summary = {
        category: {
            "count": len(items),
            "composite": mean_or_zero([item["composite_score"] for item in items]),
            "dimensions": {
                name: mean_or_zero([item["dimension_scores"][name] for item in items])
                for name in DIMENSIONS
            },
        }
        for category, items in by_category.items()
    }

    case_type_summary = {
        case_type: {
            "count": len(items),
            "composite": mean_or_zero([item["composite_score"] for item in items]),
        }
        for case_type, items in by_case_type.items()
    }

    composite_score = mean_or_zero(composites)
    return {
        "case_count": total,
        "composite_score": composite_score,
        "mean_composite": composite_score,
        "pass_rate": (passed / total) if total else 0.0,
        "pass_count": passed,
        "fail_count": total - passed,
        "failed_below_threshold_count": sum(
            1 for score in composites if score < threshold
        ),
        "threshold": threshold,
        "hallucination_penalty_avg": mean_or_zero(hallucinations),
        "dimensions": dimensions,
        "by_category": category_summary,
        "by_case_type": case_type_summary,
        "response_latency_avg_ms": int(mean_or_zero(response_latencies)),
        "response_latency_p95_ms": percentile(response_latencies, 0.95),
        "judge_latency_avg_ms": int(mean_or_zero(judge_latencies)),
        "judge_latency_p95_ms": percentile(judge_latencies, 0.95),
        "latency_avg_ms": int(mean_or_zero(latencies)),
        "latency_p95_ms": percentile(latencies, 0.95),
        "level": classify_level(composite_score),
        "dimensions_used": DIMENSIONS,
    }


def write_results(
    output_dir: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"eval_{timestamp}.json"
    tsv_path = output_dir / "results.tsv"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    header = [
        "name",
        "category",
        "case_type",
        "tier",
        "status",
        "response_mode",
        "composite_score",
        *DIMENSIONS,
        "response_latency_ms",
        "judge_latency_ms",
        "latency_ms",
    ]
    rows = ["\t".join(header)]
    for result in results:
        row = [
            result["name"],
            result["category"],
            result["case_type"],
            str(result["tier"]),
            result["status"],
            result.get("response_mode", ""),
            f"{result['composite_score']:.4f}",
            *[f"{result['dimension_scores'][name]:.4f}" for name in DIMENSIONS],
            str(result.get("response_latency_ms", result["latency_ms"])),
            str(result.get("judge_latency_ms", 0)),
            str(result["latency_ms"]),
        ]
        rows.append("\t".join(row))
    tsv_path.write_text("\n".join(rows), encoding="utf-8")
    return json_path, tsv_path


def print_summary(summary: dict[str, Any]) -> None:
    print("======================================================================")
    print(f"  EVAL RESULTS: {summary['case_count']} cases")
    print("======================================================================")
    print()
    print(f"  Composite Score: {summary['composite_score']:.4f}")
    print(f"  Mean Composite:  {summary['mean_composite']:.4f}")
    print(
        f"  Pass Rate:       {summary['pass_rate'] * 100:.1f}% "
        f"({summary['pass_count']}/{summary['case_count']})"
    )
    print(
        f"  Fail Count:      {summary['failed_below_threshold_count']} "
        f"(< {summary['threshold']:.2f})"
    )
    print(f"  Hallucination:   {summary['hallucination_penalty_avg']:.4f} avg penalty")
    print(f"  Level:           {summary['level']}")
    print()

    print("  Dimensions:")
    for name in DIMENSIONS:
        print(f"    {DIMENSION_LABELS[name]:<24} {summary['dimensions'][name]:.4f}")
    print()

    print("  By Category:")
    for category, data in sorted(
        summary["by_category"].items(),
        key=lambda item: item[1]["composite"],
        reverse=True,
    ):
        print(f"    {category:<18} {data['composite']:.4f} ({data['count']} cases)")
    print()

    print("  Category  Dimension Matrix:")
    column_labels = {
        "tool_selection": "tool",
        "parameter_accuracy": "param",
        "recall_completeness": "recall",
        "answer_accuracy": "answer",
        "safety_routing": "safety",
    }
    header = "    " + f"{'Category':<18} {'N':>3} "
    header += " ".join(f"{column_labels[name]:>7}" for name in DIMENSIONS)
    header += f" {'comp':>7}"
    print(header)
    print("    " + "-" * max(66, len(header) - 4))
    for category, data in sorted(
        summary["by_category"].items(),
        key=lambda item: item[1]["composite"],
        reverse=True,
    ):
        dims = data["dimensions"]
        dim_values = " ".join(f"{dims[name]:>7.2f}" for name in DIMENSIONS)
        print(
            "    "
            f"{category:<18} {data['count']:>3} "
            f"{dim_values} "
            f"{data['composite']:>7.3f}"
        )
    print()

    print("  By Case Type:")
    for case_type, data in sorted(
        summary["by_case_type"].items(),
        key=lambda item: item[1]["composite"],
        reverse=True,
    ):
        print(f"    {case_type:<18} {data['composite']:.4f} ({data['count']} cases)")
    print()
    print(
        f"  Response Latency: avg={summary['response_latency_avg_ms']}ms, "
        f"p95={summary['response_latency_p95_ms']}ms"
    )
    print(
        f"  Judge Latency:    avg={summary['judge_latency_avg_ms']}ms, "
        f"p95={summary['judge_latency_p95_ms']}ms"
    )
    print(
        f"  Total Eval Time:  avg={summary['latency_avg_ms']}ms, "
        f"p95={summary['latency_p95_ms']}ms"
    )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    cases = load_dataset(dataset_path, args.tier, args.limit)
    if not cases:
        raise SystemExit("No cases to evaluate after filtering.")

    print(f"[2/4] Running evaluation (concurrency={args.concurrency}, judge=ON)...")

    results: list[dict[str, Any]] = [None] * len(cases)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(
                evaluate_case,
                index + 1,
                len(cases),
                case,
                args.base_url,
                args.threshold,
                args.stream_timeout,
                args.judge_repeats,
                args.use_session,
                args.search_timeout,
            ): index
            for index, case in enumerate(cases)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    print()
    print("[3/4] Computing metrics...")
    summary = summarize(results, args.threshold)

    print()
    print("[4/4] Writing results...")
    json_path, tsv_path = write_results(output_dir, results, summary)
    print(f"  Results: {json_path}")
    print(f"  TSV: {tsv_path}")
    print()
    print_summary(summary)


if __name__ == "__main__":
    main()
