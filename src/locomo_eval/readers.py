"""Answer-model readers.

OpenAI is the Phase 1 implementation; Claude can replace it for answer/eval later.
MockReader supports offline dry-runs and unit tests.

OpenAI free-tier accounts often cap at ~50 requests/day (RPD). Each QA item is
one request unless a store is passed. LlmResponseHash is a future optimization
(utils/); run.py does not pass one. Pace with min_request_interval + 429 retries.
"""

from __future__ import annotations

import os
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from .utils.llm_request_hash import llm_request_hash, llm_request_payload
from .utils.llm_response_hash import PIPELINE_STAGE_ANSWER_READER, LlmResponseHash
from .env import load_env
from .models import chat_create_kwargs, resolve_model
from .prompts import render_qa_prompt


class Reader(ABC):
    """Sandwich bottom: Memory.text + question → predicted answer.

    Called once per unanswered question in run_locomo_pipeline_with_memory_config.
    Must not see gold.
    """

    model_name: str

    @abstractmethod
    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        """Return (predicted_answer, call_meta)."""
        ...


class MockReader(Reader):
    """Offline stand-in for smoke tests. No API; always returns 'Unknown.'"""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        return "Unknown.", {
            "cached": False,
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
        }


def _is_rate_limit_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if "RateLimit" in name:
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort parse of Retry-After or message 'try again in 28m48s'."""
    headers = getattr(exc, "response", None)
    if headers is not None:
        h = getattr(headers, "headers", None) or {}
        ra = h.get("retry-after") or h.get("Retry-After")
        if ra is not None:
            try:
                return float(ra)
            except ValueError:
                pass

    msg = str(exc)
    # e.g. "Please try again in 28m48s" or "in 1.234s"
    m = re.search(r"try again in\s+(\d+)m(\d+(?:\.\d+)?)s", msg, re.I)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", msg, re.I)
    if m:
        return float(m.group(1))
    return None


class OpenAIChatCaller:
    """Shared Chat Completions helper for reader and teacher.

    Optional ``llm_response_hash`` is implemented but not passed from run.py.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
        api_key_env: str = "OPENAI_API_KEY",
        llm_response_hash: LlmResponseHash | None = None,  # future optimization; run.py leaves None
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
    ):
        self.spec = resolve_model(model)
        self.model_name = self.spec.model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_response_hash = llm_response_hash
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.min_request_interval_s = min_request_interval_s
        self.max_wait_s = max_wait_s
        self._last_request_t = 0.0

        load_env()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {api_key_env} in a repo-root .env file "
                f"(see .env.example) or export it in the shell before using OpenAI."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        self._client = OpenAI(api_key=api_key, timeout=timeout_s)

    def _pace(self) -> None:
        if self.min_request_interval_s <= 0:
            return
        elapsed = time.time() - self._last_request_t
        if self._last_request_t and elapsed < self.min_request_interval_s:
            time.sleep(self.min_request_interval_s - elapsed)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        request_extra: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Return (text, call_meta). Hash payload matches ``llm_request_hash``."""
        # Same dict the offline SHA-256 helper hashes. No disk write unless a
        # store was passed (run.py does not pass one).
        request_payload = llm_request_payload(
            spec=self.spec,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra=request_extra,
        )
        key = llm_request_hash(request_payload) if self.llm_response_hash else None

        if self.llm_response_hash and key:
            hit = self.llm_response_hash.get(key)
            if hit is not None:
                return hit["answer"], {
                    "cached": True,
                    "latency_s": 0.0,
                    "usage": hit.get("usage", {}),
                    "llm_request_hash": key,
                    "pipeline_stage": PIPELINE_STAGE_ANSWER_READER,
                    "model": self.model_name,
                    "family": self.spec.family,
                }

        create_kwargs = chat_create_kwargs(
            self.spec,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(**create_kwargs)
                self._last_request_t = time.time()
                latency = time.time() - t0
                text = (resp.choices[0].message.content or "").strip()
                usage = {}
                if resp.usage is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }

                if self.llm_response_hash and key:
                    self.llm_response_hash.set(key, {"answer": text, "usage": usage})

                return text, {
                    "cached": False,
                    "latency_s": round(latency, 3),
                    "usage": usage,
                    "llm_request_hash": key,
                    "attempts": attempt + 1,
                    "pipeline_stage": PIPELINE_STAGE_ANSWER_READER,
                    "model": self.model_name,
                    "family": self.spec.family,
                }
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_error(exc) or attempt >= self.max_retries:
                    raise

                wait = _retry_after_seconds(exc)
                if wait is None:
                    wait = min(2 ** attempt + random.uniform(0, 1), 300.0)
                wait = min(wait, self.max_wait_s)
                print(
                    f"  [rate-limit] attempt {attempt + 1}/{self.max_retries + 1}; "
                    f"sleeping {wait:.1f}s before retry..."
                )
                time.sleep(wait)

        assert last_exc is not None
        raise last_exc


class OpenAIReader(Reader):
    """Live answer LLM (Chat Completions). Optional store hook; run.py does not pass one."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
        api_key_env: str = "OPENAI_API_KEY",
        llm_response_hash: LlmResponseHash | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
    ):
        self._chat = OpenAIChatCaller(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=api_key_env,
            llm_response_hash=llm_response_hash,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
        )
        self.model_name = self._chat.model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_response_hash = llm_response_hash

    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        prompt = render_qa_prompt(prompt_template, memory=memory, question=question)
        messages = [
            {
                "role": "system",
                "content": "You answer questions using only the provided memory.",
            },
            {"role": "user", "content": prompt},
        ]
        return self._chat.complete(
            messages,
            request_extra={
                # These fields complete llm_request_payload; offline compare
                # rebuilds the same dict via llm_request_hash_from_prediction.
                "role": "reader",
                "prompt": prompt,
                "pipeline_stage": PIPELINE_STAGE_ANSWER_READER,
            },
        )


def get_reader(
    name: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 64,
    llm_response_hash: LlmResponseHash | None = None,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
) -> Reader:
    """Build a mock or OpenAI reader. ``llm_response_hash`` defaults to None (unwired)."""
    name = name.lower()
    if name == "mock":
        # Resolve aliases (luna → gpt-5.6-luna) so mock logs match live ids.
        model_name = resolve_model(model).model_id if model else "mock"
        return MockReader(model_name=model_name)
    if name == "openai":
        return OpenAIReader(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_response_hash=llm_response_hash,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
        )
    raise ValueError(f"Unknown reader '{name}'. Use openai or mock.")
