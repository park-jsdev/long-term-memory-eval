"""Answer-model readers.

OpenAI is the Phase 1 implementation; Claude can replace it for answer/eval later.
MockReader supports offline dry-runs and unit tests.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from .cache import ResponseCache
from .env import load_env
from .prompts import render_qa_prompt


class Reader(ABC):
    model_name: str

    @abstractmethod
    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        """Return (predicted_answer, call_meta)."""
        ...


class MockReader(Reader):
    """Deterministic offline reader for pipeline smoke tests."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        # Echo a short placeholder derived only from the question — never the gold answer.
        return "Unknown.", {"cached": False, "latency_s": 0.0, "usage": {}}


class OpenAIReader(Reader):
    """Fixed answer LLM via OpenAI Chat Completions."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
        api_key_env: str = "OPENAI_API_KEY",
        cache: ResponseCache | None = None,
        timeout_s: float = 60.0,
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.timeout_s = timeout_s

        # Prefer process env; fill missing keys from repo-root .env
        load_env()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {api_key_env} in a repo-root .env file "
                f"(see .env.example) or export it in the shell before using OpenAIReader."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        self._client = OpenAI(api_key=api_key, timeout=timeout_s)

    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        prompt = render_qa_prompt(prompt_template, memory=memory, question=question)
        cache_payload = {
            "provider": "openai",
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt": prompt,
        }
        key = ResponseCache.make_key(cache_payload) if self.cache else None

        if self.cache and key:
            hit = self.cache.get(key)
            if hit is not None:
                return hit["answer"], {
                    "cached": True,
                    "latency_s": 0.0,
                    "usage": hit.get("usage", {}),
                    "cache_key": key,
                }

        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions using only the provided memory.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        latency = time.time() - t0
        text = (resp.choices[0].message.content or "").strip()
        usage = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        if self.cache and key:
            self.cache.set(key, {"answer": text, "usage": usage})

        return text, {
            "cached": False,
            "latency_s": round(latency, 3),
            "usage": usage,
            "cache_key": key,
        }


def get_reader(
    name: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 64,
    cache: ResponseCache | None = None,
) -> Reader:
    name = name.lower()
    if name == "mock":
        return MockReader(model_name=model or "mock")
    if name == "openai":
        return OpenAIReader(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
        )
    raise ValueError(f"Unknown reader '{name}'. Use openai or mock.")
