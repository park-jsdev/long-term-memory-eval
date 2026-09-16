"""Answer-model readers.

OpenAI is the Phase 1 implementation; Claude can replace it for answer/eval later.
MockReader supports offline dry-runs and unit tests.

Each QA item is one fresh request. Pace live calls with min_request_interval
and 429 retries.
"""

from __future__ import annotations

import os
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from .env import load_env
from .models import (
    TEACHER_THINKING_MIN_OUTPUT_TOKENS,
    anthropic_messages_kwargs,
    apply_deepseek_thinking,
    apply_openai_thinking,
    chat_create_kwargs,
    resolve_model,
    supports_reasoning_effort,
)
from .prompts import render_qa_prompt

READER_MESSAGE_LAYOUT_DEFAULT = "default_system_user"
READER_MESSAGE_LAYOUT_MEM0 = "mem0_system_only"
# Thinking-on Chat Completions can exceed the default 60s HTTP timeout
# (GPT-5 high effort on full_context). Teachers reuse this caller too.
THINKING_HTTP_TIMEOUT_S = 600.0


def build_reader_messages(prompt: str, message_layout: str) -> list[dict[str, str]]:
    """Build the exact Chat Completions message layout selected by config."""
    if message_layout == READER_MESSAGE_LAYOUT_MEM0:
        return [{"role": "system", "content": prompt}]
    if message_layout == READER_MESSAGE_LAYOUT_DEFAULT:
        return [
            {
                "role": "system",
                "content": "You answer questions using only the provided memory.",
            },
            {"role": "user", "content": prompt},
        ]
    raise ValueError(f"Unknown reader message_layout: {message_layout}")


class Reader(ABC):
    """Sandwich bottom: Memory.text + question → predicted answer.

    Called once per question in run_locomo_pipeline_with_memory_config.
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
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
            "thinking": False,
            "thinking_supported": False,
            "reasoning": "",
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


# OpenAI-compatible Chat Completions providers (answer / teacher / extract).
OPENAI_COMPAT_PROVIDERS: dict[str, dict[str, str | None]] = {
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": None},
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
    },
}


class OpenAIChatCaller:
    """Shared Chat Completions helper for reader, teacher, and autorater.

    ``base_url`` lets DeepSeek (and other OpenAI-compatible APIs) reuse this
    client. Anthropic is a separate caller.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = 64,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
        base_url: str | None = None,
        thinking: bool | None = None,
    ):
        self.spec = resolve_model(model)
        self.model_name = self.spec.model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        if thinking is True:
            timeout_s = max(float(timeout_s), THINKING_HTTP_TIMEOUT_S)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.min_request_interval_s = min_request_interval_s
        self.max_wait_s = max_wait_s
        self.base_url = base_url
        self._last_request_t = 0.0
        # None = catalog pin (frozen reader). Teachers pass True/False.
        self.thinking = thinking

        load_env()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {api_key_env} in a repo-root .env file "
                f"(see .env.example) or export it in the shell before using {api_key_env}."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def _pace(self) -> None:
        if self.min_request_interval_s <= 0:
            return
        elapsed = time.time() - self._last_request_t
        if self._last_request_t and elapsed < self.min_request_interval_s:
            time.sleep(self.min_request_interval_s - elapsed)

    def _is_deepseek(self) -> bool:
        return bool(self.base_url) and "deepseek" in self.base_url

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        create_extra: dict[str, Any] | None = None,
        thinking: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return (text, call_meta).

        ``create_extra`` is merged into Chat Completions kwargs (e.g. JSON
        ``response_format`` for the autorater).
        """
        from .reasoning_extractor import openai_reasoning_text, openai_reasoning_tokens

        create_kwargs = chat_create_kwargs(
            self.spec,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        use_thinking = self.thinking if thinking is None else bool(thinking)
        if self._is_deepseek():
            # None (non-teacher callers) and False (ping) both disable so
            # small max_tokens still yield visible JSON/pong. On must send
            # type=enabled: deepseek-chat aliases to flash with thinking off.
            create_kwargs = apply_deepseek_thinking(
                create_kwargs, bool(use_thinking)
            )
            if use_thinking is True:
                field = self.spec.max_tokens_field
                n = int(create_kwargs.get(field) or 16)
                create_kwargs[field] = max(n, TEACHER_THINKING_MIN_OUTPUT_TOKENS)
        else:
            create_kwargs = apply_openai_thinking(
                self.spec, create_kwargs, use_thinking
            )
        if create_extra:
            create_kwargs.update(create_extra)
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(**create_kwargs)
                self._last_request_t = time.time()
                latency = time.time() - t0
                message = resp.choices[0].message
                text = (message.content or "").strip()
                reasoning = openai_reasoning_text(message)
                usage = {}
                reasoning_tokens = openai_reasoning_tokens(resp.usage)
                if resp.usage is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                    if reasoning_tokens is not None:
                        usage["reasoning_tokens"] = reasoning_tokens
                thinking_supported = self._is_deepseek() or supports_reasoning_effort(
                    self.spec
                )
                return text, {
                    "latency_s": round(latency, 3),
                    "usage": usage,
                    "attempts": attempt + 1,
                    "model": self.model_name,
                    "family": self.spec.family,
                    "provider": "deepseek" if self._is_deepseek() else "openai",
                    "thinking": bool(use_thinking),
                    "thinking_supported": thinking_supported,
                    "reasoning": reasoning,
                    "reasoning_tokens": reasoning_tokens,
                    "reasoning_effort": create_kwargs.get("reasoning_effort"),
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
    """Live answer LLM. One Chat Completions call per question."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = 64,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
        message_layout: str = READER_MESSAGE_LAYOUT_DEFAULT,
        base_url: str | None = None,
        provider: str = "openai",
        thinking: bool | None = None,
    ):
        build_reader_messages("", message_layout)
        self._chat = OpenAIChatCaller(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            base_url=base_url,
            thinking=thinking,
        )
        self.model_name = self._chat.model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.message_layout = message_layout
        self.provider = provider

    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        prompt = render_qa_prompt(prompt_template, memory=memory, question=question)
        messages = build_reader_messages(prompt, self.message_layout)
        return self._chat.complete(messages)


class AnthropicReader(Reader):
    """Anthropic Messages API reader. Infra only until a robustness-axis run."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = 64,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
        message_layout: str = READER_MESSAGE_LAYOUT_DEFAULT,
    ):
        build_reader_messages("", message_layout)
        self.spec = resolve_model(model)
        self.model_name = self.spec.model_id
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else 64
        self.message_layout = message_layout
        self.max_retries = max_retries
        self.min_request_interval_s = min_request_interval_s
        self.max_wait_s = max_wait_s
        self._last_request_t = 0.0
        load_env()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {api_key_env} in a repo-root .env file "
                f"(see .env.example) or export it before using Anthropic."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)

    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        prompt = render_qa_prompt(prompt_template, memory=memory, question=question)
        layout = build_reader_messages(prompt, self.message_layout)
        system = ""
        messages: list[dict[str, str]] = []
        for msg in layout:
            if msg["role"] == "system":
                system = ((system + "\n") if system else "") + msg["content"]
            else:
                messages.append({"role": msg["role"], "content": msg["content"]})
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            if self.min_request_interval_s > 0 and self._last_request_t:
                elapsed = time.time() - self._last_request_t
                if elapsed < self.min_request_interval_s:
                    time.sleep(self.min_request_interval_s - elapsed)
            t0 = time.time()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": int(self.max_tokens or 64),
                }
                if system:
                    kwargs["system"] = system
                if self.spec.supports_temperature:
                    kwargs["temperature"] = float(self.temperature)
                resp = self._client.messages.create(**anthropic_messages_kwargs(kwargs))
                self._last_request_t = time.time()
                latency = time.time() - t0
                text_parts = []
                for block in getattr(resp, "content", []) or []:
                    if getattr(block, "type", None) == "text":
                        text_parts.append(getattr(block, "text", "") or "")
                text = "".join(text_parts).strip()
                usage = {}
                if getattr(resp, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": getattr(resp.usage, "input_tokens", None),
                        "completion_tokens": getattr(resp.usage, "output_tokens", None),
                    }
                return text, {
                    "latency_s": round(latency, 3),
                    "usage": usage,
                    "attempts": attempt + 1,
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
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc


def get_reader(
    name: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int | None = 64,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
    message_layout: str = READER_MESSAGE_LAYOUT_DEFAULT,
    thinking: bool | None = None,
) -> Reader:
    """Build a mock, OpenAI-compatible, or Anthropic reader."""
    name = name.lower()
    if name == "mock":
        # Resolve aliases (luna → gpt-5.6-luna) so mock logs match live ids.
        model_name = resolve_model(model).model_id if model else "mock"
        return MockReader(model_name=model_name)
    if name in OPENAI_COMPAT_PROVIDERS:
        spec = OPENAI_COMPAT_PROVIDERS[name]
        return OpenAIReader(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=str(spec["api_key_env"]),
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            message_layout=message_layout,
            base_url=spec.get("base_url"),
            provider=name,
            thinking=thinking,
        )
    if name in ("anthropic", "claude"):
        return AnthropicReader(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            message_layout=message_layout,
        )
    raise ValueError(
        f"Unknown reader '{name}'. Use openai, deepseek, anthropic, or mock."
    )
