"""Write-path LLM callers for teachers (OpenAI, Anthropic, DeepSeek).

This module is **not** the frozen answer reader. It supplies provider clients
used exclusively on the **write path**:

    ModelOrchestrator → Writer (``teachers.py``) → WriterCaller → API

``readers.py`` owns ``OpenAIChatCaller`` for QA/autorater; teachers reuse that
class for OpenAI/DeepSeek via ``get_writer_caller()``. Anthropic teachers use
``AnthropicWriterCaller`` here.

Ping / smoke: ``python -m src.locomo_eval.ping_writers`` (via ``get_writer``).
"""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any

from .env import load_env
from .models import anthropic_messages_kwargs, resolve_model
from .readers import OpenAIChatCaller, _is_rate_limit_error, _retry_after_seconds
from .reasoning_extractor import openai_reasoning_text, split_anthropic_content

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
ANTHROPIC_MIN_THINKING_BUDGET = 1024
DEFAULT_WRITER_THINKING = True

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_MOCK = "mock"

PROVIDER_API_KEY_ENV = {
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    PROVIDER_DEEPSEEK: "DEEPSEEK_API_KEY",
}


class WriterCaller(ABC):
    """messages → (text, call_meta). Used by ``ChatWriter`` on the write path."""

    model_name: str
    provider: str

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        create_extra: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        ...


class MockWriterCaller(WriterCaller):
    """Offline stand-in. Replies are tagged with the model id."""

    provider = PROVIDER_MOCK

    def __init__(self, model: str = "mock", thinking: bool = False):
        self.spec = resolve_model(model) if model and model != "mock" else None
        self.model_name = self.spec.model_id if self.spec else "mock"
        self.thinking = bool(thinking)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        create_extra: dict[str, Any] | None = None,
        thinking: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        last = messages[-1]["content"] if messages else ""
        use_thinking = self.thinking if thinking is None else bool(thinking)
        if "pong" in last.lower() or last.strip().lower().startswith("reply with"):
            text = "pong"
        else:
            text = f"[{self.model_name}] mock-ok"
        reasoning = "[mock-thinking]" if use_thinking else ""
        return text, {
            "latency_s": 0.0,
            "usage": {},
            "attempts": 1,
            "model": self.model_name,
            "family": self.spec.family if self.spec else "mock",
            "provider": self.provider,
            "thinking": use_thinking,
            "thinking_supported": False,
            "reasoning": reasoning,
        }


class AnthropicWriterCaller(WriterCaller):
    """Anthropic Messages API for graph/session teachers."""

    provider = PROVIDER_ANTHROPIC

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = 512,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
        thinking: bool = True,
        thinking_budget_tokens: int = ANTHROPIC_MIN_THINKING_BUDGET,
    ):
        self.spec = resolve_model(model)
        self.model_name = self.spec.model_id
        self.temperature = temperature
        self.max_tokens = 512 if max_tokens is None else int(max_tokens)
        if thinking:
            from .readers import THINKING_HTTP_TIMEOUT_S

            timeout_s = max(float(timeout_s), THINKING_HTTP_TIMEOUT_S)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.min_request_interval_s = min_request_interval_s
        self.max_wait_s = max_wait_s
        self.thinking = bool(thinking)
        self.thinking_budget_tokens = int(thinking_budget_tokens)
        self._last_request_t = 0.0

        load_env()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {api_key_env} in a repo-root .env file "
                f"(see .env.example) or export it in the shell before using Anthropic."
            )
        workspace_id = (
            os.environ.get("ANTHROPIC_WORKSPACE_ID")
            or os.environ.get("ANTHROPIC_WORKSPACE")
            or ""
        ).strip()
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc
        headers = {}
        if workspace_id:
            headers["anthropic-workspace-id"] = workspace_id
        self._workspace_id = workspace_id
        self._client = Anthropic(
            api_key=api_key,
            timeout=timeout_s,
            default_headers=headers or None,
        )

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
        create_extra: dict[str, Any] | None = None,
        thinking: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        system_parts: list[str] = []
        chat_messages: list[dict[str, str]] = []
        for row in messages:
            role = row.get("role") or "user"
            content = row.get("content") or ""
            if role == "system":
                system_parts.append(content)
            else:
                chat_messages.append({"role": role, "content": content})
        if not chat_messages:
            chat_messages = [{"role": "user", "content": " "}]

        use_thinking = self.thinking if thinking is None else bool(thinking)
        max_tokens = max(self.max_tokens, 16)
        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system_parts:
            create_kwargs["system"] = "\n".join(system_parts)
        if use_thinking:
            budget = max(ANTHROPIC_MIN_THINKING_BUDGET, self.thinking_budget_tokens)
            create_kwargs["max_tokens"] = max(max_tokens, budget + 256)
            create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif self.spec.supports_temperature:
            create_kwargs["temperature"] = float(self.temperature)
        if create_extra:
            create_kwargs.update(create_extra)

        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            t0 = time.time()
            try:
                resp = self._client.messages.create(
                    **anthropic_messages_kwargs(create_kwargs)
                )
                self._last_request_t = time.time()
                latency = time.time() - t0
                text, reasoning = split_anthropic_content(resp.content)
                usage = {}
                if resp.usage is not None:
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
                    "provider": self.provider,
                    "thinking": use_thinking,
                    "thinking_supported": True,
                    "reasoning": reasoning,
                }
            except Exception as exc:
                last_exc = exc
                if _is_missing_anthropic_workspace(exc):
                    raise RuntimeError(
                        "Anthropic identity-linked API keys require ANTHROPIC_WORKSPACE_ID. "
                        "Copy the id from Claude Console → Settings → Workspaces "
                        "(it looks like wrkspc_...) into repo-root .env, or create a "
                        "key scoped to one workspace so the header is not needed."
                    ) from exc
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


def get_writer_caller(
    provider: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = 512,
    timeout_s: float = 60.0,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
    thinking: bool = DEFAULT_WRITER_THINKING,
    thinking_budget_tokens: int = ANTHROPIC_MIN_THINKING_BUDGET,
) -> WriterCaller:
    """Build a mock / OpenAI / Anthropic / DeepSeek caller for teachers."""
    name = (provider or PROVIDER_MOCK).strip().lower()
    if name == PROVIDER_MOCK:
        return MockWriterCaller(model=model or "mock", thinking=thinking)
    if name == PROVIDER_OPENAI:
        return OpenAIChatCaller(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=PROVIDER_API_KEY_ENV[PROVIDER_OPENAI],
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            thinking=thinking,
        )
    if name == PROVIDER_DEEPSEEK:
        return OpenAIChatCaller(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=PROVIDER_API_KEY_ENV[PROVIDER_DEEPSEEK],
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            base_url=DEEPSEEK_BASE_URL,
            thinking=thinking,
        )
    if name == PROVIDER_ANTHROPIC:
        return AnthropicWriterCaller(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=PROVIDER_API_KEY_ENV[PROVIDER_ANTHROPIC],
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
    raise ValueError(
        f"Unknown teacher provider '{provider}'. Use openai, anthropic, deepseek, or mock."
    )


def _is_missing_anthropic_workspace(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "anthropic-workspace-id is required" in msg
