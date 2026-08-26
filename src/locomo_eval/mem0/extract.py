"""Mem0 fact extraction from user-role messages in a pair."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import resolve_model
from ..prompts import load_prompt_template
from ..readers import OpenAIChatCaller
from ..utils.llm_response_hash import LlmResponseHash
from .ingest import user_messages_text
from .json_util import parse_json_object
from .schemas import MessagePair

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTRACT_PROMPT = ROOT / "prompts" / "mem0_extract_v1.txt"


def _reject_response_hash(llm_response_hash: LlmResponseHash | None) -> None:
    if llm_response_hash is not None:
        raise ValueError(
            "Mem0 FactExtractor factory rejects llm_response_hash. "
            "Write-index resume is per-sample JSON dumps, not LlmResponseHash."
        )


class FactExtractor(ABC):
    """MessagePair (user text) → candidate facts Ω. Must not see gold."""

    model_name: str
    provider: str

    @abstractmethod
    def extract(self, pair: MessagePair) -> tuple[list[str], dict[str, Any]]:
        """Return (fact strings, call_meta)."""
        ...


def render_extract_prompt(template: str, pair: MessagePair) -> str:
    return template.format(
        session_date=pair.timestamp or "unknown",
        user_messages=user_messages_text(pair) or "(no user messages in this pair)",
    )


class MockFactExtractor(FactExtractor):
    """Offline extractor: one fact per user-role message (no API)."""

    provider = "mock"

    def __init__(self, model_name: str = "mock"):
        self.model_name = resolve_model(model_name).model_id if model_name else "mock"
        self.last_prompt: str = ""
        self.last_user_text: str = ""

    def extract(self, pair: MessagePair) -> tuple[list[str], dict[str, Any]]:
        self.last_user_text = user_messages_text(pair)
        self.last_prompt = self.last_user_text
        facts = []
        for msg in pair.messages:
            if msg.role != "user":
                continue
            body = msg.content.strip()
            if body:
                facts.append(body)
        return facts, {
            "cached": False,
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
            "role": "mem0_extract",
        }


class OpenAIFactExtractor(FactExtractor):
    """Live extract via Chat Completions. Prompt is mem0_extract_v1."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        llm_response_hash: LlmResponseHash | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
        prompt_path: str | Path | None = None,
    ):
        _reject_response_hash(llm_response_hash)
        self._chat = OpenAIChatCaller(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_response_hash=None,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
        )
        self.model_name = self._chat.model_name
        path = Path(prompt_path) if prompt_path else DEFAULT_EXTRACT_PROMPT
        self.prompt_version, self.prompt_template = load_prompt_template(path)
        self.last_prompt: str = ""

    def extract(self, pair: MessagePair) -> tuple[list[str], dict[str, Any]]:
        prompt = render_extract_prompt(self.prompt_template, pair)
        self.last_prompt = prompt
        messages = [
            {
                "role": "system",
                "content": "You extract user facts as JSON. Do not answer questions.",
            },
            {"role": "user", "content": prompt},
        ]
        text, meta = self._chat.complete(
            messages,
            request_extra={
                "role": "mem0_extract",
                "prompt_version": self.prompt_version,
                "prompt": prompt,
            },
        )
        facts = _facts_from_response(text)
        meta = {**meta, "role": "mem0_extract", "prompt_version": self.prompt_version}
        return facts, meta


def _facts_from_response(text: str) -> list[str]:
    try:
        obj = parse_json_object(text)
    except ValueError:
        return []
    raw = obj.get("facts")
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def get_fact_extractor(
    name: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    llm_response_hash: LlmResponseHash | None = None,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
    prompt_path: str | Path | None = None,
) -> FactExtractor:
    _reject_response_hash(llm_response_hash)
    key = (name or "mock").strip().lower()
    if key == "mock":
        return MockFactExtractor(model_name=model or "mock")
    if key == "openai":
        return OpenAIFactExtractor(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_response_hash=None,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            prompt_path=prompt_path,
        )
    raise ValueError(f"Unknown fact extractor '{name}'. Use openai or mock.")
