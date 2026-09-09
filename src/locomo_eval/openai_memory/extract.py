"""Extract timestamped memory entries from a full conversation transcript.

Paper protocol: privileged access to *all* extracted memories at answer
time (no top-k). This extractor is the write half; retrieve is concatenate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import resolve_model
from ..prompts import load_prompt_template
from ..readers import OpenAIChatCaller
from ..schemas import Conversation

SCHEMA_VERSION = "openai_memory_index.v1"

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTRACT_PROMPT = ROOT / "prompts" / "openai_memory_extract_v1.txt"


@dataclass
class ExtractedMemory:
    speaker: str
    timestamp: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ExtractedMemory:
        return cls(
            speaker=str(row.get("speaker") or ""),
            timestamp=str(row.get("timestamp") or ""),
            text=str(row.get("text") or ""),
        )


class OpenAIMemoryExtractor(ABC):
    """Conversation transcript → memory entries. Must not see gold."""

    model_name: str
    provider: str

    @abstractmethod
    def extract(self, conversation: Conversation, transcript: str) -> tuple[list[ExtractedMemory], dict[str, Any]]:
        ...


class MockOpenAIMemoryExtractor(OpenAIMemoryExtractor):
    """Offline: one memory per dialog turn (no API)."""

    provider = "mock"

    def __init__(self, model_name: str = "mock"):
        self.model_name = resolve_model(model_name).model_id if model_name else "mock"

    def extract(
        self, conversation: Conversation, transcript: str
    ) -> tuple[list[ExtractedMemory], dict[str, Any]]:
        memories: list[ExtractedMemory] = []
        for session in conversation.sessions:
            ts = session.date_time or "unknown"
            for turn in session.turns:
                body = (turn.text or "").strip()
                if not body:
                    continue
                memories.append(
                    ExtractedMemory(
                        speaker=turn.speaker,
                        timestamp=ts,
                        text=body,
                    )
                )
        return memories, {
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
            "role": "openai_memory_extract",
            "schema_version": SCHEMA_VERSION,
        }


class LiveOpenAIMemoryExtractor(OpenAIMemoryExtractor):
    """Chat Completions extract using prompts/openai_memory_extract_v1.txt."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
        prompt_path: str | Path | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
    ):
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
        )
        self.model_name = self._chat.model_name
        path = Path(prompt_path) if prompt_path else DEFAULT_EXTRACT_PROMPT
        self.prompt_version, self.prompt_template = load_prompt_template(path)

    def extract(
        self, conversation: Conversation, transcript: str
    ) -> tuple[list[ExtractedMemory], dict[str, Any]]:
        prompt = self.prompt_template.replace("{transcript}", transcript)
        messages = [
            {
                "role": "system",
                "content": "Extract memory JSON. Do not answer questions.",
            },
            {"role": "user", "content": prompt},
        ]
        text, meta = self._chat.complete(
            messages,
            create_extra={"response_format": {"type": "json_object"}},
        )
        from ..mem0.json_util import parse_json_object

        obj = parse_json_object(text)
        raw = obj.get("memories") or []
        memories = [
            ExtractedMemory.from_dict(row)
            for row in raw
            if isinstance(row, dict) and str(row.get("text") or "").strip()
        ]
        meta = {
            **meta,
            "role": "openai_memory_extract",
            "prompt_version": self.prompt_version,
            "schema_version": SCHEMA_VERSION,
        }
        return memories, meta


def get_openai_memory_extractor(
    name: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
    prompt_path: str | Path | None = None,
) -> OpenAIMemoryExtractor:
    key = (name or "mock").strip().lower()
    if key == "mock":
        return MockOpenAIMemoryExtractor(model_name=model)
    if key == "openai":
        return LiveOpenAIMemoryExtractor(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            prompt_path=prompt_path,
        )
    raise ValueError(f"Unknown openai_memory extractor '{name}'. Use openai or mock.")
