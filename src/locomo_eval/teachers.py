"""Single-teacher write step: session dialog → summary string.

This is the first live-teacher seam for teacher_session_summaries. Not multi-teacher fusion.
Swap ``teacher.model`` within a family to test whether the memory designer
changes Memory.text (and downstream logs/scores).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .utils.llm_response_hash import LlmResponseHash
from .models import resolve_model
from .prompts import load_prompt_template, render_teacher_session_prompt
from .readers import OpenAIChatCaller


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER_PROMPT = ROOT / "prompts" / "teacher_session_v1.txt"


class Teacher(ABC):
    """Write-path LLM: one session's turns → summary text.

    Called from TeacherSessionMemoryBuilder. Must not see gold answers.
    """

    model_name: str
    provider: str

    @abstractmethod
    def summarize_session(
        self,
        *,
        session_text: str,
        date_time: str,
        speaker_a: str,
        speaker_b: str,
    ) -> tuple[str, dict[str, Any]]:
        """Return (summary_text, call_meta)."""
        ...


class MockTeacher(Teacher):
    """Offline teacher. Summary is tagged with model_id so swaps are visible."""

    provider = "mock"

    def __init__(self, model_name: str = "mock"):
        self.model_name = resolve_model(model_name).model_id if model_name else "mock"

    def summarize_session(
        self,
        *,
        session_text: str,
        date_time: str,
        speaker_a: str,
        speaker_b: str,
    ) -> tuple[str, dict[str, Any]]:
        # Model id is part of the text so two family members cannot collide.
        body = " ".join(session_text.split())
        if len(body) > 240:
            body = body[:240] + "…"
        text = f"[{self.model_name}] {date_time}: {body}".strip()
        return text, {
            "cached": False,
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
            "role": "teacher",
        }


class OpenAITeacher(Teacher):
    """Live teacher via the same Chat Completions helper as OpenAIReader."""

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
        self._chat = OpenAIChatCaller(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_response_hash=llm_response_hash,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
        )
        self.model_name = self._chat.model_name
        path = Path(prompt_path) if prompt_path else DEFAULT_TEACHER_PROMPT
        self.prompt_version, self.prompt_template = load_prompt_template(path)

    def summarize_session(
        self,
        *,
        session_text: str,
        date_time: str,
        speaker_a: str,
        speaker_b: str,
    ) -> tuple[str, dict[str, Any]]:
        prompt = render_teacher_session_prompt(
            self.prompt_template,
            date=date_time,
            session_text=session_text,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )
        messages = [
            {
                "role": "system",
                "content": "You write concise session memories. Do not answer questions.",
            },
            {"role": "user", "content": prompt},
        ]
        text, meta = self._chat.complete(
            messages,
            request_extra={
                "role": "teacher",
                "prompt_version": self.prompt_version,
                "prompt": prompt,
            },
        )
        meta = {**meta, "role": "teacher", "prompt_version": self.prompt_version}
        return text, meta


def get_teacher(
    name: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    llm_response_hash: LlmResponseHash | None = None,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
    prompt_path: str | Path | None = None,
) -> Teacher:
    if llm_response_hash is not None:
        raise ValueError("LLM response caching is disabled for evaluation pipelines.")
    name = (name or "mock").lower()
    if name == "mock":
        return MockTeacher(model_name=model or "mock")
    if name == "openai":
        return OpenAITeacher(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_response_hash=llm_response_hash,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            prompt_path=prompt_path,
        )
    raise ValueError(f"Unknown teacher '{name}'. Use openai or mock.")
