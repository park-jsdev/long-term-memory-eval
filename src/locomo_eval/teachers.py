"""Write-path teachers: session dialog → summary or graph fragment.

Single-teacher session summaries stay on TeacherSessionMemoryBuilder.
Graph teachers (teacher_graph / pooled / fused) call extract_session_graph
and feed Mem0GraphMemory. Swap provider+model; gold answers never enter.
"""

from __future__ import annotations

import hashlib
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .teacher_callers import DEFAULT_TEACHER_THINKING, get_teacher_caller
from .mem0.graph_memory import mock_extract_entities, mock_extract_relations, normalize_entity_name
from .mem0.json_util import parse_json_object
from .models import resolve_model
from .prompts import (
    ROOT,
    TEACHER_GRAPH_V1,
    TEACHER_SESSION_V1,
    load_prompt_template,
    render_teacher_graph_prompt,
    render_teacher_session_prompt,
)

DEFAULT_TEACHER_PROMPT = ROOT / TEACHER_SESSION_V1
DEFAULT_GRAPH_PROMPT = ROOT / TEACHER_GRAPH_V1

KNOWN_TEACHER_PROVIDERS = ("openai", "anthropic", "deepseek", "mock", "codex")


def teacher_call_record(
    *,
    teacher: Teacher,
    meta: dict[str, Any],
    sample_id: str | None = None,
    session_id: str | int | None = None,
    session_index: int | None = None,
    output_text: str | None = None,
    entities: list[dict[str, str]] | None = None,
    relations: list[dict[str, str]] | None = None,
    session_text: str | None = None,
) -> dict[str, Any]:
    """One write-path LLM call for memory/teachers/ indexing."""
    ents = list(entities) if entities is not None else list(meta.get("entities") or [])
    rels = list(relations) if relations is not None else list(meta.get("relations") or [])
    out = output_text if output_text is not None else str(meta.get("output_text") or "")
    rec: dict[str, Any] = {
        "sample_id": sample_id,
        "session_id": session_id,
        "session_index": session_index,
        "teacher_id": teacher.teacher_id,
        "provider": teacher.provider,
        "model": teacher.model_name,
        "role": meta.get("role"),
        "prompt_version": meta.get("prompt_version"),
        "thinking": meta.get("thinking"),
        "thinking_supported": meta.get("thinking_supported"),
        "reasoning": meta.get("reasoning") or "",
        "reasoning_tokens": meta.get("reasoning_tokens"),
        "reasoning_effort": meta.get("reasoning_effort"),
        "output_text": out,
        "n_entities": len(ents),
        "n_relations": len(rels),
        "entities": ents,
        "relations": rels,
        "latency_s": meta.get("latency_s"),
        "usage": meta.get("usage") or {},
        "parse": meta.get("parse"),
    }
    if session_text is not None:
        rec["session_text_sha256"] = hashlib.sha256(session_text.encode("utf-8")).hexdigest()
        rec["n_session_chars"] = len(session_text)
    return rec


class Teacher(ABC):
    """Write-path LLM: one session's turns → summary text and/or graph triples.

    Called from TeacherSessionMemoryBuilder or TeacherOrchestrator.
    Must not see gold answers.
    """

    model_name: str
    provider: str
    teacher_id: str

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

    @abstractmethod
    def extract_session_graph(
        self,
        *,
        session_text: str,
        user_id: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
        """Return (entities, relations, call_meta) for Mem0GraphMemory."""
        ...

    def ping(self) -> tuple[str, dict[str, Any]]:
        """Tiny live/mock sanity call. Not used by the QA pipeline."""
        return "pong", {
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
            "provider": self.provider,
            "role": "teacher_ping",
        }


class MockTeacher(Teacher):
    """Offline teacher. Summary and graph triples are tagged with model_id."""

    provider = "mock"

    def __init__(
        self,
        model_name: str = "mock",
        teacher_id: str | None = None,
        thinking: bool = DEFAULT_TEACHER_THINKING,
    ):
        self.model_name = resolve_model(model_name).model_id if model_name else "mock"
        self.teacher_id = teacher_id or self.model_name
        self.thinking = bool(thinking)

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
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
            "provider": self.provider,
            "role": "teacher",
            "thinking": self.thinking,
            "thinking_supported": False,
            "reasoning": "[mock-thinking]" if self.thinking else "",
            "output_text": text,
        }

    def extract_session_graph(
        self,
        *,
        session_text: str,
        user_id: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
        entities = mock_extract_entities(session_text, user_id)
        relations = mock_extract_relations(session_text, entities, user_id)
        # Teacher identity is a dummy edge so pool/fusion tests can see who wrote.
        marker = normalize_entity_name(self.teacher_id or self.model_name)
        entities = list(entities) + [{"entity": marker, "entity_type": "other"}]
        relations = list(relations) + [
            {
                "source": normalize_entity_name(user_id),
                "relationship": "taught_by",
                "target": marker,
            }
        ]
        meta = {
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
            "provider": self.provider,
            "role": "teacher_graph",
            "thinking": self.thinking,
            "thinking_supported": False,
            "reasoning": "[mock-thinking]" if self.thinking else "",
            "output_text": "",
            "entities": entities,
            "relations": relations,
            "parse": "mock",
        }
        return entities, relations, meta

    def ping(self) -> tuple[str, dict[str, Any]]:
        return "pong", {
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
            "provider": self.provider,
            "role": "teacher_ping",
        }


class ChatTeacher(Teacher):
    """Live teacher via OpenAI, Anthropic, or DeepSeek TeacherCaller."""

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
        prompt_path: str | Path | None = None,
        graph_prompt_path: str | Path | None = None,
        teacher_id: str | None = None,
        thinking: bool = DEFAULT_TEACHER_THINKING,
        thinking_budget_tokens: int = 1024,
    ):
        self.provider = (provider or "openai").strip().lower()
        self.thinking = bool(thinking)
        self._caller = get_teacher_caller(
            self.provider,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        self.model_name = self._caller.model_name
        self.teacher_id = teacher_id or self.model_name
        path = Path(prompt_path) if prompt_path else DEFAULT_TEACHER_PROMPT
        self.prompt_version, self.prompt_template = load_prompt_template(path)
        gpath = Path(graph_prompt_path) if graph_prompt_path else DEFAULT_GRAPH_PROMPT
        self.graph_prompt_version, self.graph_prompt_template = load_prompt_template(gpath)

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
        text, meta = self._caller.complete(messages)
        meta = {
            **meta,
            "role": "teacher",
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "output_text": text,
        }
        return text, meta

    def extract_session_graph(
        self,
        *,
        session_text: str,
        user_id: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
        prompt = render_teacher_graph_prompt(
            self.graph_prompt_template,
            user_id=user_id,
            text=session_text,
        )
        messages = [
            {
                "role": "system",
                "content": "You extract knowledge-graph triples. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
        raw, meta = self._caller.complete(messages)
        meta = {
            **meta,
            "role": "teacher_graph",
            "prompt_version": self.graph_prompt_version,
            "provider": self.provider,
            "output_text": raw,
        }
        try:
            obj = parse_json_object(raw)
        except ValueError:
            entities = mock_extract_entities(session_text, user_id)
            relations = mock_extract_relations(session_text, entities, user_id)
            return entities, relations, {**meta, "parse": "fallback_mock"}
        entities = []
        for row in obj.get("entities") or []:
            if isinstance(row, dict) and row.get("entity"):
                entities.append(
                    {
                        "entity": str(row["entity"]),
                        "entity_type": str(row.get("entity_type") or "other"),
                    }
                )
        relations = []
        for row in obj.get("relations") or []:
            if isinstance(row, dict) and row.get("source") and row.get("target"):
                relations.append(
                    {
                        "source": str(row["source"]),
                        "relationship": str(row.get("relationship") or "related_to"),
                        "target": str(row["target"]),
                    }
                )
        return entities, relations, {**meta, "parse": "ok"}

    def ping(self) -> tuple[str, dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": "Reply with the single word pong.",
            },
            {"role": "user", "content": "Reply with pong only."},
        ]
        text, meta = self._caller.complete(messages, thinking=False)
        return text, {**meta, "role": "teacher_ping", "provider": self.provider}


class OpenAITeacher(ChatTeacher):
    """Live teacher via Chat Completions. Kept as the openai-only constructor."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
        prompt_path: str | Path | None = None,
        graph_prompt_path: str | Path | None = None,
        teacher_id: str | None = None,
        thinking: bool = DEFAULT_TEACHER_THINKING,
        thinking_budget_tokens: int = 1024,
    ):
        super().__init__(
            provider="openai",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            prompt_path=prompt_path,
            graph_prompt_path=graph_prompt_path,
            teacher_id=teacher_id,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )


class CodexTeacher(Teacher):
    """Codex CLI write-path teacher over a disposable session workspace.

    The ordinary reader never sees this workspace.  Codex must read the
    session file and return the requested summary/graph in the same contracts
    consumed by the existing teacher builders.
    """

    provider = "codex"

    def __init__(
        self,
        model: str,
        *,
        prompt_path: str | Path | None = None,
        graph_prompt_path: str | Path | None = None,
        teacher_id: str | None = None,
        thinking: bool = False,
        **_unused: Any,
    ):
        self.model_name = model
        self.teacher_id = teacher_id or f"codex/{model}"
        self.thinking = bool(thinking)
        path = Path(prompt_path) if prompt_path else DEFAULT_TEACHER_PROMPT
        self.prompt_version, self.prompt_template = load_prompt_template(path)
        gpath = Path(graph_prompt_path) if graph_prompt_path else DEFAULT_GRAPH_PROMPT
        self.graph_prompt_version, self.graph_prompt_template = load_prompt_template(gpath)

    def _run(self, session_text: str, prompt: str) -> tuple[str, dict[str, Any]]:
        from .agents.adapters.codex import CodexAgentRunner
        from .agents.protocol import AgentRequest

        with tempfile.TemporaryDirectory(prefix="locomo_codex_writer_") as temp:
            root = Path(temp)
            (root / "INDEX.md").write_text("- `sessions/session_1.md`\n", encoding="utf-8")
            sessions = root / "sessions"
            sessions.mkdir()
            (sessions / "session_1.md").write_text(session_text, encoding="utf-8")
            runner = CodexAgentRunner(
                model_name=self.model_name,
                persist_memory=False,
                tools="native",
            )
            result = runner.run(
                AgentRequest(
                    workspace_dir=root,
                    sample_id="write",
                    question_id="write",
                    question="Write the requested memory artifact.",
                    prompt=(
                        f"{prompt}\n\nRead sessions/session_1.md before answering. "
                        'Return a JSON object with only "answer".'
                    ),
                )
            )
        return result.predicted_answer, {
            **result.call_meta,
            "latency_s": result.latency_s,
            "usage": result.usage,
            "reasoning_tokens": result.usage.get("reasoning_tokens"),
            "thinking": self.thinking,
            "thinking_supported": False,
            "role": "agent_memory_writer",
            "provider": self.provider,
            "harness_failed": result.trajectory.harness_failed,
            "harness_failure_reason": result.trajectory.harness_failure_reason,
        }

    def summarize_session(
        self, *, session_text: str, date_time: str, speaker_a: str, speaker_b: str
    ) -> tuple[str, dict[str, Any]]:
        prompt = render_teacher_session_prompt(
            self.prompt_template,
            date=date_time,
            session_text="Read sessions/session_1.md instead of pasted text.",
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )
        text, meta = self._run(session_text, prompt)
        return text, {**meta, "prompt_version": self.prompt_version, "output_text": text}

    def extract_session_graph(
        self, *, session_text: str, user_id: str
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
        prompt = render_teacher_graph_prompt(
            self.graph_prompt_template,
            user_id=user_id,
            text="Read sessions/session_1.md instead of pasted text.",
        )
        raw, meta = self._run(session_text, prompt)
        try:
            obj = parse_json_object(raw)
        except ValueError:
            entities = mock_extract_entities(session_text, user_id)
            relations = mock_extract_relations(session_text, entities, user_id)
            return entities, relations, {
                **meta, "prompt_version": self.graph_prompt_version, "parse": "fallback_mock"
            }
        entities = [
            {"entity": str(row["entity"]), "entity_type": str(row.get("entity_type") or "other")}
            for row in obj.get("entities") or []
            if isinstance(row, dict) and row.get("entity")
        ]
        relations = [
            {
                "source": str(row["source"]),
                "relationship": str(row.get("relationship") or "related_to"),
                "target": str(row["target"]),
            }
            for row in obj.get("relations") or []
            if isinstance(row, dict) and row.get("source") and row.get("target")
        ]
        return entities, relations, {
            **meta, "prompt_version": self.graph_prompt_version, "parse": "ok", "output_text": raw
        }


def get_teacher(
    name: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
    prompt_path: str | Path | None = None,
    graph_prompt_path: str | Path | None = None,
    teacher_id: str | None = None,
    thinking: bool = DEFAULT_TEACHER_THINKING,
    thinking_budget_tokens: int = 1024,
) -> Teacher:
    name = (name or "mock").lower()
    if name == "mock":
        return MockTeacher(
            model_name=model or "mock",
            teacher_id=teacher_id,
            thinking=thinking,
        )
    if name in ("openai", "anthropic", "deepseek"):
        return ChatTeacher(
            provider=name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            prompt_path=prompt_path,
            graph_prompt_path=graph_prompt_path,
            teacher_id=teacher_id,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
    if name == "codex":
        return CodexTeacher(
            model=model,
            prompt_path=prompt_path,
            graph_prompt_path=graph_prompt_path,
            teacher_id=teacher_id,
            thinking=thinking,
        )
    raise ValueError(
        f"Unknown teacher '{name}'. Use openai, anthropic, deepseek, codex, or mock."
    )

