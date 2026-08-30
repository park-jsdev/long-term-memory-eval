"""Mem0 ADD / UPDATE / DELETE / NONE against a vector store."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import resolve_model
from ..prompts import load_prompt_template
from ..readers import OpenAIChatCaller
from .embeddings import Embedder
from .json_util import parse_json_object
from .schemas import ADD, DELETE, NONE, UPDATE, UPDATE_EVENTS, Fact, UpdateEvent
from .vector_store import VectorMemoryStore

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPDATE_PROMPT = ROOT / "prompts" / "mem0_update_v1.txt"


def parse_update_events(text: str) -> list[UpdateEvent]:
    """Parse updater JSON into events. Unknown event labels are skipped."""
    try:
        obj = parse_json_object(text)
    except ValueError:
        return []
    rows = obj.get("memory")
    if not isinstance(rows, list):
        return []
    events: list[UpdateEvent] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        event = str(row.get("event") or "").strip().upper()
        if event not in UPDATE_EVENTS:
            continue
        events.append(
            UpdateEvent(
                event=event,
                text=str(row.get("text") or "").strip(),
                local_id=str(row["id"]) if row.get("id") is not None else None,
                old_text=str(row["old_memory"]) if row.get("old_memory") else None,
            )
        )
    return events


def apply_update_events(
    store: VectorMemoryStore,
    events: list[UpdateEvent],
    *,
    candidates: list[Fact],
    embedder: Embedder,
    timestamp: str,
    source_turn_ids: list[str],
    speaker_index: str,
) -> list[dict[str, Any]]:
    """Apply ADD/UPDATE/DELETE/NONE. local_id indexes ``candidates`` (top-s)."""
    applied: list[dict[str, Any]] = []
    by_local = {str(i): fact for i, fact in enumerate(candidates)}
    for ev in events:
        if ev.event == NONE:
            applied.append({"event": NONE, "fact_id": ev.local_id, "text": ev.text})
            continue
        if ev.event == ADD:
            if not ev.text:
                continue
            fact = Fact(
                fact_id=store.next_fact_id(),
                text=ev.text,
                timestamp=timestamp,
                embedding=embedder.embed_one(ev.text),
                source_turn_ids=list(source_turn_ids),
                speaker_index=speaker_index,
            )
            store.add(fact)
            applied.append({"event": ADD, "fact_id": fact.fact_id, "text": fact.text})
            continue
        target = by_local.get(str(ev.local_id)) if ev.local_id is not None else None
        if target is None:
            applied.append(
                {
                    "event": ev.event,
                    "fact_id": ev.local_id,
                    "text": ev.text,
                    "skipped": "unknown_local_id",
                }
            )
            continue
        if ev.event == UPDATE:
            new_text = ev.text or target.text
            store.update(target.fact_id, new_text, embedder.embed_one(new_text))
            applied.append(
                {
                    "event": UPDATE,
                    "fact_id": target.fact_id,
                    "text": new_text,
                    "old_text": ev.old_text or target.text,
                }
            )
            continue
        if ev.event == DELETE:
            store.delete(target.fact_id)
            applied.append({"event": DELETE, "fact_id": target.fact_id, "text": target.text})
    return applied


class MemoryUpdater(ABC):
    """Decide ops for new facts vs top-s similar memories."""

    model_name: str
    provider: str

    @abstractmethod
    def propose(
        self,
        *,
        new_facts: list[str],
        candidates: list[Fact],
    ) -> tuple[list[UpdateEvent], dict[str, Any]]:
        ...


class MockMemoryUpdater(MemoryUpdater):
    """Offline updater: ADD every new fact (no UPDATE/DELETE)."""

    provider = "mock"

    def __init__(self, model_name: str = "mock"):
        self.model_name = resolve_model(model_name).model_id if model_name else "mock"
        self.last_prompt: str = ""

    def propose(
        self,
        *,
        new_facts: list[str],
        candidates: list[Fact],
    ) -> tuple[list[UpdateEvent], dict[str, Any]]:
        self.last_prompt = "\n".join(new_facts)
        events = [UpdateEvent(event=ADD, text=text) for text in new_facts if text.strip()]
        return events, {
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
            "role": "mem0_update",
        }


class OpenAIMemoryUpdater(MemoryUpdater):
    """Live 4-way updater. Prompt is mem0_update_v1."""

    provider = "openai"

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
    ):
        self._chat = OpenAIChatCaller(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
        )
        self.model_name = self._chat.model_name
        path = Path(prompt_path) if prompt_path else DEFAULT_UPDATE_PROMPT
        self.prompt_version, self.prompt_template = load_prompt_template(path)
        self.last_prompt: str = ""

    def propose(
        self,
        *,
        new_facts: list[str],
        candidates: list[Fact],
    ) -> tuple[list[UpdateEvent], dict[str, Any]]:
        existing = [
            {"id": str(i), "text": fact.text} for i, fact in enumerate(candidates)
        ]
        prompt = self.prompt_template.format(
            existing_memories=existing if existing else "[]",
            retrieved_facts=new_facts,
        )
        self.last_prompt = prompt
        messages = [
            {
                "role": "system",
                "content": "You manage memory with ADD, UPDATE, DELETE, or NONE. JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
        text, meta = self._chat.complete(messages)
        events = parse_update_events(text)
        meta = {**meta, "role": "mem0_update", "prompt_version": self.prompt_version}
        return events, meta


def get_memory_updater(
    name: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
    prompt_path: str | Path | None = None,
) -> MemoryUpdater:
    key = (name or "mock").strip().lower()
    if key == "mock":
        return MockMemoryUpdater(model_name=model or "mock")
    if key == "openai":
        return OpenAIMemoryUpdater(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            prompt_path=prompt_path,
        )
    raise ValueError(f"Unknown memory updater '{name}'. Use openai or mock.")
