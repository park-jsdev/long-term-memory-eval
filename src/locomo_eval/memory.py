"""Memory builders: conversation → Memory context for the answer LLM.

Phase 1: only SessionSummaryMemoryBuilder.
Later: raw turns, observations, structured multi-teacher memories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import Conversation, Memory, Question


class MemoryBuilder(ABC):
    """Common interface so readers/evaluators stay memory-agnostic."""

    name: str = "base"

    @abstractmethod
    def build(self, conversation: Conversation, question: Question) -> Memory:
        ...


class SessionSummaryMemoryBuilder(MemoryBuilder):
    """Concatenate LoCoMo-provided session summaries in chronological order.

    No retrieval yet — the full summary bank is always the context.
    """

    name = "session_summary"

    def build(self, conversation: Conversation, question: Question) -> Memory:
        chunks: list[str] = []
        source_ids: list[str] = []
        for sid, text in conversation.chronological_summaries():
            if not text or not str(text).strip():
                continue
            chunks.append(f"[Session {sid}]\n{text.strip()}")
            source_ids.append(f"session_{sid}_summary")

        if not chunks:
            # Fail soft so a bad sample does not crash the whole run.
            text = "(No session summaries available for this conversation.)"
        else:
            text = "\n\n".join(chunks)

        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
        )


def get_memory_builder(name: str) -> MemoryBuilder:
    registry = {
        SessionSummaryMemoryBuilder.name: SessionSummaryMemoryBuilder,
    }
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(f"Unknown memory builder '{name}'. Known: {known}")
    return registry[name]()
