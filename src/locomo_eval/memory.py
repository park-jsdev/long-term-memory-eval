"""Memory builders: conversation → Memory context for the answer LLM.

Experimental middle of the sandwich. Readers/metrics stay agnostic.

Draft conditions:
  C0  c0_raw              — raw dialogue turns (optional char budget)
  C1  c1_session_summary  — LoCoMo-provided session summaries

See docs/reports/engineering_notebook.md.
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


class RawConversationMemoryBuilder(MemoryBuilder):
    """C0: chronological raw turns. No structure beyond session/date headers.

    Optional max_chars keeps very long conversations within a rough budget
    when the full dialog would blow the answer-model context. Truncation is
    from the *start* (keep recent sessions) so late evidence is more likely
    retained — simple heuristic, not true retrieval.
    """

    name = "c0_raw"

    def __init__(self, max_chars: int | None = None):
        self.max_chars = max_chars

    def build(self, conversation: Conversation, question: Question) -> Memory:
        chunks: list[str] = []
        source_ids: list[str] = []
        header = (
            f"Conversation between {conversation.speaker_a} and "
            f"{conversation.speaker_b}.\n"
        )
        chunks.append(header)

        for session in conversation.sessions:
            if not session.turns:
                continue
            block_lines = [
                f"DATE: {session.date_time}",
                f"SESSION {session.session_id}:",
            ]
            for turn in session.turns:
                line = f'{turn.speaker}: {turn.text}'
                if turn.blip_caption:
                    line += f" [image: {turn.blip_caption}]"
                if turn.dia_id:
                    line += f" ({turn.dia_id})"
                block_lines.append(line)
                source_ids.append(turn.dia_id or f"session_{session.session_id}")
            chunks.append("\n".join(block_lines))

        text = "\n\n".join(chunks).strip()
        if not text:
            text = "(No dialogue turns available for this conversation.)"

        if self.max_chars is not None and len(text) > self.max_chars:
            # Keep the tail (more recent); mark truncation for audit.
            text = (
                "[... earlier turns truncated to max_chars ...]\n"
                + text[-self.max_chars :]
            )

        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
        )


class SessionSummaryMemoryBuilder(MemoryBuilder):
    """C1 draft: concatenate LoCoMo session summaries chronologically.

    Placeholder for a true single-teacher structured-memory write step:
    same interface (→ Memory.text), different implementation later.
    """

    name = "c1_session_summary"

    def build(self, conversation: Conversation, question: Question) -> Memory:
        chunks: list[str] = []
        source_ids: list[str] = []
        for sid, text in conversation.chronological_summaries():
            if not text or not str(text).strip():
                continue
            chunks.append(f"[Session {sid}]\n{text.strip()}")
            source_ids.append(f"session_{sid}_summary")

        if not chunks:
            text = "(No session summaries available for this conversation.)"
        else:
            text = "\n\n".join(chunks)

        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
        )


# Registry name → constructor kwargs factory
_BUILDERS: dict[str, type[MemoryBuilder]] = {
    RawConversationMemoryBuilder.name: RawConversationMemoryBuilder,
    SessionSummaryMemoryBuilder.name: SessionSummaryMemoryBuilder,
}

# Stable aliases (older/shorter names)
_ALIASES: dict[str, str] = {
    "c0": RawConversationMemoryBuilder.name,
    "raw": RawConversationMemoryBuilder.name,
    "raw_dialog": RawConversationMemoryBuilder.name,
    "c1": SessionSummaryMemoryBuilder.name,
    "session_summary": SessionSummaryMemoryBuilder.name,
}


def resolve_memory_name(name: str) -> str:
    key = name.strip().lower() if name else ""
    if key in _BUILDERS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"Unknown memory builder '{name}'. "
        f"Known: {sorted(_BUILDERS)} aliases={sorted(_ALIASES)}"
    )


def get_memory_builder(
    name: str,
    max_chars: int | None = None,
) -> MemoryBuilder:
    resolved = resolve_memory_name(name)
    cls = _BUILDERS[resolved]
    if resolved == RawConversationMemoryBuilder.name:
        return cls(max_chars=max_chars)
    return cls()


def is_question_independent(name: str) -> bool:
    """True if memory text does not depend on the question (safe to cache per sample)."""
    resolved = resolve_memory_name(name)
    return resolved in (
        RawConversationMemoryBuilder.name,
        SessionSummaryMemoryBuilder.name,
    )
