"""Memory builders: conversation → Memory context for the answer LLM.

Experimental middle of the sandwich. Readers/metrics stay agnostic.

Current conditions (ids are what reviewers see in logs):
  raw_chunks                  — raw dialogue turns (optional char budget)
  session_summaries           — LoCoMo-provided session summaries
  teacher_session_summaries   — live single-teacher session summaries (model-swappable)
  mem0 / mem0g                — load a Mem0 write-index dump + cosine retrieve (no re-extract)

See docs/reports/engineering_notebook.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import Conversation, Memory, Question, Session

if TYPE_CHECKING:
    from .teachers import Teacher
    from .mem0.embeddings import Embedder


class MemoryBuilder(ABC):
    """Sandwich middle: Conversation (+ Question) → Memory.

    Called from run_locomo_pipeline_with_memory_config (one YAML per call).
    Swap the subclass without touching the reader or metrics.
    """

    name: str = "base"

    @abstractmethod
    def build(self, conversation: Conversation, question: Question) -> Memory:
        ...


def format_session_turns(session: Session) -> str:
    """Speaker lines for one session. Shared by raw_chunks dump and teacher input."""
    lines: list[str] = []
    for turn in session.turns:
        line = f"{turn.speaker}: {turn.text}"
        if turn.blip_caption:
            line += f" [image: {turn.blip_caption}]"
        if turn.dia_id:
            line += f" ({turn.dia_id})"
        lines.append(line)
    return "\n".join(lines)


class RawConversationMemoryBuilder(MemoryBuilder):
    """Chronological raw turns, grouped into session chunks.

    Optional max_chars keeps very long conversations within a rough budget
    when the full dialog would blow the answer-model context. Truncation is
    from the *start* (keep recent sessions) so late evidence is more likely
    retained — simple heuristic, not true retrieval.
    """

    name = "raw_chunks"

    def __init__(
        self,
        max_chars: int | None = None,
        preprocess_index_dir: str | Path | None = None,
        retrieve_top_k: int | None = None,
    ):
        self.max_chars = max_chars
        self.preprocess_index_dir = (
            Path(preprocess_index_dir) if preprocess_index_dir else None
        )
        self.retrieve_top_k = retrieve_top_k

    def build(self, conversation: Conversation, question: Question) -> Memory:
        if self.preprocess_index_dir is not None:
            from .preprocess.retrieve import build_raw_chunks_from_index

            text, source_ids = build_raw_chunks_from_index(
                self.preprocess_index_dir,
                conversation,
                question,
                max_chars=self.max_chars,
                top_k=self.retrieve_top_k,
            )
            return Memory(
                memory_type=self.name,
                text=text,
                source_ids=source_ids,
            )

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
                source_ids.append(turn.dia_id or f"session_{session.session_id}")
            block_lines.append(format_session_turns(session))
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
    """Concatenate LoCoMo session summaries chronologically.

    Dataset-provided summaries (no teacher API). Contrast with
    teacher_session_summaries.
    """

    name = "session_summaries"

    def __init__(
        self,
        preprocess_index_dir: str | Path | None = None,
        retrieve_top_k: int | None = None,
    ):
        self.preprocess_index_dir = (
            Path(preprocess_index_dir) if preprocess_index_dir else None
        )
        self.retrieve_top_k = retrieve_top_k

    def build(self, conversation: Conversation, question: Question) -> Memory:
        if self.preprocess_index_dir is not None:
            from .preprocess.retrieve import build_session_summaries_from_index

            text, source_ids = build_session_summaries_from_index(
                self.preprocess_index_dir,
                conversation,
                question,
                top_k=self.retrieve_top_k,
            )
            return Memory(
                memory_type=self.name,
                text=text,
                source_ids=source_ids,
            )

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


class TeacherSessionMemoryBuilder(MemoryBuilder):
    """Per-session LLM summaries, concatenated like session_summaries.

    Same inject path as SessionSummaryMemoryBuilder. The variable is the
    teacher model, not the answer prompt. Gold answers never enter the teacher.
    """

    name = "teacher_session_summaries"

    def __init__(self, teacher: Teacher):
        self.teacher = teacher

    @property
    def teacher_model(self) -> str:
        return self.teacher.model_name

    @property
    def teacher_provider(self) -> str:
        return self.teacher.provider

    def build(self, conversation: Conversation, question: Question) -> Memory:
        chunks: list[str] = []
        source_ids: list[str] = []
        for session in conversation.sessions:
            if not session.turns:
                continue
            session_text = format_session_turns(session)
            summary, _meta = self.teacher.summarize_session(
                session_text=session_text,
                date_time=session.date_time,
                speaker_a=conversation.speaker_a,
                speaker_b=conversation.speaker_b,
            )
            summary = (summary or "").strip()
            if not summary:
                continue
            chunks.append(f"[Session {session.session_id}]\n{summary}")
            source_ids.append(f"session_{session.session_id}_teacher")

        if not chunks:
            text = "(No teacher session memories available for this conversation.)"
        else:
            text = "\n\n".join(chunks)

        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
            teacher_model=self.teacher.model_name,
            teacher_provider=self.teacher.provider,
        )


# Registry name → constructor kwargs factory
_BUILDERS: dict[str, type[MemoryBuilder]] = {
    RawConversationMemoryBuilder.name: RawConversationMemoryBuilder,
    SessionSummaryMemoryBuilder.name: SessionSummaryMemoryBuilder,
    TeacherSessionMemoryBuilder.name: TeacherSessionMemoryBuilder,
}

# Legacy / short names. Canonical ids are the builder ``name`` values above.
_ALIASES: dict[str, str] = {
    "c0": RawConversationMemoryBuilder.name,
    "c0_raw": RawConversationMemoryBuilder.name,
    "raw": RawConversationMemoryBuilder.name,
    "raw_dialog": RawConversationMemoryBuilder.name,
    "c1": SessionSummaryMemoryBuilder.name,
    "c1_session_summary": SessionSummaryMemoryBuilder.name,
    "session_summary": SessionSummaryMemoryBuilder.name,
    "c1_teacher": TeacherSessionMemoryBuilder.name,
    "teacher": TeacherSessionMemoryBuilder.name,
    "mem0": "mem0",
    "mem0g": "mem0g",
}


def resolve_memory_name(name: str) -> str:
    key = name.strip().lower() if name else ""
    if key in _BUILDERS or key in ("mem0", "mem0g"):
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"Unknown memory builder '{name}'. "
        f"Known: {sorted(list(_BUILDERS) + ['mem0', 'mem0g'])} aliases={sorted(_ALIASES)}"
    )


def get_memory_builder(
    name: str,
    max_chars: int | None = None,
    teacher: Teacher | None = None,
    mem0_index_dir: str | Path | None = None,
    mem0_top_k: int = 30,
    mem0_embedder: Embedder | None = None,
    preprocess_index_dir: str | Path | None = None,
    retrieve_top_k: int | None = None,
) -> MemoryBuilder:
    resolved = resolve_memory_name(name)
    if resolved in ("mem0", "mem0g"):
        from .mem0.builders import Mem0IndexMemoryBuilder, Mem0gIndexMemoryBuilder

        if mem0_index_dir is None:
            raise ValueError(
                f"{resolved} requires a write-index directory. "
                "Set mem0.index_run_id in YAML (experiments/<id>/mem0_index) "
                "after python -m src.locomo_eval.mem0.run_index."
            )
        cls = Mem0IndexMemoryBuilder if resolved == "mem0" else Mem0gIndexMemoryBuilder
        return cls(
            index_dir=mem0_index_dir,
            embedder=mem0_embedder,
            top_k=mem0_top_k,
        )
    cls = _BUILDERS[resolved]
    if resolved == RawConversationMemoryBuilder.name:
        return cls(
            max_chars=max_chars,
            preprocess_index_dir=preprocess_index_dir,
            retrieve_top_k=retrieve_top_k,
        )
    if resolved == SessionSummaryMemoryBuilder.name:
        return cls(
            preprocess_index_dir=preprocess_index_dir,
            retrieve_top_k=retrieve_top_k,
        )
    if resolved == TeacherSessionMemoryBuilder.name:
        if teacher is None:
            raise ValueError(
                "teacher_session_summaries requires a Teacher "
                "(set teacher.model / --teacher-model)."
            )
        return cls(teacher=teacher)
    return cls()


def is_question_independent(name: str, *, retrieve_top_k: int | None = None) -> bool:
    """True if memory text does not depend on the question (safe to cache per sample)."""
    if retrieve_top_k is not None:
        return False
    resolved = resolve_memory_name(name)
    return resolved in (
        RawConversationMemoryBuilder.name,
        SessionSummaryMemoryBuilder.name,
        TeacherSessionMemoryBuilder.name,
    )
