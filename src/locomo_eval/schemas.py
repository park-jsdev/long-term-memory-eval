"""Small, readable data classes for the baseline pipeline.

Keep fields explicit so JSONL / CSV exports stay easy to audit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Turn:
    dia_id: str
    speaker: str
    text: str
    blip_caption: str | None = None


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: list[Turn] = field(default_factory=list)


@dataclass
class Question:
    sample_id: str
    question_id: str
    question: str
    answer: str
    category: int
    evidence: list[str] = field(default_factory=list)
    qa_index: int = 0


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    session_summaries: dict[int, str]
    observations: dict[int, Any]
    questions: list[Question]

    def chronological_summaries(self) -> list[tuple[int, str]]:
        return sorted(self.session_summaries.items(), key=lambda kv: kv[0])


@dataclass
class Memory:
    """Context string passed to the fixed answer model.

    Schema: docs/schemas/memory_runtime.md (memory_io.v1)
    """

    memory_type: str
    text: str
    source_ids: list[str] = field(default_factory=list)
    schema_version: str = "memory_io.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    sample_id: str
    question_id: str
    question: str
    reference_answer: str
    predicted_answer: str
    category: int
    memory_type: str
    memory_text: str
    reader_model: str
    prompt_version: str
    evidence: list[str] = field(default_factory=list)
    # Optional run metadata (filled by runner)
    run_id: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
