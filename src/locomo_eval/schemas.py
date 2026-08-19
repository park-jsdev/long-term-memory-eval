"""Typed records that move through one LoCoMo QA run.

Read path (wired in run.py):
    locomo10.json → dataset.py Conversation → Memory → Reader → Prediction → score

Write-path preprocess (HLD i; not wired into run.py yet):
    locomo10.json → DataIngestor → PreprocessingPipeline
        → ProcessedConversation / SessionBlock → TeacherOrchestrator (passthrough)

    locomo10.json
         │
         ▼
    Conversation ── contains ── Session ── contains ── Turn
         │
         ├── Question[]          gold QA items (answer is for scoring only)
         └── session_summaries   C1 uses these; C0 uses sessions/turns instead
         │
         ├── preprocess/         SessionBlock[] (stable ids, normalized speakers/times)
         │
         ▼
    Memory.text                  sandwich middle — the one thing we vary
         │
         ▼
    Reader.answer()              frozen bottom — must not see the gold answer
         │
         ▼
    Prediction                   one JSONL row: Q + gold + pred + memory snapshot
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Turn:
    """One dialog utterance. C0 concatenates these into Memory.text."""

    dia_id: str
    speaker: str
    text: str
    blip_caption: str | None = None


@dataclass
class Session:
    """One dated chat session. Groups turns; C0 walks these in order."""

    session_id: int
    date_time: str
    turns: list[Turn] = field(default_factory=list)


@dataclass
class Question:
    """One LoCoMo QA item on a Conversation.

    ``question`` goes to the reader. ``answer`` is gold for metrics only —
    it must not be copied into Memory or the reader prompt.
    """

    sample_id: str
    question_id: str
    question: str
    answer: str
    category: int
    evidence: list[str] = field(default_factory=list)
    qa_index: int = 0


@dataclass
class Conversation:
    """One LoCoMo sample: dialog history + gold questions.

    Built by dataset.py from locomo10.json. Input to MemoryBuilder.
    """

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
class ProcessedTurn:
    """One utterance inside a SessionBlock.

    Built by PreprocessingPipeline. Consumed by TeacherOrchestrator and
    later eval attribution. ``source_dia_id`` is LoCoMo's ``dia_id``; ``turn_id``
    is ours so downstream can join even if ``dia_id`` is missing.
    """

    turn_id: str
    source_dia_id: str
    turn_index: int
    speaker_raw: str
    speaker_role: str
    text: str
    blip_caption: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionBlock:
    """One LoCoMo session after preprocess — the unit for later teacher batching.

    Segmentation follows dataset ``session_N`` keys (no new time-gap cuts).
    Gold answers never live here. Schema: docs/schemas/preprocess_runtime.md
    """

    sample_id: str
    session_id: int
    session_index: int
    source_key: str
    date_time_raw: str
    date_time_normalized: str | None
    speaker_a: str
    speaker_b: str
    turns: list[ProcessedTurn] = field(default_factory=list)
    schema_version: str = "preprocess_io.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedConversation:
    """Ingest + preprocess output for one sample.

    ``question_ids`` are pass-through handles for later join; gold answers stay
    on Conversation / Prediction (scorer-only). Not yet consumed by run.py.
    """

    sample_id: str
    speaker_a: str
    speaker_b: str
    session_blocks: list[SessionBlock]
    question_ids: list[str] = field(default_factory=list)
    schema_version: str = "preprocess_io.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Memory:
    """Context string the frozen answer model is allowed to see.

    Built by MemoryBuilder (C0/C1). Schema: docs/schemas/memory_runtime.md
    """

    memory_type: str
    text: str
    source_ids: list[str] = field(default_factory=list)
    schema_version: str = "memory_io.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    """One scored QA row written to experiments/<run_id>/predictions.jsonl.

    Snapshot of question, gold, model answer, and the memory that was used.
    offline_evaluate.py rescores these with string metrics only (no LLM).
    """

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
    run_id: str = ""
    cached: bool = False  # LlmResponseCache hit if a cache is passed; run.py currently does not

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
