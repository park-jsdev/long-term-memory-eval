"""Typed records that move through one LoCoMo QA run.

Load (dataset.py) → Memory (memory.py) → Reader (readers.py) → Prediction
→ score (metrics.py / offline_evaluate.py) and optionally → Autorater
(autorater.py / scripts.analysis.run_benchmark). Orchestrated by
run_locomo_pipeline_with_memory_config in run.py (one memory YAML per call).

Write-path preprocess (HLD i; not the QA scorer):
    locomo10.json → DataIngestor → PreprocessingPipeline
        → experiments/<run_id>/preprocess/ (SessionBlock + SessionDocument, no LLM)
        → raw_chunks / session_summaries retrieve/format (or Conversation fallback)

    locomo10.json
         │
         ▼
    Conversation ── contains ── Session ── contains ── Turn
         │
         ├── Question[]          gold QA items (answer is for scoring only)
         └── session_summaries   session_summaries condition uses these; raw_chunks uses sessions/turns
         │
         ├── preprocess/         SessionBlock[] (stable ids, normalized speakers/times)
         │
         ▼
    Memory.text                  memory representation — the experimental variable
         │
         ▼
    Reader.answer()              fixed evaluation — must not see the gold answer
         │
         ▼
    Prediction                   one JSONL row: Q + gold + pred + memory snapshot
         ├── deterministic scorer (EM / token F1 / LoCoMo F1)
         └── optional LLM autorater (Mem0 F1 / BLEU-1 / J)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Turn:
    """One dialog utterance. raw_chunks concatenates these into Memory.text."""

    dia_id: str
    speaker: str
    text: str
    blip_caption: str | None = None


@dataclass
class Session:
    """One dated chat session. Groups turns; raw_chunks walks these in order."""

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

    Built by PreprocessingPipeline. Consumed by ModelOrchestrator and
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

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ProcessedTurn:
        return cls(
            turn_id=str(row.get("turn_id") or ""),
            source_dia_id=str(row.get("source_dia_id") or ""),
            turn_index=int(row.get("turn_index") or 0),
            speaker_raw=str(row.get("speaker_raw") or ""),
            speaker_role=str(row.get("speaker_role") or ""),
            text=str(row.get("text") or ""),
            blip_caption=row.get("blip_caption"),
        )


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

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> SessionBlock:
        turns = [ProcessedTurn.from_dict(t) for t in (row.get("turns") or [])]
        return cls(
            sample_id=str(row["sample_id"]),
            session_id=int(row["session_id"]),
            session_index=int(row["session_index"]),
            source_key=str(row.get("source_key") or ""),
            date_time_raw=str(row.get("date_time_raw") or ""),
            date_time_normalized=row.get("date_time_normalized"),
            speaker_a=str(row.get("speaker_a") or ""),
            speaker_b=str(row.get("speaker_b") or ""),
            turns=turns,
            schema_version=str(row.get("schema_version") or "preprocess_io.v1"),
        )


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

    Built by MemoryBuilder (raw_chunks / session_summaries / graph /
    full_context / rag / openai_memory / mem0 / mem0g / workspace_files).
    Schema: docs/schemas/memory_runtime.md
    """

    memory_type: str
    text: str
    source_ids: list[str] = field(default_factory=list)
    schema_version: str = "memory_io.v1"
    writer_model: str | None = None
    writer_provider: str | None = None
    search_latency_s: float | None = None

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
    writer_model: str | None = None
    writer_provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
