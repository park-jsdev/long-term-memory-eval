"""Typed records that move through one LoCoMo QA run.

Load (dataset.py) → Memory (memory.py) → Reader (readers.py) → Prediction
→ score (metrics.py / offline_evaluate.py) and optionally → Autorater
(autorater.py / scripts.analysis.run_benchmark). Orchestrated by
run_locomo_pipeline_with_memory_config in run.py (one memory YAML per call).

    locomo10.json
         │
         ▼
    Conversation ── contains ── Session ── contains ── Turn
         │
         ├── Question[]          gold QA items (answer is for scoring only)
         └── session_summaries   session_summaries condition uses these; raw_chunks uses sessions/turns
         │
         ▼
    Memory.text                  sandwich middle — the one thing we vary
         │
         ▼
    Reader.answer()              frozen bottom — must not see the gold answer
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
class Memory:
    """Context string the frozen answer model is allowed to see.

    Built by MemoryBuilder (raw_chunks / session_summaries / teacher_session_summaries). Schema: docs/schemas/memory_runtime.md
    """

    memory_type: str
    text: str
    source_ids: list[str] = field(default_factory=list)
    schema_version: str = "memory_io.v1"
    teacher_model: str | None = None
    teacher_provider: str | None = None

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
    cached: bool = False  # LlmResponseHash hit if a store is passed; run.py currently does not
    teacher_model: str | None = None
    teacher_provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
