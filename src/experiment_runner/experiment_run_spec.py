"""One immutable scientific configuration the harness can execute.

Who consumes it: matrix expansion, hashed run ids, QA/autorater executors.
Gold answers never live here; they stay in LoCoMo JSON for the scorer only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReaderModelRef:
    """Answer-model identity for one cell. Snapshot fields are for provenance."""

    display_name: str
    provider: str
    family: str
    generation: str | int | None
    api_model_id: str
    model_snapshot: str | None = None
    catalog_id: str | None = None
    status: str = "runnable"
    thinking: bool | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class WriterModelRef:
    """Memory-write model when the cell does not use a shared frozen index."""

    display_name: str
    provider: str
    api_model_id: str
    catalog_id: str | None = None
    status: str = "runnable"
    thinking: bool | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class ExperimentRunSpec:
    """All parameters needed to run one QA job (and later its autorater).

    ``run_id`` is a deterministic hash of the scientific fields, not a timestamp.
    """

    experiment_name: str
    experiment_type: str
    run_index: int
    run_id: str
    benchmark: str
    memory_method: str
    reader: ReaderModelRef
    seed: int
    prompt_path: str
    judge_provider: str
    judge_model: str
    writer: WriterModelRef | None = None
    max_questions: int | None = None
    max_samples: int | None = None
    sample_id: str | None = None
    method_yaml: str = ""
    mem0_index_run_id: str | None = None
    rag_index_run_id: str | None = None
    status: str = "runnable"
    agent: str | None = None
    agent_persist: bool | None = None
    agent_sessions: str | None = None
    agent_tools: str | None = None
    agent_prompt_mode: str | None = None
    agent_comparison: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_manifest_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["reader_provider"] = self.reader.provider
        row["reader_model"] = self.reader.api_model_id
        row["reader_display_name"] = self.reader.display_name
        row["reader_family"] = self.reader.family
        row["reader_generation"] = self.reader.generation
        row["writer_model"] = self.writer.api_model_id if self.writer else None
        row["thinking"] = self.thinking_label()
        row["reader_thinking"] = self.reader.thinking
        row["reader_max_tokens"] = self.reader.max_tokens
        row["teacher_thinking"] = self.writer.thinking if self.writer else None
        row["teacher_max_tokens"] = self.writer.max_tokens if self.writer else None
        row["agent"] = self.agent
        row["agent_persist"] = self.agent_persist
        row["agent_sessions"] = self.agent_sessions
        row["agent_tools"] = self.agent_tools
        row["agent_prompt_mode"] = self.agent_prompt_mode
        row["comparison_contract"] = self.agent_comparison
        return row

    def qa_identity(self) -> dict[str, Any]:
        """Fields that define the hashed run id (QA, not judge, not timestamps)."""
        identity = {
            "benchmark": self.benchmark,
            "experiment_name": self.experiment_name,
            "max_questions": self.max_questions,
            "max_samples": self.max_samples,
            "mem0_index_run_id": self.mem0_index_run_id,
            "memory_method": self.memory_method,
            "prompt_path": self.prompt_path,
            "rag_index_run_id": self.rag_index_run_id,
            "reader_catalog_id": self.reader.catalog_id,
            "reader_model": self.reader.api_model_id,
            "reader_provider": self.reader.provider,
            "sample_id": self.sample_id,
            "seed": self.seed,
            "writer_model": self.writer.api_model_id if self.writer else None,
        }
        if self.agent:
            identity["agent"] = self.agent
            identity["agent_persist"] = self.agent_persist
            if self.agent_sessions and self.agent_sessions != "full":
                identity["agent_sessions"] = self.agent_sessions
            identity["agent_tools"] = self.agent_tools
            identity["agent_prompt_mode"] = self.agent_prompt_mode
            identity["agent_comparison"] = self.agent_comparison
        if self.reader.thinking is not None:
            identity["reader_thinking"] = self.reader.thinking
        if self.reader.max_tokens is not None:
            identity["reader_max_tokens"] = self.reader.max_tokens
        if self.writer is not None and self.writer.thinking is not None:
            identity["teacher_thinking"] = self.writer.thinking
        if self.writer is not None and self.writer.max_tokens is not None:
            identity["teacher_max_tokens"] = self.writer.max_tokens
        return identity

    def thinking_label(self) -> str | None:
        """Campaign axis value: ``on`` / ``off``, or None when the YAML omitted thinking."""
        flag = None
        if self.writer is not None and self.writer.thinking is not None:
            flag = self.writer.thinking
        elif self.reader.thinking is not None:
            flag = self.reader.thinking
        if flag is None:
            return None
        return "on" if flag else "off"
