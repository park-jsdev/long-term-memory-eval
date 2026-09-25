"""Harness protocol for agent-level LoCoMo eval (not the one-shot reader).

Model-only QA still goes through ``readers.Reader``. This module is the
answer path when ``pipeline.answer_mode: agent``: a coding-agent harness
(Codex first; Claude Code / OpenCode / Pi later) retrieves from a file
workspace and returns a predicted answer plus a retrieval trajectory.

Who consumes it: ``run_locomo_pipeline_with_memory_config`` and adapters
under ``src/locomo_eval/agents/adapters/``. Gold answers never live here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AGENT_TRAJECTORY_SCHEMA = "agent_trajectory.v1"
FAILURE_NONE = "none"
FAILURE_RETRIEVAL = "retrieval_failure"
FAILURE_REASONING = "reasoning_failure"
FAILURE_PARAMETRIC = "parametric_success"
FAILURE_HARNESS = "harness_execution_failure"

# Navigation reads (INDEX.md) are not scored as evidence retrieval.
EVENT_CATALOG = "catalog"
EVENT_RETRIEVE = "retrieve"
EVENT_WRITE = "write"
EVENT_OTHER = "other"
EVENT_ERROR = "error"
# Hosted search / MCP are outside the workspace. Not evidence; audited.
EVENT_WEB_SEARCH = "web_search"
EVENT_MCP = "mcp"

KNOWN_ADAPTERS = ("mock", "codex", "claude_code", "opencode", "pi")
STUB_ADAPTERS = ("claude_code", "opencode", "pi")


@dataclass
class RetrievalEvent:
    """One tool call that touched the workspace or an external tool."""

    step: int
    kind: str
    tool: str
    target: str
    retrieved_text: str = ""
    retrieved_tokens: int = 0
    evidence_ids_hit: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSONL-safe row for ``agent/trajectory.jsonl`` event lists."""
        return asdict(self)


@dataclass
class RetrievalTrajectory:
    """Normalized retrieval trace for one question.

    Distinguishes *never saw the gold evidence* from *saw it and still
    answered wrong*. Schema: ``docs/schemas/agent_runtime.md``.
    """

    schema_version: str = AGENT_TRAJECTORY_SCHEMA
    n_retrieval_calls: int = 0
    retrieved_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    evidence_retrieved: bool | None = None
    evidence_ids_required: list[str] = field(default_factory=list)
    evidence_ids_hit: list[str] = field(default_factory=list)
    memory_recall: float | None = None
    memory_precision: float | None = None
    unnecessary_retrievals: int = 0
    n_web_search: int = 0
    n_mcp: int = 0
    n_write_events: int = 0
    hop_to_evidence: int | None = None
    notes_retrieved: bool = False
    used_non_workspace_tools: bool = False
    harness_failed: bool = False
    harness_failure_reason: str | None = None
    events: list[RetrievalEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flatten nested events so json.dump does not need a default=."""
        row = asdict(self)
        row["events"] = [e.to_dict() if hasattr(e, "to_dict") else e for e in self.events]
        return row


@dataclass
class AgentRequest:
    """One LoCoMo question handed to a harness. Gold is not included."""

    workspace_dir: Path
    sample_id: str
    question_id: str
    question: str
    prompt: str
    evidence_ids: list[str] = field(default_factory=list)
    tool_budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Harness output for one question: answer + trajectory + billed usage."""

    predicted_answer: str
    trajectory: RetrievalTrajectory
    latency_s: float
    usage: dict[str, Any] = field(default_factory=dict)
    events_raw: list[dict[str, Any]] = field(default_factory=list)
    call_meta: dict[str, Any] = field(default_factory=dict)
    stderr: str = ""

    def to_trace_row(
        self,
        *,
        sample_id: str,
        question_id: str,
        adapter: str,
        model: str,
        prompt_version: str,
    ) -> dict[str, Any]:
        """One ``agent/traces.jsonl`` row: answer + usage + retrieval flags."""
        return {
            "sample_id": sample_id,
            "question_id": question_id,
            "role": "agent",
            "provider": adapter,
            "model": model,
            "predicted_answer": self.predicted_answer,
            "reasoning": self.call_meta.get("reasoning") or "",
            "reasoning_tokens": (self.usage or {}).get("reasoning_tokens"),
            "latency_s": self.latency_s,
            "usage": self.usage or {},
            "prompt_version": prompt_version,
            "n_retrieval_calls": self.trajectory.n_retrieval_calls,
            "retrieved_tokens": self.trajectory.retrieved_tokens,
            "evidence_retrieved": self.trajectory.evidence_retrieved,
            "n_web_search": self.trajectory.n_web_search,
            "n_mcp": self.trajectory.n_mcp,
            "n_write_events": self.trajectory.n_write_events,
            "hop_to_evidence": self.trajectory.hop_to_evidence,
            "notes_retrieved": self.trajectory.notes_retrieved,
            "used_non_workspace_tools": self.trajectory.used_non_workspace_tools,
            "web_search": self.call_meta.get("web_search"),
            "argv": self.call_meta.get("argv"),
            "returncode": self.call_meta.get("returncode"),
        }


class AgentRunner(ABC):
    """Sandwich bottom for harness eval: workspace + question → answer.

    One fresh invocation per question. Must not see gold.
    """

    adapter_id: str
    model_name: str

    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResult:
        """Return the predicted answer and the retrieval trajectory."""
        ...
