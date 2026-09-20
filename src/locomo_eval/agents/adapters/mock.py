"""Offline harness stand-in. No Codex binary, no API.

Retrieves INDEX.md plus every session file whose text contains a gold
evidence id (or the first session when the item has no evidence). Returns
``Unknown.`` so mock smoke stays plumbing-only — like ``MockReader``.
"""

from __future__ import annotations

from pathlib import Path

from ..protocol import (
    EVENT_CATALOG,
    EVENT_RETRIEVE,
    AgentRequest,
    AgentResult,
    AgentRunner,
    RetrievalEvent,
)
from ..trajectory import build_trajectory, count_retrieved_tokens, evidence_ids_in_text
from ..workspace import INDEX_NAME, SESSIONS_DIR


class MockAgentRunner(AgentRunner):
    """Deterministic file walk. Used by unit tests and ``--agent mock``."""

    adapter_id = "mock"

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def run(self, request: AgentRequest) -> AgentResult:
        events: list[RetrievalEvent] = []
        root = Path(request.workspace_dir)
        index_path = root / INDEX_NAME
        step = 1
        if index_path.is_file():
            text = index_path.read_text(encoding="utf-8")
            events.append(
                RetrievalEvent(
                    step=step,
                    kind=EVENT_CATALOG,
                    tool="read",
                    target=INDEX_NAME,
                    retrieved_text=text,
                    retrieved_tokens=count_retrieved_tokens(text),
                )
            )
            step += 1
        sessions = sorted((root / SESSIONS_DIR).glob("session_*.md")) if (root / SESSIONS_DIR).is_dir() else []
        chosen: list[Path] = []
        required = [str(e).strip() for e in request.evidence_ids if str(e).strip()]
        if required:
            for path in sessions:
                body = path.read_text(encoding="utf-8")
                if evidence_ids_in_text(body, required):
                    chosen.append(path)
        elif sessions:
            chosen = [sessions[0]]
        for path in chosen:
            max_calls = request.tool_budget.get("max_tool_calls")
            if max_calls is not None and len(events) >= int(max_calls):
                break
            body = path.read_text(encoding="utf-8")
            max_tokens = request.tool_budget.get("max_retrieved_tokens")
            if max_tokens is not None:
                used = sum(event.retrieved_tokens for event in events)
                allowed = max(0, int(max_tokens) - used)
                words = body.split()
                body = " ".join(words[:allowed])
            rel = path.relative_to(root).as_posix()
            events.append(
                RetrievalEvent(
                    step=step,
                    kind=EVENT_RETRIEVE,
                    tool="read",
                    target=rel,
                    retrieved_text=body,
                    retrieved_tokens=count_retrieved_tokens(body),
                )
            )
            step += 1
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
        trajectory = build_trajectory(
            events, evidence_ids=request.evidence_ids, usage=usage
        )
        max_calls = request.tool_budget.get("max_tool_calls")
        max_tokens = request.tool_budget.get("max_retrieved_tokens")
        if (
            (max_calls is not None and len(events) > int(max_calls))
            or (max_tokens is not None and trajectory.retrieved_tokens > int(max_tokens))
        ):
            trajectory.harness_failed = True
            trajectory.harness_failure_reason = "tool_budget_exceeded"
        return AgentResult(
            predicted_answer="Unknown.",
            trajectory=trajectory,
            latency_s=0.0,
            usage=usage,
            events_raw=[e.to_dict() for e in events],
            call_meta={
                "adapter": self.adapter_id,
                "model": self.model_name,
                "thinking": False,
                "thinking_supported": False,
            },
        )
