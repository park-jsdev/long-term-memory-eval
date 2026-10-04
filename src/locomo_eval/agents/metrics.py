"""Per-question and run-level harness metrics.

String correctness still comes from ``src.locomo_eval.metrics.score_row``.
This module only adds retrieval / failure-mode fields on top of a scored
prediction row.
"""

from __future__ import annotations

from typing import Any

from .protocol import FAILURE_PROMPT_INJECTED, RetrievalTrajectory
from .trajectory import failure_mode


def score_agent_row(
    *,
    locomo_f1: float,
    exact_match: float,
    trajectory: RetrievalTrajectory,
    evidence_delivery: str = "workspace_retrieval",
) -> dict[str, Any]:
    """Attach retrieval stats + failure mode to one already-scored QA row."""
    correct = float(locomo_f1 or 0.0) > 0.0 or float(exact_match or 0.0) >= 1.0
    prompt_injected = evidence_delivery == "prompt_injected"
    mode = (
        FAILURE_PROMPT_INJECTED
        if prompt_injected
        else failure_mode(
            correct=correct,
            evidence_retrieved=trajectory.evidence_retrieved,
            harness_failed=trajectory.harness_failed,
        )
    )
    return {
        "n_retrieval_calls": trajectory.n_retrieval_calls,
        "retrieved_tokens": trajectory.retrieved_tokens,
        "evidence_retrieved": None if prompt_injected else trajectory.evidence_retrieved,
        "memory_recall": None if prompt_injected else trajectory.memory_recall,
        "memory_precision": None if prompt_injected else trajectory.memory_precision,
        "unnecessary_retrievals": None if prompt_injected else trajectory.unnecessary_retrievals,
        "n_web_search": trajectory.n_web_search,
        "n_mcp": trajectory.n_mcp,
        "n_write_events": trajectory.n_write_events,
        "hop_to_evidence": trajectory.hop_to_evidence,
        "notes_retrieved": trajectory.notes_retrieved,
        "used_non_workspace_tools": trajectory.used_non_workspace_tools,
        "failure_mode": mode,
        "harness_failed": trajectory.harness_failed,
        "harness_failure_reason": trajectory.harness_failure_reason,
        "correct": correct,
        "evidence_delivery": evidence_delivery,
    }


def summarize_agent_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Means + failure-mode counts. Empty list → None metrics, not zeros."""
    if not rows:
        return {
            "n_questions": 0,
            "n_retrieval_calls_mean": None,
            "retrieved_tokens_mean": None,
            "memory_recall_mean": None,
            "memory_precision_mean": None,
            "unnecessary_retrievals_mean": None,
            "evidence_retrieved_rate": None,
            "n_web_search_sum": None,
            "n_mcp_sum": None,
            "n_write_events_mean": None,
            "hop_to_evidence_mean": None,
            "notes_retrieved_rate": None,
            "used_non_workspace_tools_rate": None,
            "n_harness_failed": 0,
            "harness_failed_rate": None,
            "failure_modes": {},
        }
    n = len(rows)
    recalls = [r["memory_recall"] for r in rows if r.get("memory_recall") is not None]
    precs = [
        r["memory_precision"] for r in rows if r.get("memory_precision") is not None
    ]
    hits = [r.get("evidence_retrieved") for r in rows if r.get("evidence_retrieved") is not None]
    n_harness_failed = sum(1 for r in rows if r.get("harness_failed"))
    modes: dict[str, int] = {}
    for row in rows:
        key = str(row.get("failure_mode") or "unknown")
        modes[key] = modes.get(key, 0) + 1
    return {
        "n_questions": n,
        "n_harness_failed": n_harness_failed,
        "harness_failed_rate": n_harness_failed / n,
        "n_retrieval_calls_mean": _mean(
            [r.get("n_retrieval_calls") for r in rows]
        ),
        "retrieved_tokens_mean": _mean([r.get("retrieved_tokens") for r in rows]),
        "memory_recall_mean": _mean(recalls) if recalls else None,
        "memory_precision_mean": _mean(precs) if precs else None,
        "unnecessary_retrievals_mean": _mean(
            [r.get("unnecessary_retrievals") for r in rows]
        ),
        "evidence_retrieved_rate": (
            sum(1 for h in hits if h) / len(hits) if hits else None
        ),
        "n_web_search_sum": sum(int(r.get("n_web_search") or 0) for r in rows),
        "n_mcp_sum": sum(int(r.get("n_mcp") or 0) for r in rows),
        "n_write_events_mean": _mean([r.get("n_write_events") for r in rows]),
        "hop_to_evidence_mean": _mean([r.get("hop_to_evidence") for r in rows]),
        "notes_retrieved_rate": _mean(
            [bool(r.get("notes_retrieved")) for r in rows]
        ),
        "used_non_workspace_tools_rate": _mean(
            [bool(r.get("used_non_workspace_tools")) for r in rows]
        ),
        "failure_modes": modes,
    }


def _mean(values: list[Any]) -> float | None:
    """Arithmetic mean, skipping None / non-numeric. Empty → None, not 0."""
    nums = []
    for value in values:
        if value is None:
            continue
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    if not nums:
        return None
    return sum(nums) / len(nums)
