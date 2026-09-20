"""Per-question and run-level harness metrics.

String correctness still comes from ``src.locomo_eval.metrics.score_row``.
This module only adds retrieval / failure-mode fields on top of a scored
prediction row.
"""

from __future__ import annotations

from typing import Any

from .protocol import RetrievalTrajectory
from .trajectory import failure_mode


def score_agent_row(
    *,
    locomo_f1: float,
    exact_match: float,
    trajectory: RetrievalTrajectory,
) -> dict[str, Any]:
    """Attach retrieval stats + failure mode to one already-scored QA row."""
    correct = float(locomo_f1 or 0.0) > 0.0 or float(exact_match or 0.0) >= 1.0
    mode = failure_mode(
        correct=correct,
        evidence_retrieved=trajectory.evidence_retrieved,
        harness_failed=trajectory.harness_failed,
    )
    return {
        "n_retrieval_calls": trajectory.n_retrieval_calls,
        "retrieved_tokens": trajectory.retrieved_tokens,
        "evidence_retrieved": trajectory.evidence_retrieved,
        "memory_recall": trajectory.memory_recall,
        "memory_precision": trajectory.memory_precision,
        "unnecessary_retrievals": trajectory.unnecessary_retrievals,
        "failure_mode": mode,
        "harness_failed": trajectory.harness_failed,
        "harness_failure_reason": trajectory.harness_failure_reason,
        "correct": correct,
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
            "failure_modes": {},
        }
    n = len(rows)
    recalls = [r["memory_recall"] for r in rows if r.get("memory_recall") is not None]
    precs = [
        r["memory_precision"] for r in rows if r.get("memory_precision") is not None
    ]
    hits = [r.get("evidence_retrieved") for r in rows if r.get("evidence_retrieved") is not None]
    modes: dict[str, int] = {}
    for row in rows:
        key = str(row.get("failure_mode") or "unknown")
        modes[key] = modes.get(key, 0) + 1
    return {
        "n_questions": n,
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
        "failure_modes": modes,
    }


def _mean(values: list[Any]) -> float | None:
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
