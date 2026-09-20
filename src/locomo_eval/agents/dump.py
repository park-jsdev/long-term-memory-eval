"""Write ``experiments/<run_id>/agent/`` during a harness QA run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.locomo_eval.experiments.audit_layout import AuditPaths
from src.locomo_eval.report import write_json, write_jsonl

from .metrics import summarize_agent_rows


def write_agent_module(
    run_dir: Path,
    *,
    traces: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    events: list[dict[str, Any]],
    metrics: dict[str, Any],
    summary_rows: list[dict[str, Any]] | None = None,
    comparison_contract: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Dump harness traces. Empty lists still write the folder for agent runs."""
    paths = AuditPaths.from_run_dir(run_dir)
    paths.agent_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.agent_traces, traces)
    write_jsonl(paths.agent_trajectory, trajectories)
    write_jsonl(paths.agent_events, events)
    payload = dict(metrics)
    if summary_rows is not None:
        payload.update(summarize_agent_rows(summary_rows))
    if comparison_contract is not None:
        payload["comparison_status"] = comparison_contract.get("status")
        payload["comparison_contract_sha256"] = comparison_contract.get("sha256")
    write_json(paths.agent_metrics, payload)
    comparison_path = paths.agent_dir / "COMPARISON.md"
    if comparison_contract is not None:
        controls = comparison_contract
        lines = [
            "# Agent comparison controls",
            "",
            f"- schema: `{controls.get('schema_version')}`",
            f"- status: **{controls.get('status')}**",
            f"- contract sha256: `{controls.get('sha256')}`",
            f"- backbone: `{(controls.get('backbone') or {}).get('adapter')}` / "
            f"`{(controls.get('backbone') or {}).get('model')}`",
            f"- task prompt sha256: `{(controls.get('task_prompt') or {}).get('sha256')}`",
            f"- workspace manifest sha256: `{(controls.get('context') or {}).get('workspace_manifest_sha256')}`",
            f"- retrieval: `{controls.get('retrieval')}`",
            f"- memory write: `{controls.get('memory_write')}`",
            f"- judge: `{controls.get('judge')}`",
            f"- tool budget: `{controls.get('tool_budget')}`",
        ]
        comparison_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "agent_dir": str(paths.agent_dir),
        "agent_traces": str(paths.agent_traces),
        "agent_trajectory": str(paths.agent_trajectory),
        "agent_events": str(paths.agent_events),
        "agent_metrics": str(paths.agent_metrics),
        "comparison": str(comparison_path) if comparison_contract is not None else "",
    }
