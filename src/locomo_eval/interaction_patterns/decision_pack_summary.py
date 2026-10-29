"""First-pass counts from decision files and hook logs.

These checks are mechanical. Observer usefulness and "the model used
the context appropriately" stay empty until a person or a calibrated
judge fills them. A B/C id mismatch fails the pack: that pair is the
Stage 2 comparison, and it is invalid if C reselected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def selection_identity(b_ids: list[str], c_ids: list[str]) -> bool:
    """True when B and C carry the same selected ids in the same order."""
    return list(b_ids) == list(c_ids)


def instruction_cites_unselected(
    instruction: str,
    selected_ids: list[str],
    known_ids: list[str],
) -> list[str]:
    """Ids named in Stage 2 text that Stage 1 did not select."""
    selected = set(selected_ids)
    return [
        record_id
        for record_id in known_ids
        if record_id not in selected and record_id in instruction
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hook_actions(directory: Path) -> list[str]:
    path = directory / "hook_log.jsonl"
    if not path.is_file():
        return []
    actions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "UserPromptSubmit":
            actions.append(str(row.get("action") or ""))
    return actions


def summarize_request_dir(request_dir: Path, known_ids: list[str]) -> dict[str, Any]:
    """Summarize one request's P/A/B/C folders."""
    rows = []
    for condition in ("P", "A", "B", "C"):
        folder = request_dir / condition
        decision = _read_json(folder / "memory" / "decision.json")
        actions = _hook_actions(folder)
        rows.append(
            {
                "condition": condition,
                "action": decision.get("action"),
                "selected_ids": list(decision.get("selected_ids") or []),
                "abstain": bool(decision.get("abstain")),
                "hook_actions": actions,
                "unselected_ids_in_instruction": instruction_cites_unselected(
                    str(decision.get("stage2_instruction") or ""),
                    list(decision.get("selected_ids") or []),
                    known_ids,
                ),
            }
        )
    by_condition = {row["condition"]: row for row in rows}
    b_ids = by_condition["B"]["selected_ids"]
    c_ids = by_condition["C"]["selected_ids"]
    hook_rows = [row for row in rows if row["condition"] != "P"]
    injections = sum(1 for row in hook_rows if row["action"] == "inject")
    abstentions = sum(1 for row in hook_rows if row["action"] == "abstain")
    delivered = [
        row["condition"]
        for row in hook_rows
        if row["hook_actions"]
    ]
    return {
        "request_id": request_dir.name,
        "conditions": rows,
        "bc_identity_ok": selection_identity(b_ids, c_ids),
        "injection_rate": injections / len(hook_rows) if hook_rows else 0.0,
        "abstention_rate": abstentions / len(hook_rows) if hook_rows else 0.0,
        "hook_invoked_conditions": delivered,
        "context_delivered_conditions": delivered,
        "context_used_appropriately": None,
    }


def summarize_pack(pack_dir: Path, known_ids: list[str]) -> dict[str, Any]:
    """Read requests/* and fail when any B/C pair diverges."""
    request_root = pack_dir / "requests"
    summaries = [
        summarize_request_dir(path, known_ids)
        for path in sorted(request_root.iterdir())
        if path.is_dir()
    ]
    mismatches = [row["request_id"] for row in summaries if not row["bc_identity_ok"]]
    report = {
        "requests": summaries,
        "bc_identity_ok": not mismatches,
        "bc_mismatches": mismatches,
        "context_used_appropriately": None,
        "stub": True,
    }
    if mismatches:
        raise ValueError(
            "B and C selected different ids on: " + ", ".join(mismatches)
        )
    return report
