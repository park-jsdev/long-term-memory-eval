"""Reusable overlay of harness traces onto campaign example rows (no LLM).

Finished QA parquet often omits hop-to-evidence / notes size because those
fields were added after a pack ran. ``overlay_collected_agent_audit`` fills
them from ``collected/runs/<id>/agent/trajectory.jsonl`` and
``notes_ledger.jsonl`` when present. Campaign YAML then groups on
``hop_bin`` / ``qidx_bin`` / ``notes_bytes_bin``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.locomo_eval.agents.trajectory import notes_path_in
from src.locomo_eval.agents.workspace import PERSIST_NOTES


def overlay_collected_agent_audit(
    examples: pd.DataFrame, pack: Path
) -> pd.DataFrame:
    """Copy hop / write / notes fields from collected agent dumps onto examples."""
    if examples.empty or "run_id" not in examples.columns:
        return examples
    if "question_id" not in examples.columns:
        return examples
    overlay: dict[tuple[str, str], dict[str, Any]] = {}
    run_ids = [str(v) for v in examples["run_id"].dropna().unique()]
    for run_id in run_ids:
        agent_dir = _first_agent_dir(pack, run_id)
        if agent_dir is None:
            continue
        overlay.update(_overlay_from_agent_dir(run_id, agent_dir))
    if not overlay:
        return examples
    out = examples.copy()
    keys = [
        (str(rid), str(qid))
        for rid, qid in zip(out["run_id"].tolist(), out["question_id"].tolist())
    ]
    fields = (
        "hop_to_evidence",
        "n_write_events",
        "notes_retrieved",
        "notes_bytes",
        "notes_words",
        "notes_grew",
    )
    for field in fields:
        incoming = [overlay.get(key, {}).get(field) for key in keys]
        if field not in out.columns:
            out[field] = incoming
            continue
        current = out[field]
        filled = [
            cur if _present(cur) else new
            for cur, new in zip(current.tolist(), incoming)
        ]
        out[field] = filled
    return out


def hop_from_trajectory_row(row: dict[str, Any]) -> int | None:
    """First retrieve step whose text (or hits) contains a gold ``dia_id``."""
    stored = row.get("hop_to_evidence")
    if stored not in (None, "", 0):
        try:
            return int(stored)
        except (TypeError, ValueError):
            pass
    required = [str(e).strip() for e in (row.get("evidence_ids_required") or []) if str(e).strip()]
    for event in row.get("events") or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("kind") or "") != "retrieve":
            continue
        hits = event.get("evidence_ids_hit") or []
        text = str(event.get("retrieved_text") or "")
        if hits or any(eid and eid in text for eid in required):
            try:
                step = int(event.get("step") or 0)
            except (TypeError, ValueError):
                return None
            return step or None
    return None


def _overlay_from_agent_dir(
    run_id: str, agent_dir: Path
) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    traj_path = agent_dir / "trajectory.jsonl"
    if traj_path.is_file():
        for row in _read_jsonl(traj_path):
            qid = str(row.get("question_id") or "")
            if not qid:
                continue
            events = row.get("events") or []
            n_write = row.get("n_write_events")
            if n_write in (None, ""):
                n_write = sum(
                    1
                    for event in events
                    if isinstance(event, dict) and str(event.get("kind") or "") == "write"
                )
            notes_hit = row.get("notes_retrieved")
            if notes_hit is None:
                notes_hit = any(
                    isinstance(event, dict)
                    and str(event.get("kind") or "") == "retrieve"
                    and notes_path_in(str(event.get("target") or ""))
                    for event in events
                )
            rec = {
                "hop_to_evidence": hop_from_trajectory_row(row),
                "n_write_events": n_write,
                "notes_retrieved": notes_hit,
            }
            out[(run_id, qid)] = rec
    ledger_path = agent_dir / "notes_ledger.jsonl"
    if ledger_path.is_file():
        for row in _read_jsonl(ledger_path):
            qid = str(row.get("question_id") or "")
            if not qid or qid.startswith("_"):
                continue
            rec = dict(out.get((run_id, qid)) or {})
            rec["notes_bytes"] = row.get("notes_bytes")
            rec["notes_words"] = row.get("notes_words")
            rec["notes_grew"] = row.get("notes_grew")
            out[(run_id, qid)] = rec
    if not any("notes_bytes" in rec for rec in out.values()):
        notes = _final_notes_bytes(agent_dir)
        if notes is not None:
            for key, rec in list(out.items()):
                filled = dict(rec)
                filled.setdefault("notes_bytes", notes)
                out[key] = filled
    return out


def _final_notes_bytes(agent_dir: Path) -> int | None:
    workspaces = agent_dir / "workspaces"
    if not workspaces.is_dir():
        return None
    sizes = []
    for notes in workspaces.glob(f"*/{PERSIST_NOTES}"):
        try:
            sizes.append(notes.stat().st_size)
        except OSError:
            continue
    if not sizes:
        return None
    return int(sum(sizes) / len(sizes))


def _first_agent_dir(pack: Path, run_id: str) -> Path | None:
    candidates = (
        pack / "collected" / "runs" / run_id / "agent",
        pack / "runs" / run_id / "agent",
        pack / "aggregate" / "by_run" / run_id / "agent",
    )
    for path in candidates:
        if (path / "trajectory.jsonl").is_file() or (path / "notes_ledger.jsonl").is_file():
            return path
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True
