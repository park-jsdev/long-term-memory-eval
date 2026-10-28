"""Merge the Plus starter sheet with AI drafts and human saves.

The starter file keeps the raw cue in `what`. This sheet copies that
into `cue_dialogue` and uses `what` for the constraint statement.
A revised or confirmed row is left alone when drafts are reapplied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ai_drafts import AI_DRAFTS

HUMAN_STATUSES = ("revised", "confirmed")
EDITABLE = (
    "who_basis",
    "what_type",
    "what",
    "scope",
    "exceptions",
    "uncertainty",
    "if_decision",
    "when",
    "how_then",
    "annotator_notes",
)


def _copy_cue(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the dialogue text even after `what` becomes an interpretation."""
    out = dict(row)
    if not str(out.get("cue_dialogue") or "").strip():
        out["cue_dialogue"] = str(out.get("what") or "")
    out.setdefault("exceptions", [])
    out.setdefault("annotator_notes", "")
    out.setdefault("label_status", "unlabeled")
    out.setdefault("gold", False)
    return out


def apply_draft(row: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    """Fill empty label fields. Store the draft under ai_* so edits can differ."""
    out = _copy_cue(row)
    if out.get("label_status") in HUMAN_STATUSES:
        return out
    for field in EDITABLE:
        if field not in draft and field != "annotator_notes":
            continue
        value = draft.get(field, out.get(field, ""))
        out[field] = value
        out[f"ai_{field}"] = value
    out["ai_confidence"] = draft.get("confidence", "")
    out["label_status"] = "ai_draft"
    out["label_source"] = "ai_draft_v0"
    out["gold"] = False
    if not str(out.get("what") or "").strip():
        out["what"] = ""
    return out


def merge_sheet(
    starter_rows: list[dict[str, Any]],
    saved_rows: list[dict[str, Any]] | None = None,
    drafts: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Starter order, drafts on unlabeled rows, human rows win field by field."""
    draft_map = AI_DRAFTS if drafts is None else drafts
    saved = {str(row.get("record_id")): row for row in (saved_rows or [])}
    merged: list[dict[str, Any]] = []
    for starter in starter_rows:
        row = _copy_cue(starter)
        if row.get("label_status") == "unlabeled" and row["what"] == row["cue_dialogue"]:
            row["what"] = ""
        draft = draft_map.get(str(row.get("record_id")))
        if draft:
            row = apply_draft(row, draft)
        prior = saved.get(str(row.get("record_id")))
        if prior and prior.get("label_status") in HUMAN_STATUSES:
            for field in EDITABLE:
                if field in prior:
                    row[field] = prior[field]
            row["label_status"] = prior["label_status"]
            row["gold"] = bool(prior.get("gold"))
            row["label_source"] = prior.get("label_source") or "human"
        merged.append(row)
    return merged


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def load_or_build_sheet(starter_path: Path, sheet_path: Path) -> list[dict[str, Any]]:
    """Build the working sheet and keep any rows already marked by a person."""
    starter = load_jsonl(starter_path)
    if not starter:
        raise FileNotFoundError(f"No starter rows in {starter_path}")
    saved = load_jsonl(sheet_path)
    rows = merge_sheet(starter, saved)
    write_jsonl(sheet_path, rows)
    return rows


def update_row(rows: list[dict[str, Any]], record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Replace editable fields. gold=true confirms the row. gold=false keeps a revision."""
    for row in rows:
        if str(row.get("record_id")) != record_id:
            continue
        for field in EDITABLE:
            if field in fields:
                row[field] = fields[field]
        if isinstance(row.get("exceptions"), str):
            row["exceptions"] = [
                part.strip() for part in row["exceptions"].split(",") if part.strip()
            ]
        gold = bool(fields.get("gold"))
        row["gold"] = gold
        row["label_status"] = "confirmed" if gold else "revised"
        row["label_source"] = "human"
        return row
    raise KeyError(record_id)
