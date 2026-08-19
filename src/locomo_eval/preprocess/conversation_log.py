"""Dump preprocessed conversation session blocks for audit (not wired into run.py).

Analog of memory_log.py: that file records Memory.text; this file records
ProcessedConversation / SessionBlock so later teacher and eval steps can join
the same session and turn ids.

A later slice can write experiments/<run_id>/preprocess/.
Gold answers are not in this dump — only SessionBlock fields.
"""

from __future__ import annotations

from pathlib import Path

from ..report import write_json, write_jsonl
from ..schemas import ProcessedConversation

SCHEMA_VERSION = "preprocess_io.v1"
DOC_PATH = "docs/schemas/preprocess_runtime.md"
JSON_SCHEMA_PATH = "docs/schemas/preprocess_io.schema.json"


def write_conversation_run_log(
    out_dir: str | Path,
    processed: list[ProcessedConversation],
    *,
    doc_path: str = DOC_PATH,
) -> Path:
    """Write schema.json, index.jsonl, and per-sample sessions.jsonl."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    write_json(
        root / "schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "documentation": doc_path,
            "json_schema": JSON_SCHEMA_PATH,
            "n_samples": len(processed),
            "object_fields": {
                "session_blocks": "ordered non-empty LoCoMo sessions",
                "turns.turn_id": "{sample_id}:s{session_id}:t{iii}",
                "turns.source_dia_id": "LoCoMo dia_id",
                "date_time_raw": "unmodified session_N_date_time",
                "date_time_normalized": "ISO-8601 or null",
            },
            "gold_answer_in_session_blocks": False,
            "wired_from_run_py": False,
        },
    )

    index_rows: list[dict] = []
    for item in processed:
        sample_dir = root / "by_sample" / item.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        rel = f"by_sample/{item.sample_id}/sessions.jsonl"
        write_jsonl(sample_dir / "sessions.jsonl", [b.to_dict() for b in item.session_blocks])
        n_turns = sum(len(b.turns) for b in item.session_blocks)
        index_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": item.sample_id,
                "n_session_blocks": len(item.session_blocks),
                "n_turns": n_turns,
                "question_ids": list(item.question_ids),
                "sessions_path": rel,
            }
        )
    write_jsonl(root / "index.jsonl", index_rows)
    return root
