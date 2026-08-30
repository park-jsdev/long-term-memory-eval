"""Write experiments/<run_id>/preprocess/ deterministic index (no LLM).

Gold answers are not in this dump. Session blocks + session documents are the
write-side units later retrieve/format paths consume (raw_chunks, session
summaries, later top-k or LLM compress).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..report import write_json, write_jsonl
from ..schemas import ProcessedConversation, SessionBlock
from .conversation_log import DOC_PATH, JSON_SCHEMA_PATH, SCHEMA_VERSION
from .session_documents import SessionDocument

RUN_INDEX_HINT = (
    "Run: python -m src.locomo_eval.preprocess.run_index "
    "--config configs/preprocess.yaml --run-id <index_run_id>"
)


class MissingPreprocessIndexError(FileNotFoundError):
    """Raised when a MemoryBuilder needs a dump that run_index has not written."""


def sample_index_dir(index_root: Path, sample_id: str) -> Path:
    return Path(index_root) / "by_sample" / sample_id


def dump_complete(sample_dir: Path) -> bool:
    return (sample_dir / "sessions.jsonl").is_file() and (
        sample_dir / "documents.jsonl"
    ).is_file()


def sample_complete(index_root: Path, sample_id: str) -> bool:
    return dump_complete(sample_index_dir(index_root, sample_id))


def write_sample_dump(
    index_root: Path,
    *,
    processed: ProcessedConversation,
    documents: list[SessionDocument],
) -> Path:
    sample_dir = sample_index_dir(index_root, processed.sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        sample_dir / "sessions.jsonl",
        [block.to_dict() for block in processed.session_blocks],
    )
    write_jsonl(sample_dir / "documents.jsonl", [doc.to_dict() for doc in documents])
    return sample_dir


def write_index_meta(
    index_root: Path,
    *,
    samples: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> Path:
    root = Path(index_root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "documentation": DOC_PATH,
            "json_schema": JSON_SCHEMA_PATH,
            "n_samples": len(samples),
            "object_fields": {
                "sessions.jsonl": "SessionBlock rows (gold-free)",
                "documents.jsonl": (
                    "SessionDocument rows: turns, dataset session_summary, "
                    "observations, events (no QA gold)"
                ),
            },
            "gold_answer_in_index": False,
            "llm_calls": 0,
        },
    )
    write_json(root / "run_meta.json", run_meta)
    write_jsonl(root / "index.jsonl", samples)
    return root


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_session_blocks(sample_dir: Path) -> list[SessionBlock]:
    path = sample_dir / "sessions.jsonl"
    return [SessionBlock.from_dict(row) for row in load_jsonl(path)]


def load_session_documents(sample_dir: Path) -> list[SessionDocument]:
    path = sample_dir / "documents.jsonl"
    return [SessionDocument.from_dict(row) for row in load_jsonl(path)]


def require_sample_dump(index_root: Path, sample_id: str) -> Path:
    sample_dir = sample_index_dir(index_root, sample_id)
    if not dump_complete(sample_dir):
        raise MissingPreprocessIndexError(
            f"Missing deterministic preprocess dump for sample {sample_id} "
            f"under {index_root}. {RUN_INDEX_HINT}"
        )
    return sample_dir
