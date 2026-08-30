"""Write experiments/<run_id>/openai_memory_index/ audit pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..report import write_json, write_jsonl
from .extract import ExtractedMemory

SCHEMA_VERSION = "openai_memory_index.v1"
DOC_PATH = "docs/schemas/openai_memory_index.md"


def write_sample_dump(
    index_root: Path,
    *,
    sample_id: str,
    transcript: str,
    memories: list[ExtractedMemory],
) -> Path:
    sample_dir = Path(index_root) / "by_sample" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        sample_dir / "memories.json",
        {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "n_memories": len(memories),
            "memories": [m.to_dict() for m in memories],
        },
    )
    (sample_dir / "transcript.txt").write_text(transcript or "", encoding="utf-8")
    return sample_dir


def load_sample_memories(sample_dir: Path) -> list[ExtractedMemory]:
    import json

    path = Path(sample_dir) / "memories.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    return [ExtractedMemory.from_dict(m) for m in (row.get("memories") or [])]


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
            "gold_answer_in_index": False,
            "object_fields": {
                "memories.json": "extracted timestamped facts (no gold, no top-k)",
                "transcript.txt": "source dialog the extractor saw",
            },
        },
    )
    write_json(root / "run_meta.json", run_meta)
    write_jsonl(root / "index.jsonl", samples)
    return root
