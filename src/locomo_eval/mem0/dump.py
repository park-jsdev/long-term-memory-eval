"""Write experiments/<run_id>/mem0_index/ audit pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..report import write_json, write_jsonl
from .graph_memory import GraphMemory
from .retrieve import dump_complete
from .schemas import SCHEMA_VERSION
from .vector_store import VectorMemoryStore

DOC_PATH = "docs/schemas/mem0_index.md"


def write_sample_dump(
    index_root: Path,
    *,
    sample_id: str,
    store_a: VectorMemoryStore,
    store_b: VectorMemoryStore,
    ingest_log: list[dict[str, Any]],
    graph: GraphMemory | None = None,
) -> Path:
    sample_dir = Path(index_root) / "by_sample" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(sample_dir / "speaker_a.json", store_a.to_dict())
    write_json(sample_dir / "speaker_b.json", store_b.to_dict())
    write_jsonl(sample_dir / "ingest_log.jsonl", ingest_log)
    if graph is not None:
        write_json(sample_dir / "graph.json", graph.to_dict())
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
            "gold_answer_in_index": False,
            "object_fields": {
                "speaker_a.json / speaker_b.json": "vector facts + embeddings",
                "graph.json": "in-memory nodes/edges when enable_graph",
                "ingest_log.jsonl": "pair ids, turn_ids, ops (no gold)",
            },
        },
    )
    write_json(root / "run_meta.json", run_meta)
    write_jsonl(root / "index.jsonl", samples)
    return root


def sample_complete(index_root: Path, sample_id: str, *, enable_graph: bool) -> bool:
    return dump_complete(
        Path(index_root) / "by_sample" / sample_id,
        enable_graph=enable_graph,
    )
