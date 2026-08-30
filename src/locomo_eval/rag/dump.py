"""Write experiments/<run_id>/rag_index/ audit pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..report import write_json, write_jsonl
from .chunk import RagChunk

SCHEMA_VERSION = "rag_index.v1"
DOC_PATH = "docs/schemas/rag_index.md"


def chunk_to_dict(chunk: RagChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "n_tokens": chunk.n_tokens,
        "embedding": list(chunk.embedding) if chunk.embedding is not None else None,
    }


def chunk_from_dict(row: dict[str, Any]) -> RagChunk:
    emb = row.get("embedding")
    return RagChunk(
        chunk_id=str(row.get("chunk_id") or ""),
        text=str(row.get("text") or ""),
        n_tokens=int(row.get("n_tokens") or 0),
        embedding=list(emb) if isinstance(emb, list) else None,
    )


def write_sample_dump(
    index_root: Path,
    *,
    sample_id: str,
    transcript: str,
    chunks: list[RagChunk],
    chunk_size: int,
    encoding: str,
) -> Path:
    sample_dir = Path(index_root) / "by_sample" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        sample_dir / "chunks.json",
        {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "chunk_size": chunk_size,
            "encoding": encoding,
            "n_transcript_chars": len(transcript or ""),
            "n_chunks": len(chunks),
            "chunks": [chunk_to_dict(c) for c in chunks],
        },
    )
    (sample_dir / "transcript.txt").write_text(transcript or "", encoding="utf-8")
    return sample_dir


def load_sample_chunks(sample_dir: Path) -> tuple[str, list[RagChunk]]:
    import json

    path = Path(sample_dir) / "chunks.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    chunks = [chunk_from_dict(c) for c in (row.get("chunks") or [])]
    transcript_path = Path(sample_dir) / "transcript.txt"
    transcript = (
        transcript_path.read_text(encoding="utf-8") if transcript_path.is_file() else ""
    )
    return transcript, chunks


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
                "chunks.json": "token windows + embeddings (no gold)",
                "transcript.txt": "full timestamped dialog used for chunking",
            },
        },
    )
    write_json(root / "run_meta.json", run_meta)
    write_jsonl(root / "index.jsonl", samples)
    return root
