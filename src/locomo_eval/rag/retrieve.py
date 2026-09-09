"""Load a RAG dump and cosine-retrieve top-k chunks."""

from __future__ import annotations

import time
from pathlib import Path

from ..mem0.embeddings import Embedder, cosine_similarity
from .chunk import CHUNK_JOIN, RagChunk
from .dump import load_sample_chunks

RUN_INDEX_HINT = (
    "Run: python -m src.locomo_eval.rag.run_index "
    "--config configs/rag.yaml --run-id <index_run_id>"
)


class MissingRagIndexError(FileNotFoundError):
    """Raised when a RAG builder needs a dump that run_index has not written."""


def sample_index_dir(index_root: Path, sample_id: str) -> Path:
    return Path(index_root) / "by_sample" / sample_id


def require_sample_dump(index_root: Path, sample_id: str) -> Path:
    sample_dir = sample_index_dir(index_root, sample_id)
    if not (sample_dir / "chunks.json").is_file():
        raise MissingRagIndexError(
            f"No RAG index dump for sample {sample_id} under {sample_dir}. "
            f"{RUN_INDEX_HINT}"
        )
    return sample_dir


def retrieve_top_chunks(
    chunks: list[RagChunk],
    query_embedding: list[float],
    k: int,
) -> tuple[list[RagChunk], float]:
    """Return top-k chunks by cosine similarity and search latency (seconds)."""
    t0 = time.perf_counter()
    scored: list[tuple[float, RagChunk]] = []
    for chunk in chunks:
        if not chunk.embedding:
            continue
        scored.append((cosine_similarity(query_embedding, chunk.embedding), chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    k = max(1, int(k))
    picked = [chunk for _score, chunk in scored[:k]]
    if not picked and chunks:
        picked = chunks[:k]
    return picked, time.perf_counter() - t0


def format_retrieved_chunks(chunks: list[RagChunk]) -> str:
    if not chunks:
        return "(No RAG chunks retrieved.)"
    return CHUNK_JOIN.join(c.text for c in chunks)


def retrieve_rag_text(
    index_root: Path,
    sample_id: str,
    query: str,
    embedder: Embedder,
    k: int,
) -> tuple[str, list[str], float]:
    """Load dump → embed query → join top-k. Returns (text, source_ids, search_s)."""
    sample_dir = require_sample_dump(index_root, sample_id)
    _transcript, chunks = load_sample_chunks(sample_dir)
    q_emb = embedder.embed_one(query)
    picked, search_s = retrieve_top_chunks(chunks, q_emb, k)
    return format_retrieved_chunks(picked), [c.chunk_id for c in picked], search_s
