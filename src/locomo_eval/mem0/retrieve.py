"""Load dumps and format retrieved memories for the later eval seam.

Does not call extract. Missing dumps name the run_index command.
"""

from __future__ import annotations

from pathlib import Path

from .embeddings import Embedder
from .graph_memory import Mem0GraphMemory
from .schemas import Fact, GraphEdge
from .vector_store import VectorMemoryStore

RUN_INDEX_HINT = (
    "Run: python -m src.locomo_eval.mem0.run_index "
    "--config configs/mem0.yaml --run-id <index_run_id>"
)


class MissingMem0IndexError(FileNotFoundError):
    """Raised when a MemoryBuilder needs a dump that run_index has not written."""


def sample_index_dir(index_root: Path, sample_id: str) -> Path:
    return Path(index_root) / "by_sample" / sample_id


def dump_complete(sample_dir: Path, *, enable_graph: bool) -> bool:
    needed = [
        sample_dir / "speaker_a.json",
        sample_dir / "speaker_b.json",
    ]
    if enable_graph:
        needed.append(sample_dir / "graph.json")
    return all(p.is_file() for p in needed)


def format_fact_line(fact: Fact) -> str:
    ts = fact.timestamp or "unknown"
    return f"{ts}: {fact.text}"


def format_edge_line(edge: GraphEdge) -> str:
    return f"{edge.source} -- {edge.relationship} -- {edge.target}"


def retrieve_speaker_facts(
    store: VectorMemoryStore,
    query_embedding: list[float],
    top_k: int,
) -> list[Fact]:
    return [fact for fact, _score in store.search(query_embedding, top_k)]


def format_mem0_text(
    *,
    speaker_a: str,
    speaker_b: str,
    facts_a: list[Fact],
    facts_b: list[Fact],
    edges: list[GraphEdge] | None = None,
) -> str:
    """Eval-hook string: timestamped facts (+ optional graph relations)."""
    chunks: list[str] = [f"Speaker {speaker_a} memories:"]
    if facts_a:
        chunks.extend(format_fact_line(f) for f in facts_a)
    else:
        chunks.append("(none)")
    chunks.append("")
    chunks.append(f"Speaker {speaker_b} memories:")
    if facts_b:
        chunks.extend(format_fact_line(f) for f in facts_b)
    else:
        chunks.append("(none)")
    if edges is not None:
        chunks.append("")
        chunks.append("Graph relations:")
        valid = [e for e in edges if e.valid]
        if valid:
            chunks.extend(format_edge_line(e) for e in valid)
        else:
            chunks.append("(none)")
    return "\n".join(chunks).strip()


def load_speaker_store(sample_dir: Path, speaker_index: str) -> VectorMemoryStore:
    import json

    path = Path(sample_dir) / f"speaker_{speaker_index}.json"
    if not path.is_file():
        raise MissingMem0IndexError(
            f"No Mem0 speaker_{speaker_index} dump at {path}. {RUN_INDEX_HINT}"
        )
    row = json.loads(path.read_text(encoding="utf-8"))
    return VectorMemoryStore.from_dict(row)


def load_graph(sample_dir: Path, embedder: Embedder) -> Mem0GraphMemory:
    import json

    path = Path(sample_dir) / "graph.json"
    if not path.is_file():
        raise MissingMem0IndexError(f"No Mem0g graph dump at {path}. {RUN_INDEX_HINT}")
    row = json.loads(path.read_text(encoding="utf-8"))
    return Mem0GraphMemory.from_dict(row, embedder)


def require_sample_dump(index_root: Path, sample_id: str, *, enable_graph: bool) -> Path:
    sample_dir = sample_index_dir(index_root, sample_id)
    if not dump_complete(sample_dir, enable_graph=enable_graph):
        raise MissingMem0IndexError(
            f"No Mem0 index dump for sample {sample_id} under {sample_dir}. "
            f"{RUN_INDEX_HINT}"
        )
    return sample_dir
