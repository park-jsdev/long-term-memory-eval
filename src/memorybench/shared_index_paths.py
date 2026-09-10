"""Shared write-index ids (Mem0 extract / RAG chunks) reused across reader cells.

The longitudinal sweep freezes the writer: one gpt-4o-mini Mem0 dump and one
text-embedding-3-small RAG dump, then varies only the answer model.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MEM0_INDEX = "mem0_locomo10"
DEFAULT_RAG_INDEX = "rag_locomo10"


def index_run_ids_for_memory(
    memory_method: str, shared: dict[str, Any] | None = None
) -> dict[str, str | None]:
    shared = shared or {}
    mem0_id = _pick(shared, "mem0", DEFAULT_MEM0_INDEX)
    rag_id = _pick(shared, "rag", DEFAULT_RAG_INDEX)
    method = str(memory_method)
    out: dict[str, str | None] = {"mem0": None, "rag": None}
    if method in ("mem0", "mem0g"):
        out["mem0"] = mem0_id
    if method == "rag":
        out["rag"] = rag_id
    return out


def _pick(shared: dict[str, Any], key: str, default: str) -> str:
    block = shared.get(key) or {}
    if isinstance(block, str):
        return block
    return str(block.get("index_run_id") or default)
