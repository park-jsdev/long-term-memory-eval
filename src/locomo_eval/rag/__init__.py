"""RAG / full-context write-index (Mem0 paper baselines).

Architecture clone of Mem0 ``evaluation/src/rag.py`` (v1.0.10): tiktoken
chunks, ``text-embedding-3-small``, cosine top-k, chunk join ``\\n<->\\n``.
Not a number clone of Table 2. Offline index + online retrieve.

  python -m src.locomo_eval.rag.run_index --config configs/rag.yaml --run-id rag_locomo10
  python -m src.locomo_eval.run --config configs/rag.yaml --run-id rag_k2_256_qa
  python -m src.locomo_eval.run --config configs/full_context.yaml --run-id full_context_qa
"""

from .builders import FullContextMemoryBuilder, RagMemoryBuilder
from .chunk import chunk_transcript, format_conversation_transcript
from .retrieve import MissingRagIndexError, retrieve_rag_text

SCHEMA_VERSION = "rag_index.v1"

__all__ = [
    "SCHEMA_VERSION",
    "FullContextMemoryBuilder",
    "MissingRagIndexError",
    "RagMemoryBuilder",
    "chunk_transcript",
    "format_conversation_transcript",
    "retrieve_rag_text",
]
