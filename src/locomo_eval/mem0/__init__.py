"""Mem0 / Mem0g write-index (extract + update + optional in-memory graph).

Not a LoCoMo QA runner. Index with ``python -m src.locomo_eval.mem0.run_index``,
then load dumps via ``mem0`` / ``mem0g`` MemoryBuilders.
"""

from .builders import Mem0IndexMemoryBuilder, Mem0gIndexMemoryBuilder
from .embeddings import MockEmbedder, get_embedder
from .extract import MockFactExtractor, get_fact_extractor
from .graph_memory import GraphMemory, Mem0GraphMemory
from .ingest import iter_speaker_pairs, user_messages_text
from .indexer import Mem0Indexer
from .retrieve import MissingMem0IndexError
from .schemas import SCHEMA_VERSION
from .update import MockMemoryUpdater, apply_update_events, get_memory_updater

__all__ = [
    "SCHEMA_VERSION",
    "GraphMemory",
    "Mem0GraphMemory",
    "Mem0IndexMemoryBuilder",
    "Mem0Indexer",
    "Mem0gIndexMemoryBuilder",
    "MissingMem0IndexError",
    "MockEmbedder",
    "MockFactExtractor",
    "MockMemoryUpdater",
    "apply_update_events",
    "get_embedder",
    "get_fact_extractor",
    "get_memory_updater",
    "iter_speaker_pairs",
    "user_messages_text",
]
