"""Mem0 / Mem0g **write-index** package (architecture clone, not the QA runner).

Why this exists
---------------
``MemoryBuilder`` already swaps the sandwich middle (``raw_chunks`` vs
``mem0`` vs …). This package clones Mem0's *method* so that, *inside* that
method, we can freeze write ops (extract + ADD/UPDATE/DELETE/NONE) and the
read path, and swap only ``GraphMemory`` (e.g. a later distilled graph).
That is the stored-graph claim. Multi-teacher / fusion is a different
claim: another ``MemoryBuilder``, not a ``GraphMemory`` subclass.

Architecture clone of Mem0 OSS, not Platform / Neo4j / Qdrant, and **not**
a claim of paper J (66.88 / 68.44).

Two phases (do not fold them into ``run.py``):

1. **Write (this package's CLI):**
   ``python -m src.locomo_eval.mem0.run_index``
   walks HLD (i) session blocks as eval-style speaker pairs, extracts, updates,
   optionally fills ``GraphMemory``, dumps JSON under
   ``experiments/<run_id>/mem0_index/``. Resume is per-sample JSON there —
   not ``predictions.jsonl`` and not ``LlmResponseHash``.
2. **Read (builders, called from ``memory.get_memory_builder``):**
   ``mem0`` / ``mem0g`` load that dump and cosine-retrieve top-k NL facts
   (default 30 per speaker). They must not re-extract. ``run.py`` then uses
   the same frozen reader as ``raw_chunks`` / ``session_summaries``.

Configs: ``configs/mem0.yaml``, ``configs/mem0g.yaml``.
Schema: ``docs/schemas/mem0_index.md``. Tests: ``tests/test_mem0_index.py``.

Claims → what to swap (one middle variable)
------------------------------------------
- **Different memory method** (``raw_chunks`` vs ``mem0`` vs ``mem0g`` vs
  a later multi-teacher builder): freeze reader, prompt, metrics. Swap
  ``MemoryBuilder`` (YAML ``pipeline.memory``). The Mem0 write path is
  irrelevant to this claim except as one of the methods.
- **Different graph/store, same Mem0 method** (``mem0g`` vs distilled
  graph): freeze extract, ADD/UPDATE/DELETE/NONE, retrieve, reader. Swap
  ``GraphMemory`` at **index** time, re-dump, then the same ``mem0g``
  builder loads it. The indexer only calls ``ingest`` / ``to_dict()``;
  dump shape (nodes + edges with source / relationship / target / valid)
  is the contract. A new JSON schema would need ``retrieve.load_graph``
  changed. LLM callables on ``Mem0GraphMemory`` are a smaller knob inside
  this same slot (same MERGE / invalidation, different models).

Not this package
----------------
- ``mem0_baseline.yaml`` / ``prompts/qa_mem0_v1.txt`` — *evaluation* parity
  (GPT-4o-mini answer prompt over ``session_summaries``). No extract/update.
- ``autorater.py`` / ``mem0_metrics.py`` / ``mem0_baselines.py`` — judge and
  literature tables over *finished predictions*, not the write-index.
- ``preprocess/`` — deterministic session-block dump (no LLM). This indexer
  consumes those blocks via ``DataIngestor`` + ``PreprocessingPipeline``.

Public names below are the seam ``run_index``, ``memory.py``, and tests import.
Factories reject a non-null ``llm_response_hash`` (same cache-free rule as
the answer reader).
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
