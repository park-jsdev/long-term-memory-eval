"""Session blocks → Mem0 vector stores (+ optional graph). No QA."""

from __future__ import annotations

from typing import Any

from ..schemas import ProcessedConversation
from .embeddings import Embedder
from .extract import FactExtractor
from .graph_memory import GraphMemory
from .ingest import iter_speaker_pairs, user_messages_text
from .update import MemoryUpdater, apply_update_events
from .vector_store import VectorMemoryStore


class Mem0Indexer:
    """Walk pairs, extract, update top-s, optionally ingest graph text."""

    def __init__(
        self,
        *,
        extractor: FactExtractor,
        updater: MemoryUpdater,
        embedder: Embedder,
        graph: GraphMemory | None = None,
        batch_size: int = 2,
        similar_s: int = 10,
        max_sessions: int | None = None,
    ):
        self.extractor = extractor
        self.updater = updater
        self.embedder = embedder
        self.graph = graph
        self.batch_size = int(batch_size)
        self.similar_s = int(similar_s)
        self.max_sessions = max_sessions

    def index_conversation(
        self, processed: ProcessedConversation
    ) -> tuple[VectorMemoryStore, VectorMemoryStore, list[dict[str, Any]]]:
        store_a = VectorMemoryStore(
            sample_id=processed.sample_id,
            speaker_index="a",
            speaker_name=processed.speaker_a,
        )
        store_b = VectorMemoryStore(
            sample_id=processed.sample_id,
            speaker_index="b",
            speaker_name=processed.speaker_b,
        )
        stores = {"a": store_a, "b": store_b}
        log: list[dict[str, Any]] = []
        for pair in iter_speaker_pairs(
            processed,
            batch_size=self.batch_size,
            max_sessions=self.max_sessions,
        ):
            store = stores[pair.speaker_index]
            facts, extract_meta = self.extractor.extract(pair)
            applied: list[dict[str, Any]] = []
            for fact_text in facts:
                query_emb = self.embedder.embed_one(fact_text)
                hits = store.search(query_emb, self.similar_s)
                candidates = [fact for fact, _ in hits]
                events, _update_meta = self.updater.propose(
                    new_facts=[fact_text],
                    candidates=candidates,
                )
                applied.extend(
                    apply_update_events(
                        store,
                        events,
                        candidates=candidates,
                        embedder=self.embedder,
                        timestamp=pair.timestamp,
                        source_turn_ids=list(pair.turn_ids),
                        speaker_index=pair.speaker_index,
                    )
                )
            graph_ops: list[dict[str, Any]] = []
            user_text = user_messages_text(pair)
            if self.graph is not None and user_text:
                graph_ops = self.graph.ingest(
                    user_text,
                    user_id=pair.speaker_name,
                    timestamp=pair.timestamp,
                )
            log.append(
                {
                    "pair_id": pair.pair_id,
                    "sample_id": pair.sample_id,
                    "session_id": pair.session_id,
                    "speaker_index": pair.speaker_index,
                    "speaker_name": pair.speaker_name,
                    "timestamp": pair.timestamp,
                    "turn_ids": list(pair.turn_ids),
                    "n_extracted": len(facts),
                    "ops": applied,
                    "graph_ops": graph_ops,
                    "extract_model": extract_meta.get("model"),
                }
            )
        return store_a, store_b, log
