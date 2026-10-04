"""MemoryBuilders that load a Mem0 write-index dump (no re-extract)."""

from __future__ import annotations

from pathlib import Path

from ..memory import MemoryBuilder
from ..schemas import Conversation, Memory, Question
from .embeddings import Embedder, MockEmbedder
from .retrieve import (
    format_mem0_text,
    load_graph,
    load_speaker_store,
    rank_speaker_facts,
    require_sample_dump,
    retrieve_speaker_facts,
)


class Mem0IndexMemoryBuilder(MemoryBuilder):
    """Sandwich middle: load vector dump, cosine top-k for both speakers."""

    name = "mem0"

    def __init__(
        self,
        index_dir: str | Path,
        embedder: Embedder | None = None,
        top_k: int = 30,
        enable_graph: bool = False,
    ):
        self.index_dir = Path(index_dir)
        self.embedder = embedder or MockEmbedder()
        self.top_k = int(top_k)
        self.enable_graph = bool(enable_graph)
        self.retrieve_log: list[dict] = []

    def build(self, conversation: Conversation, question: Question) -> Memory:
        sample_dir = require_sample_dump(
            self.index_dir,
            conversation.sample_id,
            enable_graph=self.enable_graph,
        )
        store_a = load_speaker_store(sample_dir, "a")
        store_b = load_speaker_store(sample_dir, "b")
        q_emb = self.embedder.embed_one(question.question)
        facts_a = retrieve_speaker_facts(store_a, q_emb, self.top_k)
        facts_b = retrieve_speaker_facts(store_b, q_emb, self.top_k)
        edges = None
        source_ids = [f.fact_id for f in facts_a] + [f.fact_id for f in facts_b]
        candidates = rank_speaker_facts(store_a, q_emb, self.top_k)
        candidates.extend(rank_speaker_facts(store_b, q_emb, self.top_k))
        if self.enable_graph:
            from ..experiment_pack.claim_audit import preview

            graph = load_graph(sample_dir, self.embedder)
            ranked_edges = graph.rank_relations(question.question)
            edges = [edge for edge, _score in ranked_edges[: self.top_k]]
            source_ids.extend(e.edge_id for e in edges)
            selected_edge_ids = {e.edge_id for e in edges}
            for i, (edge, score) in enumerate(ranked_edges, start=1):
                candidates.append(
                    {
                        "item_id": edge.edge_id,
                        "item_kind": "graph_edge",
                        "score": round(float(score), 6),
                        "rank": i,
                        "selected": edge.edge_id in selected_edge_ids,
                        "text_preview": preview(
                            f"{edge.source} -- {edge.relationship} -- {edge.target}"
                        ),
                        "source": edge.source,
                        "relationship": edge.relationship,
                        "target": edge.target,
                    }
                )
        from ..experiment_pack.claim_audit import retrieve_rank_row

        self.retrieve_log.append(
            retrieve_rank_row(
                sample_id=conversation.sample_id,
                question_id=question.question_id,
                retriever="mem0g" if self.enable_graph else "mem0",
                top_k=self.top_k,
                candidates=candidates,
            )
        )
        text = format_mem0_text(
            speaker_a=conversation.speaker_a,
            speaker_b=conversation.speaker_b,
            facts_a=facts_a,
            facts_b=facts_b,
            edges=edges,
        )
        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
        )


class Mem0gIndexMemoryBuilder(Mem0IndexMemoryBuilder):
    """Same vector retrieve plus serialized graph relations."""

    name = "mem0g"

    def __init__(
        self,
        index_dir: str | Path,
        embedder: Embedder | None = None,
        top_k: int = 30,
        enable_graph: bool = True,
    ):
        super().__init__(
            index_dir=index_dir,
            embedder=embedder,
            top_k=top_k,
            enable_graph=True,
        )
