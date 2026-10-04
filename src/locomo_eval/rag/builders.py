"""MemoryBuilders for Mem0 paper RAG and full-context baselines."""

from __future__ import annotations

from pathlib import Path

from ..mem0.embeddings import Embedder, MockEmbedder
from ..memory import MemoryBuilder
from ..schemas import Conversation, Memory, Question
from .chunk import format_conversation_transcript
from .retrieve import retrieve_rag_with_ranks


class RagMemoryBuilder(MemoryBuilder):
    """Sandwich middle: load RAG dump, cosine top-k token chunks.

    Does not re-chunk or re-embed the conversation. Question-dependent.
    """

    name = "rag"

    def __init__(
        self,
        index_dir: str | Path,
        embedder: Embedder | None = None,
        top_k: int = 2,
    ):
        self.index_dir = Path(index_dir)
        self.embedder = embedder or MockEmbedder()
        self.top_k = int(top_k)
        self.retrieve_log: list[dict] = []

    def build(self, conversation: Conversation, question: Question) -> Memory:
        text, source_ids, search_s, ranks = retrieve_rag_with_ranks(
            self.index_dir,
            conversation.sample_id,
            question.question,
            self.embedder,
            self.top_k,
        )
        from ..experiment_pack.claim_audit import retrieve_rank_row

        self.retrieve_log.append(
            retrieve_rank_row(
                sample_id=conversation.sample_id,
                question_id=question.question_id,
                retriever="rag",
                top_k=self.top_k,
                candidates=ranks,
                search_latency_s=round(float(search_s), 6),
            )
        )
        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
            search_latency_s=round(float(search_s), 6),
        )


class FullContextMemoryBuilder(MemoryBuilder):
    """Entire timestamped dialog as Memory.text (Mem0 full-context baseline).

    No retrieval. Search latency is 0. Does not use the RAG dump.
    """

    name = "full_context"

    def build(self, conversation: Conversation, question: Question) -> Memory:
        text = format_conversation_transcript(conversation).strip()
        if not text:
            text = "(No dialogue turns available for this conversation.)"
        source_ids = [
            turn.dia_id or f"session_{session.session_id}"
            for session in conversation.sessions
            for turn in session.turns
        ]
        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
            search_latency_s=0.0,
        )
