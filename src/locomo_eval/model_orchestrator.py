"""One writer model turns session blocks into a locked Mem0 graph.

Walks one SessionBlock at a time (same ids preprocess recorded). With no
model attached this stays a passthrough seam. With one model it extracts
triples and MERGE-writes them into Mem0GraphMemory. Gold never enters
the model prompt. This file is software, not one giant LLM call.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .mem0.embeddings import Embedder, MockEmbedder
from .mem0.graph_memory import Mem0GraphMemory, normalize_entity_name
from .mem0.retrieve import format_edge_line
from .mem0.schemas import GraphEdge
from .schemas import ProcessedConversation, SessionBlock
from .writer_model import Writer, writer_call_record


def iter_session_blocks(processed: ProcessedConversation) -> Iterator[SessionBlock]:
    """Yield session blocks in preprocess order (one block at a time)."""
    yield from processed.session_blocks


def process_session_block(block: SessionBlock) -> dict[str, Any]:
    """Identity record for one session when no writer model is attached."""
    return {
        "sample_id": block.sample_id,
        "session_id": block.session_id,
        "session_index": block.session_index,
        "source_key": block.source_key,
        "n_turns": len(block.turns),
        "turn_ids": [t.turn_id for t in block.turns],
        "status": "passthrough",
    }


def format_session_block(block: SessionBlock) -> str:
    """Speaker lines for one preprocess block. Shared model input."""
    lines = [
        f"DATE: {block.date_time_raw}",
        f"SESSION {block.session_id}:",
    ]
    for turn in block.turns:
        line = f"{turn.speaker_raw}: {turn.text}"
        if turn.blip_caption:
            line += f" [image: {turn.blip_caption}]"
        if turn.source_dia_id:
            line += f" ({turn.source_dia_id})"
        lines.append(line)
    return "\n".join(lines)


def format_graph_memory_text(
    graph: Mem0GraphMemory,
    *,
    speaker_a: str,
    speaker_b: str,
) -> str:
    """Serialize valid Mem0g edges into the frozen reader {memory} string."""
    chunks = [
        f"Conversation between {speaker_a} and {speaker_b}.",
        "",
        "Graph relations:",
    ]
    valid = [e for e in graph.edges if e.valid]
    if valid:
        chunks.extend(format_edge_line(e) for e in valid)
    else:
        chunks.append("(none)")
    return "\n".join(chunks).strip()


class ModelOrchestrator:
    """Software controller: one writer model → locked Mem0GraphMemory.

    ``model`` None keeps the passthrough walk for preprocess tests.
    """

    def __init__(
        self,
        model: Writer | None = None,
        *,
        embedder: Embedder | None = None,
        graph_threshold: float = 0.7,
        thinking: bool | None = None,
    ):
        self.model = model
        self.embedder = embedder or MockEmbedder()
        self.graph_threshold = float(graph_threshold)
        self.thinking = thinking
        self.call_log: list[dict[str, Any]] = []
        self.ingest_log: list[dict[str, Any]] = []
        self.session_texts: dict[tuple[str, int], dict[str, Any]] = {}

    @property
    def model_name(self) -> str | None:
        if self.model is None:
            return None
        return self.model.model_name

    @property
    def writer_model(self) -> str | None:
        return self.model_name

    @property
    def model_provider(self) -> str | None:
        if self.model is None:
            return None
        return self.model.provider

    @property
    def writer_provider(self) -> str | None:
        return self.model_provider

    def iter_session_blocks(self, processed: ProcessedConversation) -> Iterator[SessionBlock]:
        return iter_session_blocks(processed)

    def process_session_block(self, block: SessionBlock) -> dict[str, Any]:
        if self.model is None:
            return process_session_block(block)
        return {
            **process_session_block(block),
            "status": "model",
            "model_id": self.model.writer_id,
        }

    def build_graph(self, processed: ProcessedConversation) -> Mem0GraphMemory:
        """Walk every session block and MERGE triples into Mem0GraphMemory."""
        graph = Mem0GraphMemory(self.embedder, threshold=self.graph_threshold)
        for block in self.iter_session_blocks(processed):
            entities, relations = self._extract(block)
            ops = graph.ingest_triples(
                entities=entities,
                relations=relations,
                user_id=block.speaker_a,
                timestamp=block.date_time_raw or block.date_time_normalized or "",
                text=format_session_block(block),
            )
            n_valid = sum(1 for e in graph.edges if e.valid)
            self.ingest_log.append(
                {
                    "sample_id": block.sample_id,
                    "session_id": block.session_id,
                    "session_index": block.session_index,
                    "n_entities": len(entities),
                    "n_relations": len(relations),
                    "n_ops": len(ops),
                    "n_add_edge": sum(1 for op in ops if op.get("op") == "add_edge"),
                    "n_invalidate": sum(1 for op in ops if op.get("op") == "invalidate"),
                    "n_skip_dup": sum(1 for op in ops if op.get("op") == "skip_dup"),
                    "n_new_node": sum(1 for op in ops if op.get("op") == "new_node"),
                    "n_reuse_node": sum(1 for op in ops if op.get("op") == "reuse_node"),
                    "n_nodes_after": len(graph.nodes),
                    "n_edges_after": len(graph.edges),
                    "n_valid_edges_after": n_valid,
                    "ops": ops,
                }
            )
        return graph

    def _extract(
        self,
        block: SessionBlock,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        if self.model is None:
            return [], []
        model = self.model
        session_text = format_session_block(block)
        self.session_texts[(block.sample_id, int(block.session_id))] = {
            "sample_id": block.sample_id,
            "session_id": block.session_id,
            "session_index": block.session_index,
            "text": session_text,
        }
        user_id = normalize_entity_name(block.speaker_a) or block.speaker_a
        entities, relations, meta = model.extract_session_graph(
            session_text=session_text,
            user_id=user_id,
        )
        self.call_log.append(
            writer_call_record(
                writer=model,
                meta=meta,
                sample_id=block.sample_id,
                session_id=block.session_id,
                session_index=block.session_index,
                entities=entities,
                relations=relations,
                session_text=session_text,
            )
        )
        return entities, relations


def valid_edges(graph: Mem0GraphMemory) -> list[GraphEdge]:
    return [e for e in graph.edges if e.valid]
