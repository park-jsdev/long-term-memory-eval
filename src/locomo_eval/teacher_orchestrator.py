"""HLD (ii) teacher orchestration — pooling, fusion, locked Mem0 graph.

Walks one SessionBlock at a time (same ids preprocess recorded). With no
teachers attached this stays a passthrough seam. With teachers it:

1. selects which models to call (single / round_robin / random / equal_weight)
2. collects GraphProposal triples (gold never enters)
3. fuses (union or majority_vote)
4. writes into Mem0GraphMemory.ingest_triples (locked graph schema)

This file is software, not one giant LLM call.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from random import Random
from typing import Any

from .fusion import (
    FUSION_NONE,
    FUSION_RESOLVE_POLICIES,
    POOL_SINGLE,
    GraphProposal,
    fuse_proposals,
    fusion_relation_audit,
    fusion_slot_audit,
    select_teacher_ids,
)
from .mem0.embeddings import Embedder, MockEmbedder
from .mem0.graph_memory import Mem0GraphMemory, normalize_entity_name
from .mem0.retrieve import format_edge_line
from .mem0.schemas import GraphEdge
from .schemas import ProcessedConversation, SessionBlock
from .teachers import Teacher, teacher_call_record


def iter_session_blocks(processed: ProcessedConversation) -> Iterator[SessionBlock]:
    """Yield session blocks in preprocess order (one block at a time)."""
    yield from processed.session_blocks


def process_session_block(block: SessionBlock) -> dict[str, Any]:
    """Identity record for one session when no teacher is attached."""
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
    """Speaker lines for one preprocess block. Shared teacher input."""
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


class TeacherOrchestrator:
    """Software controller: session blocks → pooled/fused Mem0GraphMemory.

    Empty ``teachers`` keeps the old passthrough walk for preprocess tests.
    """

    def __init__(
        self,
        teachers: Sequence[Teacher] | None = None,
        *,
        pool: str = POOL_SINGLE,
        fusion: str = FUSION_NONE,
        min_votes: int | None = None,
        rng_seed: int = 0,
        embedder: Embedder | None = None,
        graph_threshold: float = 0.7,
        thinking: bool | None = None,
    ):
        self.teachers = list(teachers or [])
        self.pool = (pool or POOL_SINGLE).strip().lower()
        self.fusion = (fusion or FUSION_NONE).strip().lower()
        self.min_votes = min_votes
        self.rng_seed = int(rng_seed)
        self.embedder = embedder or MockEmbedder()
        self.graph_threshold = float(graph_threshold)
        self.thinking = thinking
        self.call_log: list[dict[str, Any]] = []
        self.fusion_log: list[dict[str, Any]] = []

    @property
    def teacher_model(self) -> str | None:
        if not self.teachers:
            return None
        return "+".join(t.model_name for t in self.teachers)

    @property
    def teacher_provider(self) -> str | None:
        if not self.teachers:
            return None
        providers = [t.provider for t in self.teachers]
        if len(set(providers)) == 1:
            return providers[0]
        return "+".join(providers)

    def iter_session_blocks(self, processed: ProcessedConversation) -> Iterator[SessionBlock]:
        return iter_session_blocks(processed)

    def process_session_block(self, block: SessionBlock) -> dict[str, Any]:
        if not self.teachers:
            return process_session_block(block)
        proposals = self._propose(block, session_index=block.session_index, rng=Random(self.rng_seed))
        return {
            **process_session_block(block),
            "status": "teacher",
            "pool": self.pool,
            "fusion": self.fusion,
            "n_proposals": len(proposals),
            "teacher_ids": [p.teacher_id for p in proposals],
        }

    def build_graph(self, processed: ProcessedConversation) -> Mem0GraphMemory:
        """Walk every session block and MERGE fused triples into Mem0GraphMemory."""
        graph = Mem0GraphMemory(self.embedder, threshold=self.graph_threshold)
        rng = Random(self.rng_seed)
        for block in self.iter_session_blocks(processed):
            proposals = self._propose(block, session_index=block.session_index, rng=rng)
            entities, relations = fuse_proposals(
                proposals,
                fusion=self.fusion,
                min_votes=self.min_votes,
                rng=rng,
                session_index=block.session_index,
                teacher_order=[p.teacher_id for p in proposals],
            )
            audit_relations = fusion_relation_audit(proposals, relations)
            slot_audit = (
                fusion_slot_audit(proposals, relations)
                if self.fusion in FUSION_RESOLVE_POLICIES
                else []
            )
            self.fusion_log.append(
                {
                    "sample_id": block.sample_id,
                    "session_id": block.session_id,
                    "session_index": block.session_index,
                    "pool": self.pool,
                    "fusion": self.fusion,
                    "teacher_ids": [p.teacher_id for p in proposals],
                    "n_kept_entities": len(entities),
                    "n_kept_relations": len(relations),
                    "relations": audit_relations,
                    "slots": slot_audit,
                }
            )
            graph.ingest_triples(
                entities=entities,
                relations=relations,
                user_id=block.speaker_a,
                timestamp=block.date_time_raw or block.date_time_normalized or "",
                text=format_session_block(block),
            )
        return graph

    def _propose(
        self,
        block: SessionBlock,
        *,
        session_index: int,
        rng: Random,
    ) -> list[GraphProposal]:
        if not self.teachers:
            return []
        session_text = format_session_block(block)
        user_id = normalize_entity_name(block.speaker_a) or block.speaker_a
        ids = [t.teacher_id for t in self.teachers]
        chosen = set(
            select_teacher_ids(ids, pool=self.pool, session_index=session_index, rng=rng)
        )
        proposals: list[GraphProposal] = []
        for teacher in self.teachers:
            if teacher.teacher_id not in chosen:
                continue
            entities, relations, meta = teacher.extract_session_graph(
                session_text=session_text,
                user_id=user_id,
            )
            self.call_log.append(
                teacher_call_record(
                    teacher=teacher,
                    meta=meta,
                    sample_id=block.sample_id,
                    session_id=block.session_id,
                    session_index=session_index,
                    entities=entities,
                    relations=relations,
                )
            )
            proposals.append(
                GraphProposal(
                    teacher_id=teacher.teacher_id,
                    provider=teacher.provider,
                    model=teacher.model_name,
                    entities=entities,
                    relations=relations,
                    session_id=block.session_id,
                    call_meta=meta,
                )
            )
        return proposals


def valid_edges(graph: Mem0GraphMemory) -> list[GraphEdge]:
    return [e for e in graph.edges if e.valid]
