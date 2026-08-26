"""Mem0 write-index types.

Write path (offline index, not run.py QA):
    SessionBlock → message pairs → FactExtractor → MemoryUpdater
        → VectorMemoryStore (+ optional GraphMemory) → JSON dump

Read seam (later eval, no re-extract):
    dump → retrieve top-k → Memory.text
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "mem0_index.v1"

ADD = "ADD"
UPDATE = "UPDATE"
DELETE = "DELETE"
NONE = "NONE"
UPDATE_EVENTS = (ADD, UPDATE, DELETE, NONE)


@dataclass
class RoleMessage:
    """One turn in a Mem0 ingest pair (eval add.py role-flip)."""

    role: str  # user | assistant
    content: str
    speaker: str
    turn_id: str
    source_dia_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MessagePair:
    """batch_size consecutive turns for one speaker index.

    Who consumes it: FactExtractor (user-role text only) and ingest_log.
    """

    pair_id: str
    sample_id: str
    session_id: int
    session_index: int
    speaker_index: str  # "a" | "b"
    speaker_name: str
    timestamp: str
    messages: list[RoleMessage]
    turn_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fact:
    """One stored natural-language memory in a speaker vector index."""

    fact_id: str
    text: str
    timestamp: str
    embedding: list[float] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    speaker_index: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Fact:
        return cls(
            fact_id=str(row["fact_id"]),
            text=str(row["text"]),
            timestamp=str(row.get("timestamp") or ""),
            embedding=[float(x) for x in (row.get("embedding") or [])],
            source_turn_ids=list(row.get("source_turn_ids") or []),
            speaker_index=str(row.get("speaker_index") or ""),
        )


@dataclass
class UpdateEvent:
    """One ADD/UPDATE/DELETE/NONE decision from MemoryUpdater."""

    event: str
    text: str
    local_id: str | None = None
    old_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphNode:
    """Entity node in the in-memory Mem0g graph (no Neo4j)."""

    node_id: str
    name: str
    entity_type: str
    embedding: list[float] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> GraphNode:
        return cls(
            node_id=str(row["node_id"]),
            name=str(row["name"]),
            entity_type=str(row.get("entity_type") or "other"),
            embedding=[float(x) for x in (row.get("embedding") or [])],
            timestamp=str(row.get("timestamp") or ""),
        )


@dataclass
class GraphEdge:
    """Directed labeled edge. ``valid=False`` is paper-style invalidation."""

    edge_id: str
    source: str
    relationship: str
    target: str
    valid: bool = True
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> GraphEdge:
        return cls(
            edge_id=str(row["edge_id"]),
            source=str(row["source"]),
            relationship=str(row["relationship"]),
            target=str(row["target"]),
            valid=bool(row.get("valid", True)),
            timestamp=str(row.get("timestamp") or ""),
        )
