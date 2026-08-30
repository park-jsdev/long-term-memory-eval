"""In-memory Mem0g graph (no Neo4j).

GraphMemory is the swap point for a later distilled graph. This file's
Mem0GraphMemory clones the OSS write path: entities → relations →
neighborhood conflict → MERGE (node reuse if cosine >= t).

Conflicting edges are marked valid=false (paper invalidation) rather than
removed (OSS Cypher DELETE).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from ..prompts import load_prompt_template
from ..readers import OpenAIChatCaller
from .embeddings import Embedder, cosine_similarity
from .json_util import parse_json_object
from .schemas import GraphEdge, GraphNode

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENTITIES_PROMPT = ROOT / "prompts" / "mem0g_entities_v1.txt"
DEFAULT_RELATIONS_PROMPT = ROOT / "prompts" / "mem0g_relations_v1.txt"
DEFAULT_CONFLICT_PROMPT = ROOT / "prompts" / "mem0g_conflict_v1.txt"

EntityExtractor = Callable[[str, str], list[dict[str, str]]]
RelationExtractor = Callable[[str, list[dict[str, str]], str], list[dict[str, str]]]
ConflictResolver = Callable[[list[GraphEdge], list[dict[str, str]], str], list[GraphEdge]]


def normalize_entity_name(name: str) -> str:
    return (name or "").strip().casefold().replace(" ", "_")


def mock_extract_entities(text: str, user_id: str) -> list[dict[str, str]]:
    """Heuristic entities for offline tests (no API)."""
    uid = normalize_entity_name(user_id)
    ents = [{"entity": uid, "entity_type": "person"}]
    low = (text or "").casefold()
    extras = (
        ("painting", "activity"),
        ("nursing", "occupation"),
        ("nurse", "occupation"),
        ("boston", "place"),
        ("seattle", "place"),
        ("pizza", "object"),
        ("burger", "object"),
    )
    seen = {uid}
    for token, etype in extras:
        if token in low and token not in seen:
            ents.append({"entity": token, "entity_type": etype})
            seen.add(token)
    return ents


def mock_extract_relations(
    text: str, entities: list[dict[str, str]], user_id: str
) -> list[dict[str, str]]:
    uid = normalize_entity_name(user_id)
    low = (text or "").casefold()
    rels: list[dict[str, str]] = []
    if "painting" in low:
        rels.append({"source": uid, "relationship": "started", "target": "painting"})
    if "nursing" in low or "nurse" in low:
        rels.append({"source": uid, "relationship": "works_as", "target": "nurse"})
    if "boston" in low:
        rels.append({"source": uid, "relationship": "lives_in", "target": "boston"})
    if "seattle" in low:
        rels.append({"source": uid, "relationship": "lives_in", "target": "seattle"})
    if "pizza" in low:
        rels.append({"source": uid, "relationship": "loves_to_eat", "target": "pizza"})
    if "burger" in low:
        rels.append({"source": uid, "relationship": "loves_to_eat", "target": "burger"})
    return rels


def mock_resolve_conflicts(
    existing: list[GraphEdge],
    new_edges: list[dict[str, str]],
    _new_text: str,
) -> list[GraphEdge]:
    """Invalidate same source+rel with a different target (exclusive facts).

    loves_to_eat is kept (Mem0 example: pizza vs burger). lives_in is exclusive.
    """
    exclusive = {"lives_in", "works_as"}
    invalidate: list[GraphEdge] = []
    for old in existing:
        if not old.valid:
            continue
        if old.relationship not in exclusive:
            continue
        for new in new_edges:
            if (
                old.source == new.get("source")
                and old.relationship == new.get("relationship")
                and old.target != new.get("target")
            ):
                invalidate.append(old)
                break
    return invalidate


class GraphMemory(ABC):
    """Swap-out graph store. Indexer depends only on this ABC."""

    @abstractmethod
    def ingest(
        self,
        text: str,
        *,
        user_id: str,
        timestamp: str,
    ) -> list[dict[str, Any]]:
        """Add entities/relations from one user-text blob. Return op log."""
        ...

    @abstractmethod
    def search_relations(self, query_text: str, top_k: int = 5) -> list[GraphEdge]:
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        ...


class Mem0GraphMemory(GraphMemory):
    """Paper-architecture graph in dicts. Later distilled graphs subclass GraphMemory."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = 0.7,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
        conflict_resolver: ConflictResolver | None = None,
    ):
        self.embedder = embedder
        self.threshold = float(threshold)
        self.entity_extractor = entity_extractor or mock_extract_entities
        self.relation_extractor = relation_extractor or mock_extract_relations
        self.conflict_resolver = conflict_resolver or mock_resolve_conflicts
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._edge_n = 0

    def ingest_triples(
        self,
        *,
        entities: list[dict[str, str]],
        relations: list[dict[str, str]],
        user_id: str,
        timestamp: str,
        text: str = "",
    ) -> list[dict[str, Any]]:
        """MERGE + invalidate from already-extracted triples (no LLM).

        Teacher fusion feeds this so the store stays Mem0GraphMemory regardless
        of which model proposed the edges.
        """
        return self._apply_triples(
            entities=entities,
            relations=relations,
            user_id=user_id,
            timestamp=timestamp,
            text=text,
        )

    def _new_edge_id(self) -> str:
        eid = f"e{self._edge_n:04d}"
        self._edge_n += 1
        return eid

    def _merge_node(self, name: str, entity_type: str, timestamp: str) -> GraphNode:
        key = normalize_entity_name(name)
        emb = self.embedder.embed_one(key)
        best: GraphNode | None = None
        best_sim = -1.0
        for node in self.nodes.values():
            sim = cosine_similarity(emb, node.embedding)
            if sim >= self.threshold and sim > best_sim:
                best = node
                best_sim = sim
        if best is not None:
            return best
        node = GraphNode(
            node_id=key,
            name=key,
            entity_type=entity_type or "other",
            embedding=emb,
            timestamp=timestamp,
        )
        self.nodes[key] = node
        return node

    def ingest(
        self,
        text: str,
        *,
        user_id: str,
        timestamp: str,
    ) -> list[dict[str, Any]]:
        entities = self.entity_extractor(text, user_id)
        relations = self.relation_extractor(text, entities, user_id)
        return self._apply_triples(
            entities=entities,
            relations=relations,
            user_id=user_id,
            timestamp=timestamp,
            text=text,
        )

    def _apply_triples(
        self,
        *,
        entities: list[dict[str, str]],
        relations: list[dict[str, str]],
        user_id: str,
        timestamp: str,
        text: str,
    ) -> list[dict[str, Any]]:
        ops: list[dict[str, Any]] = []
        names = {normalize_entity_name(user_id)}
        for ent in entities:
            names.add(normalize_entity_name(ent.get("entity") or ""))
        neighborhood = [
            e
            for e in self.edges
            if e.valid and (e.source in names or e.target in names)
        ]
        to_invalidate = self.conflict_resolver(neighborhood, relations, text)
        for edge in to_invalidate:
            if edge.valid:
                edge.valid = False
                ops.append(
                    {
                        "op": "invalidate",
                        "source": edge.source,
                        "relationship": edge.relationship,
                        "target": edge.target,
                    }
                )
        type_by_name = {
            normalize_entity_name(ent.get("entity") or ""): ent.get("entity_type") or "other"
            for ent in entities
        }
        type_by_name.setdefault(normalize_entity_name(user_id), "person")
        for rel in relations:
            src = normalize_entity_name(rel.get("source") or "")
            tgt = normalize_entity_name(rel.get("target") or "")
            relationship = (rel.get("relationship") or "related_to").strip()
            if not src or not tgt or not relationship:
                continue
            self._merge_node(src, type_by_name.get(src, "other"), timestamp)
            self._merge_node(tgt, type_by_name.get(tgt, "other"), timestamp)
            dup = next(
                (
                    e
                    for e in self.edges
                    if e.valid
                    and e.source == src
                    and e.relationship == relationship
                    and e.target == tgt
                ),
                None,
            )
            if dup:
                continue
            edge = GraphEdge(
                edge_id=self._new_edge_id(),
                source=src,
                relationship=relationship,
                target=tgt,
                valid=True,
                timestamp=timestamp,
            )
            self.edges.append(edge)
            ops.append(
                {
                    "op": "add_edge",
                    "source": src,
                    "relationship": relationship,
                    "target": tgt,
                    "edge_id": edge.edge_id,
                }
            )
        return ops

    def search_relations(self, query_text: str, top_k: int = 5) -> list[GraphEdge]:
        """Score valid edges by cosine of ``source relationship target`` vs query."""
        q = self.embedder.embed_one(query_text or "")
        scored: list[tuple[GraphEdge, float]] = []
        for edge in self.edges:
            if not edge.valid:
                continue
            blob = f"{edge.source} {edge.relationship} {edge.target}"
            scored.append((edge, cosine_similarity(q, self.embedder.embed_one(blob))))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [edge for edge, _ in scored[: max(0, int(top_k))]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "next_edge_n": self._edge_n,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any], embedder: Embedder) -> Mem0GraphMemory:
        g = cls(embedder, threshold=float(row.get("threshold") or 0.7))
        g._edge_n = int(row.get("next_edge_n") or 0)
        for node_row in row.get("nodes") or []:
            node = GraphNode.from_dict(node_row)
            g.nodes[node.node_id] = node
        g.edges = [GraphEdge.from_dict(e) for e in (row.get("edges") or [])]
        if g.edges and g._edge_n == 0:
            g._edge_n = len(g.edges)
        return g


def openai_graph_callables(
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
) -> tuple[EntityExtractor, RelationExtractor, ConflictResolver]:
    """Live entity/relation/conflict LLMs."""
    chat = OpenAIChatCaller(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        min_request_interval_s=min_request_interval_s,
        max_wait_s=max_wait_s,
    )
    _, ent_t = load_prompt_template(DEFAULT_ENTITIES_PROMPT)
    _, rel_t = load_prompt_template(DEFAULT_RELATIONS_PROMPT)
    _, con_t = load_prompt_template(DEFAULT_CONFLICT_PROMPT)

    def entities(text: str, user_id: str) -> list[dict[str, str]]:
        prompt = ent_t.format(user_id=user_id, text=text)
        raw, _ = chat.complete([{"role": "user", "content": prompt}])
        try:
            obj = parse_json_object(raw)
        except ValueError:
            return mock_extract_entities(text, user_id)
        rows = obj.get("entities")
        if not isinstance(rows, list):
            return []
        out = []
        for row in rows:
            if isinstance(row, dict) and row.get("entity"):
                out.append(
                    {
                        "entity": str(row["entity"]),
                        "entity_type": str(row.get("entity_type") or "other"),
                    }
                )
        return out

    def relations(
        text: str, ents: list[dict[str, str]], user_id: str
    ) -> list[dict[str, str]]:
        prompt = rel_t.format(user_id=user_id, entities=ents, text=text)
        raw, _ = chat.complete([{"role": "user", "content": prompt}])
        try:
            obj = parse_json_object(raw)
        except ValueError:
            return mock_extract_relations(text, ents, user_id)
        rows = obj.get("relations")
        if not isinstance(rows, list):
            return []
        out = []
        for row in rows:
            if isinstance(row, dict) and row.get("source") and row.get("target"):
                out.append(
                    {
                        "source": str(row["source"]),
                        "relationship": str(row.get("relationship") or "related_to"),
                        "target": str(row["target"]),
                    }
                )
        return out

    def conflict(
        existing: list[GraphEdge], new_edges: list[dict[str, str]], new_text: str
    ) -> list[GraphEdge]:
        mem_lines = [
            f"{e.source} -- {e.relationship} -- {e.target}"
            for e in existing
            if e.valid
        ]
        prompt = con_t.format(
            user_id="USER",
            existing_memories="\n".join(mem_lines) or "(none)",
            new_text=new_text,
        )
        raw, _ = chat.complete([{"role": "user", "content": prompt}])
        try:
            obj = parse_json_object(raw)
        except ValueError:
            return mock_resolve_conflicts(existing, new_edges, new_text)
        rows = obj.get("invalidate")
        if not isinstance(rows, list):
            return []
        wanted = {
            (
                normalize_entity_name(str(r.get("source") or "")),
                str(r.get("relationship") or ""),
                normalize_entity_name(str(r.get("target") or "")),
            )
            for r in rows
            if isinstance(r, dict)
        }
        return [
            e
            for e in existing
            if (e.source, e.relationship, e.target) in wanted
        ]

    return entities, relations, conflict
