"""Pool and fuse teacher graph proposals (software, not an LLM).

Policies
--------
Pool (which teacher's triples enter the store this session):
  single        — one teacher (interchangeable model)
  round_robin   — cycle teachers across sessions
  random        — one teacher per session (seeded)
  equal_weight  — all teachers; union of triples (equal vote)

Fusion (how overlapping triples are kept):
  none            — ingest the pooled set as-is
  majority_vote   — keep a triple if it appears in >= k teachers
                    (default k = ceil(n_teachers / 2))

The store is always Mem0GraphMemory.ingest_triples (locked graph schema).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .mem0.graph_memory import normalize_entity_name

POOL_SINGLE = "single"
POOL_ROUND_ROBIN = "round_robin"
POOL_RANDOM = "random"
POOL_EQUAL_WEIGHT = "equal_weight"
POOL_POLICIES = (POOL_SINGLE, POOL_ROUND_ROBIN, POOL_RANDOM, POOL_EQUAL_WEIGHT)

FUSION_NONE = "none"
FUSION_MAJORITY = "majority_vote"
FUSION_POLICIES = (FUSION_NONE, FUSION_MAJORITY)


@dataclass
class GraphProposal:
    """One teacher's extracted graph fragment for one session.

    Who consumes it: pool_proposals / fuse_proposals, then Mem0GraphMemory.
    """

    teacher_id: str
    provider: str
    model: str
    entities: list[dict[str, str]] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    session_id: int = 0
    call_meta: dict[str, Any] = field(default_factory=dict)


def relation_key(rel: dict[str, str]) -> tuple[str, str, str]:
    """Normalized (source, relationship, target) for counting votes."""
    return (
        normalize_entity_name(rel.get("source") or ""),
        (rel.get("relationship") or "related_to").strip().casefold().replace(" ", "_"),
        normalize_entity_name(rel.get("target") or ""),
    )


def entity_key(ent: dict[str, str]) -> str:
    return normalize_entity_name(ent.get("entity") or "")


def canonical_relation(rel: dict[str, str]) -> dict[str, str]:
    src, relationship, tgt = relation_key(rel)
    return {"source": src, "relationship": relationship, "target": tgt}


def canonical_entity(ent: dict[str, str]) -> dict[str, str]:
    return {
        "entity": entity_key(ent),
        "entity_type": (ent.get("entity_type") or "other").strip() or "other",
    }


def select_teacher_ids(
    teacher_ids: list[str],
    *,
    pool: str,
    session_index: int,
    rng: random.Random,
) -> list[str]:
    """Which teachers to call for this session. Does not fuse triples."""
    if not teacher_ids:
        return []
    policy = (pool or POOL_SINGLE).strip().lower()
    if policy not in POOL_POLICIES:
        raise ValueError(f"Unknown pool policy '{pool}'. Use {POOL_POLICIES}.")
    if policy == POOL_SINGLE:
        return [teacher_ids[0]]
    if policy == POOL_ROUND_ROBIN:
        return [teacher_ids[session_index % len(teacher_ids)]]
    if policy == POOL_RANDOM:
        return [rng.choice(teacher_ids)]
    return list(teacher_ids)


def pool_proposals(
    proposals: list[GraphProposal],
    *,
    pool: str,
    session_index: int,
    rng: random.Random,
) -> list[GraphProposal]:
    """Subset of already-collected proposals according to the pool policy."""
    if not proposals:
        return []
    ids = select_teacher_ids(
        [p.teacher_id for p in proposals],
        pool=pool,
        session_index=session_index,
        rng=rng,
    )
    wanted = set(ids)
    return [p for p in proposals if p.teacher_id in wanted]


def _union_entities(proposals: list[GraphProposal]) -> list[dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    for prop in proposals:
        for ent in prop.entities:
            row = canonical_entity(ent)
            if row["entity"]:
                by_name.setdefault(row["entity"], row)
    return list(by_name.values())


def _union_relations(proposals: list[GraphProposal]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for prop in proposals:
        for rel in prop.relations:
            row = canonical_relation(rel)
            key = relation_key(row)
            if key[0] and key[2]:
                by_key.setdefault(key, row)
    return list(by_key.values())


def fuse_majority(
    proposals: list[GraphProposal],
    *,
    min_votes: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep triples proposed by at least ``min_votes`` teachers.

    Default threshold is a simple majority. Skeleton: later we can add
    weighted votes, validators, or claim-level evidence. Entities are the
    union of those that appear on kept triples (plus any majority entities).
    """
    n = len(proposals)
    if n == 0:
        return [], []
    k = int(min_votes) if min_votes is not None else max(1, math.ceil(n / 2))
    votes: dict[tuple[str, str, str], int] = {}
    examples: dict[tuple[str, str, str], dict[str, str]] = {}
    for prop in proposals:
        seen: set[tuple[str, str, str]] = set()
        for rel in prop.relations:
            row = canonical_relation(rel)
            key = relation_key(row)
            if not key[0] or not key[2] or key in seen:
                continue
            seen.add(key)
            votes[key] = votes.get(key, 0) + 1
            examples.setdefault(key, row)
    kept_rels = [examples[key] for key, count in votes.items() if count >= k]
    mentioned = {r["source"] for r in kept_rels} | {r["target"] for r in kept_rels}
    ents = [
        e
        for e in _union_entities(proposals)
        if e["entity"] in mentioned
    ]
    return ents, kept_rels


def fuse_proposals(
    proposals: list[GraphProposal],
    *,
    fusion: str = FUSION_NONE,
    min_votes: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Collapse teacher proposals into one entity/relation set."""
    policy = (fusion or FUSION_NONE).strip().lower()
    if policy not in FUSION_POLICIES:
        raise ValueError(f"Unknown fusion policy '{fusion}'. Use {FUSION_POLICIES}.")
    if not proposals:
        return [], []
    if policy == FUSION_MAJORITY:
        return fuse_majority(proposals, min_votes=min_votes)
    return _union_entities(proposals), _union_relations(proposals)


def fusion_relation_audit(
    proposals: list[GraphProposal],
    kept_relations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Per-triple vote table: which teachers proposed it and whether fusion kept it."""
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for prop in proposals:
        seen: set[tuple[str, str, str]] = set()
        for rel in prop.relations:
            row = canonical_relation(rel)
            key = relation_key(row)
            if not key[0] or not key[2] or key in seen:
                continue
            seen.add(key)
            rec = by_key.setdefault(
                key,
                {
                    "source": row["source"],
                    "relationship": row["relationship"],
                    "target": row["target"],
                    "proposed_by": [],
                    "votes": 0,
                },
            )
            rec["proposed_by"].append(prop.teacher_id)
            rec["votes"] += 1
    kept = {relation_key(r) for r in kept_relations}
    rows = [{**rec, "kept": key in kept} for key, rec in by_key.items()]
    rows.sort(key=lambda r: (r["source"], r["relationship"], r["target"]))
    return rows
