"""Pool and fuse teacher graph proposals (software, not an LLM).

Policies
--------
Pool (which teachers run this session):
  single        — one teacher (interchangeable model)
  round_robin   — cycle teachers across sessions
  random        — one teacher per session (seeded)
  equal_weight  — all teachers

Fusion (how overlapping triples are kept):
  none              — union of pooled triples (dedupe exact keys)
  majority_vote     — keep a triple if >= k teachers propose the exact same triple
                      (default k = ceil(n_teachers / 2))

Resolve (pick one target when teachers disagree on source+relationship):
  resolve_top_voted     — highest vote count; 1-2 splits pick the pair
  resolve_first         — first teacher in roster order wins the slot
  resolve_random        — uniform random among tied targets (seeded)
  resolve_round_robin   — cycle among tied targets by session index
  resolve_confidence    — sum of per-relation confidence × teacher weight

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
FUSION_RESOLVE_TOP_VOTED = "resolve_top_voted"
FUSION_RESOLVE_FIRST = "resolve_first"
FUSION_RESOLVE_RANDOM = "resolve_random"
FUSION_RESOLVE_ROUND_ROBIN = "resolve_round_robin"
FUSION_RESOLVE_CONFIDENCE = "resolve_confidence"
FUSION_RESOLVE_POLICIES = (
    FUSION_RESOLVE_TOP_VOTED,
    FUSION_RESOLVE_FIRST,
    FUSION_RESOLVE_RANDOM,
    FUSION_RESOLVE_ROUND_ROBIN,
    FUSION_RESOLVE_CONFIDENCE,
)
FUSION_POLICIES = (FUSION_NONE, FUSION_MAJORITY) + FUSION_RESOLVE_POLICIES


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


def relation_slot_key(rel: dict[str, str]) -> tuple[str, str]:
    """Normalized (source, relationship) — competing targets form a disagreement."""
    src, relationship, _ = relation_key(rel)
    return (src, relationship)


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


def _relation_confidence(rel: dict[str, str], call_meta: dict[str, Any]) -> float:
    """Per-relation weight for confidence-aware fusion. Defaults to 1.0."""
    for key in ("confidence", "score", "weight"):
        raw = rel.get(key)
        if raw is not None:
            try:
                val = float(raw)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
    raw = call_meta.get("confidence")
    if raw is not None:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return 1.0


def _teacher_order(proposals: list[GraphProposal]) -> list[str]:
    seen: list[str] = []
    for prop in proposals:
        if prop.teacher_id not in seen:
            seen.append(prop.teacher_id)
    return seen


@dataclass
class _SlotCandidate:
    target: str
    example: dict[str, str]
    teachers: list[str] = field(default_factory=list)
    votes: int = 0
    weight: float = 0.0


def _collect_slot_candidates(
    proposals: list[GraphProposal],
    *,
    use_confidence: bool = False,
) -> dict[tuple[str, str], dict[str, _SlotCandidate]]:
    """Group proposals by (source, relationship) → target candidates."""
    slots: dict[tuple[str, str], dict[str, _SlotCandidate]] = {}
    for prop in proposals:
        seen: set[tuple[str, str, str]] = set()
        for rel in prop.relations:
            row = canonical_relation(rel)
            key = relation_key(row)
            if not key[0] or not key[2] or key in seen:
                continue
            seen.add(key)
            slot = relation_slot_key(row)
            by_target = slots.setdefault(slot, {})
            cand = by_target.setdefault(
                row["target"],
                _SlotCandidate(target=row["target"], example=row),
            )
            cand.teachers.append(prop.teacher_id)
            cand.votes += 1
            w = _relation_confidence(rel, prop.call_meta) if use_confidence else 1.0
            cand.weight += w
    return slots


def _pick_top_voted(candidates: dict[str, _SlotCandidate], teacher_order: list[str]) -> str:
    best_votes = max(c.votes for c in candidates.values())
    tied = [c for c in candidates.values() if c.votes == best_votes]
    if len(tied) == 1:
        return tied[0].target
    order = {tid: i for i, tid in enumerate(teacher_order)}
    tied.sort(key=lambda c: min(order.get(t, 10**9) for t in c.teachers))
    return tied[0].target


def _pick_first(candidates: dict[str, _SlotCandidate], teacher_order: list[str]) -> str:
    order = {tid: i for i, tid in enumerate(teacher_order)}
    best_idx = 10**9
    chosen = next(iter(candidates))
    for cand in candidates.values():
        idx = min(order.get(t, 10**9) for t in cand.teachers)
        if idx < best_idx:
            best_idx = idx
            chosen = cand.target
    return chosen


def _pick_confidence(candidates: dict[str, _SlotCandidate], teacher_order: list[str]) -> str:
    best_weight = max(c.weight for c in candidates.values())
    tied = [c for c in candidates.values() if abs(c.weight - best_weight) < 1e-9]
    if len(tied) == 1:
        return tied[0].target
    return _pick_first({c.target: c for c in tied}, teacher_order)


def _pick_random(
    candidates: dict[str, _SlotCandidate],
    *,
    rng: random.Random,
) -> str:
    best_votes = max(c.votes for c in candidates.values())
    tied = [c.target for c in candidates.values() if c.votes == best_votes]
    return rng.choice(tied)


def _pick_round_robin(
    candidates: dict[str, _SlotCandidate],
    *,
    session_index: int,
    slot: tuple[str, str],
    teacher_order: list[str],
) -> str:
    best_votes = max(c.votes for c in candidates.values())
    tied = sorted(c.target for c in candidates.values() if c.votes == best_votes)
    if len(tied) == 1:
        return tied[0]
    slot_hash = hash(slot) & 0xFFFF
    return tied[(session_index + slot_hash) % len(tied)]


def fuse_resolve(
    proposals: list[GraphProposal],
    *,
    strategy: str,
    rng: random.Random | None = None,
    session_index: int = 0,
    teacher_order: list[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Pick one target per (source, relationship) slot when teachers disagree."""
    if not proposals:
        return [], []
    policy = (strategy or FUSION_RESOLVE_TOP_VOTED).strip().lower()
    if policy not in FUSION_RESOLVE_POLICIES:
        raise ValueError(f"Unknown resolve strategy '{strategy}'. Use {FUSION_RESOLVE_POLICIES}.")
    order = teacher_order or _teacher_order(proposals)
    rng = rng or random.Random(0)
    use_confidence = policy == FUSION_RESOLVE_CONFIDENCE
    slots = _collect_slot_candidates(proposals, use_confidence=use_confidence)
    kept_rels: list[dict[str, str]] = []
    for slot, candidates in sorted(slots.items()):
        if not candidates:
            continue
        if policy == FUSION_RESOLVE_FIRST:
            target = _pick_first(candidates, order)
        elif policy == FUSION_RESOLVE_RANDOM:
            target = _pick_random(candidates, rng=rng)
        elif policy == FUSION_RESOLVE_ROUND_ROBIN:
            target = _pick_round_robin(
                candidates, session_index=session_index, slot=slot, teacher_order=order
            )
        elif policy == FUSION_RESOLVE_CONFIDENCE:
            target = _pick_confidence(candidates, order)
        else:
            target = _pick_top_voted(candidates, order)
        kept_rels.append(candidates[target].example)
    mentioned = {r["source"] for r in kept_rels} | {r["target"] for r in kept_rels}
    ents = [e for e in _union_entities(proposals) if e["entity"] in mentioned]
    return ents, kept_rels


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
    """Keep triples proposed by at least ``min_votes`` teachers (exact triple match).

    Default threshold is a simple majority. Entities are the union of those
    that appear on kept triples.
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
    rng: random.Random | None = None,
    session_index: int = 0,
    teacher_order: list[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Collapse teacher proposals into one entity/relation set."""
    policy = (fusion or FUSION_NONE).strip().lower()
    if policy not in FUSION_POLICIES:
        raise ValueError(f"Unknown fusion policy '{fusion}'. Use {FUSION_POLICIES}.")
    if not proposals:
        return [], []
    if policy == FUSION_MAJORITY:
        return fuse_majority(proposals, min_votes=min_votes)
    if policy in FUSION_RESOLVE_POLICIES:
        return fuse_resolve(
            proposals,
            strategy=policy,
            rng=rng,
            session_index=session_index,
            teacher_order=teacher_order,
        )
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


def fusion_slot_audit(
    proposals: list[GraphProposal],
    kept_relations: list[dict[str, str]],
    *,
    use_confidence: bool = False,
) -> list[dict[str, Any]]:
    """Per-slot disagreement table for resolve-* fusion policies."""
    slots = _collect_slot_candidates(proposals, use_confidence=use_confidence)
    kept_by_slot = {relation_slot_key(r): r["target"] for r in kept_relations}
    rows: list[dict[str, Any]] = []
    for slot, candidates in sorted(slots.items()):
        src, relationship = slot
        options = []
        for cand in sorted(candidates.values(), key=lambda c: c.target):
            options.append(
                {
                    "target": cand.target,
                    "proposed_by": list(cand.teachers),
                    "votes": cand.votes,
                    "weight": round(cand.weight, 4),
                }
            )
        rows.append(
            {
                "source": src,
                "relationship": relationship,
                "n_targets": len(options),
                "disagreement": len(options) > 1,
                "options": options,
                "kept_target": kept_by_slot.get(slot),
            }
        )
    return rows
