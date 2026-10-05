"""Claim-level pack audit: lineage, ranks, writer quality, cost, SUMMARY.

Call traces (reader/writer JSONL) record that an LLM ran. These helpers
record **what entered {memory}** and who is responsible for each item, so a
reviewer can audit a QA claim without re-running models.

``attribution_call_rows`` joins each LLM call to its pipeline role and the
claims that call produced (triples, session summaries, predicted answers).

Pure functions. ``audit_writer`` dumps the dicts; ``run.py`` collects inputs.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from ..mem0.graph_memory import normalize_entity_name
from ..pricing import add_costs, estimate_usd
from ..pricing import default_pricing

PREVIEW_CHARS = 200

# USD per 1M tokens. Rollup pins, not invoices. Unknown models stay token-only.
# Source of truth: configs/models/pricing.yaml
try:
    PRICING_AS_OF = default_pricing().as_of
except FileNotFoundError:
    PRICING_AS_OF = "unknown"


def relation_key(rel: dict[str, str]) -> tuple[str, str, str]:
    """Normalized (source, relationship, target) so call JSON matches ingest rows."""
    return (
        normalize_entity_name(rel.get("source") or ""),
        (rel.get("relationship") or "related_to").strip().casefold().replace(" ", "_"),
        normalize_entity_name(rel.get("target") or ""),
    )


def sha256_text(text: str) -> str:
    """Stable content hash for session text / memory dumps (audit, not security)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def preview(text: str, n: int = PREVIEW_CHARS) -> str:
    """One-line clip for JSONL/SUMMARY so reviewers need not open the full blob."""
    compact = " ".join((text or "").split())
    if len(compact) <= n:
        return compact
    return compact[:n] + "…"


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int, int]:
    """Normalize OpenAI- vs Anthropic-shaped usage dicts to prompt/completion/total."""
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return prompt, completion, total


def _reasoning_tokens(usage: dict[str, Any] | None, *, fallback: Any = None) -> int:
    """Thinking tokens from usage, else a call-level ``reasoning_tokens`` field."""
    usage = usage or {}
    raw = usage.get("reasoning_tokens")
    if raw is None:
        raw = usage.get("reasoning_output_tokens")
    if raw is None:
        raw = fallback
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _session_id_key(session_id: Any) -> str:
    """Join key so int ``1`` and str ``\"1\"`` hit the same ingest row."""
    if session_id is None:
        return ""
    return str(session_id)


def ranked_candidates(
    items: list[dict[str, Any]],
    *,
    selected_ids: set[str],
    id_key: str = "item_id",
) -> list[dict[str, Any]]:
    """Mark 1-based rank + selected. ``items`` must already be score-sorted.

    Losers stay in the list so retrieve audit is not winners-only.
    """
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        item_id = str(item.get(id_key) or "")
        row = dict(item)
        row["rank"] = i
        row["selected"] = item_id in selected_ids if item_id else bool(item.get("selected"))
        rows.append(row)
    return rows


def retrieve_rank_row(
    *,
    sample_id: str,
    question_id: str,
    retriever: str,
    top_k: int | None,
    candidates: list[dict[str, Any]],
    search_latency_s: float | None = None,
) -> dict[str, Any]:
    """One ``retrieve_ranks.jsonl`` row: full candidate list for one question."""
    selected = [c for c in candidates if c.get("selected")]
    return {
        "sample_id": sample_id,
        "question_id": question_id,
        "retriever": retriever,
        "top_k": top_k,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "search_latency_s": search_latency_s,
        "candidates": candidates,
    }


def writer_quality_stats(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse and yield rates per writer. Software stats, not an LLM judge.

    One writer. These counts do not score whether a triple is true.
    """
    by_writer: dict[str, dict[str, Any]] = {}
    for call in calls:
        tid = str(call.get("writer_id") or "writer")
        rec = by_writer.setdefault(
            tid,
            {
                "writer_id": tid,
                "provider": call.get("provider"),
                "model": call.get("model"),
                "n_calls": 0,
                "n_parse_ok": 0,
                "n_parse_fallback": 0,
                "n_parse_unset": 0,
                "n_entities": 0,
                "n_relations": 0,
                "latency_sum_s": 0.0,
                "n_latency": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "usd": None,
            },
        )
        rec["n_calls"] += 1
        rec["n_entities"] += int(call.get("n_entities") or 0)
        rec["n_relations"] += int(call.get("n_relations") or 0)
        parse = call.get("parse")
        if parse in ("ok", "mock"):
            rec["n_parse_ok"] += 1
        elif parse:
            rec["n_parse_fallback"] += 1
        else:
            rec["n_parse_unset"] += 1
        latency = call.get("latency_s")
        if latency is not None:
            rec["latency_sum_s"] += float(latency)
            rec["n_latency"] += 1
        prompt, completion, total = _usage_tokens(call.get("usage"))
        rec["prompt_tokens"] += prompt
        rec["completion_tokens"] += completion
        rec["total_tokens"] += total
        rec["reasoning_tokens"] += _reasoning_tokens(
            call.get("usage"), fallback=call.get("reasoning_tokens")
        )

    for rec in by_writer.values():
        rec["mean_latency_s"] = (
            round(rec["latency_sum_s"] / rec["n_latency"], 4) if rec["n_latency"] else None
        )
        rec["usd"] = estimate_usd(rec.get("model"), rec["prompt_tokens"], rec["completion_tokens"])
        rec.pop("latency_sum_s", None)
        rec.pop("n_latency", None)

    return {"by_writer": by_writer}


def cost_rollup(
    *,
    reader_traces: list[dict[str, Any]],
    writer_calls: list[dict[str, Any]],
    extra_usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Token + optional USD totals for reader vs writer vs other.

    Mock traces with empty usage cost $0. Unknown models stay token-only
    (``usd=null``). Not an invoice.
    """

    def _bucket(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
        """One cost bucket (reader / writer / other) with per-model split."""
        prompt = completion = total = reasoning = 0
        usd: float | None = 0.0
        any_priced = False
        by_model: dict[str, dict[str, Any]] = {}
        for row in rows:
            model = str(row.get("model") or "")
            p, c, t = _usage_tokens(row.get("usage"))
            r = _reasoning_tokens(row.get("usage"), fallback=row.get("reasoning_tokens"))
            prompt += p
            completion += c
            total += t
            reasoning += r
            rec = by_model.setdefault(
                model or "(unknown)",
                {
                    "model": model or None,
                    "role": role,
                    "n_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "reasoning_tokens": 0,
                    "usd": None,
                },
            )
            rec["n_calls"] += 1
            rec["prompt_tokens"] += p
            rec["completion_tokens"] += c
            rec["total_tokens"] += t
            rec["reasoning_tokens"] += r
            priced = estimate_usd(model, p, c)
            if priced is not None:
                any_priced = True
                rec["usd"] = add_costs(rec["usd"] if rec["usd"] else 0.0, priced)
                usd = add_costs(usd, priced)
            elif rec["usd"] is None and not any_priced:
                pass
        if not any_priced:
            usd = 0.0 if prompt == 0 and completion == 0 else None
        return {
            "role": role,
            "n_calls": len(rows),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "reasoning_tokens": reasoning,
            "usd": usd,
            "by_model": by_model,
        }

    reader = _bucket(reader_traces, "reader")
    writer = _bucket(writer_calls, "writer")
    extra = _bucket(extra_usage or [], "other")
    total_usd = add_costs(reader["usd"], writer["usd"], extra["usd"])
    if reader["usd"] is None or writer["usd"] is None:
        if (reader["prompt_tokens"] + writer["prompt_tokens"]) == 0:
            total_usd = 0.0
        elif reader["usd"] is None and writer["usd"] is None and extra["usd"] is None:
            total_usd = None
    return {
        "pricing_as_of": PRICING_AS_OF,
        "pricing_note": (
            "USD uses pinned list prices from configs/models/pricing.yaml. "
            "Unknown models contribute tokens but usd=null. Not an invoice."
        ),
        "reader": reader,
        "writer": writer,
        "other": extra,
        "total": {
            "prompt_tokens": reader["prompt_tokens"] + writer["prompt_tokens"] + extra["prompt_tokens"],
            "completion_tokens": (
                reader["completion_tokens"] + writer["completion_tokens"] + extra["completion_tokens"]
            ),
            "total_tokens": reader["total_tokens"] + writer["total_tokens"] + extra["total_tokens"],
            "reasoning_tokens": (
                reader["reasoning_tokens"] + writer["reasoning_tokens"] + extra["reasoning_tokens"]
            ),
            "usd": total_usd,
        },
    }


def graph_edge_provenance(
    ingest_rows: list[dict[str, Any]],
    writer_calls: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (sample_id, edge_id) to the writer who ingested that edge.

    Key includes sample_id so two conversations that both mint ``e0000``
    cannot leak writers. An ingested edge is kept: one writer, no vote.
    """
    writers_by_session: dict[tuple[str, str], list[str]] = {}
    for call in writer_calls or []:
        sample_id = str(call.get("sample_id") or "")
        session_id = _session_id_key(call.get("session_id"))
        writer_id = str(call.get("writer_id") or "")
        if not writer_id:
            continue
        bucket = writers_by_session.setdefault((sample_id, session_id), [])
        if writer_id not in bucket:
            bucket.append(writer_id)
    by_edge: dict[tuple[str, str], dict[str, Any]] = {}
    for session in ingest_rows:
        sample_id = str(session.get("sample_id") or "")
        session_id = session.get("session_id")
        proposed = writers_by_session.get((sample_id, _session_id_key(session_id)), [])
        for op in session.get("ops") or []:
            if op.get("op") != "add_edge":
                continue
            edge_id = str(op.get("edge_id") or "")
            if not edge_id:
                continue
            by_edge[(sample_id, edge_id)] = {
                "item_id": edge_id,
                "item_kind": "graph_edge",
                "sample_id": sample_id,
                "session_id": session_id,
                "source": op.get("source"),
                "relationship": op.get("relationship"),
                "target": op.get("target"),
                "proposed_by": list(proposed),
                "votes": None,
                "kept": True,
                "text_preview": preview(
                    f"{op.get('source')} -- {op.get('relationship')} -- {op.get('target')}"
                ),
            }
    return by_edge


def _memory_for_question(
    *,
    question_id: str,
    sample_id: str,
    memories_by_question: dict[str, Any] | None,
    memories_by_sample: dict[str, Any] | None,
) -> Any | None:
    """Memory object used for this question (per-Q retrieve, else per-sample dump)."""
    if memories_by_question and question_id in memories_by_question:
        return memories_by_question[question_id]
    if memories_by_sample and sample_id in memories_by_sample:
        return memories_by_sample[sample_id]
    return None


def _index_retrieve_ranks(
    retrieve_ranks: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index rank rows by (sample_id, question_id). Missing sample_id is ``\"\"``."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in retrieve_ranks:
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        out[(str(row.get("sample_id") or ""), qid)] = row
    return out


def _rank_row_for(
    ranks_by: dict[tuple[str, str], dict[str, Any]],
    *,
    sample_id: str,
    question_id: str,
) -> dict[str, Any] | None:
    """Rank list for this question; sample_id blocks a sibling conversation's list."""
    row = ranks_by.get((sample_id, question_id))
    if row is not None:
        return row
    if sample_id:
        return ranks_by.get(("", question_id))
    return None


def _writer_for_session(
    calls: list[dict[str, Any]],
    *,
    sample_id: str,
    session_id: Any,
) -> list[str]:
    """Writer ids that ran this sample/session (for dump-all summary lineage)."""
    ids: list[str] = []
    seen: set[str] = set()
    for call in calls:
        if str(call.get("sample_id")) != str(sample_id):
            continue
        if call.get("session_id") != session_id and str(call.get("session_id")) != str(session_id):
            continue
        tid = str(call.get("writer_id") or "")
        if tid and tid not in seen:
            seen.add(tid)
            ids.append(tid)
    return ids


def lineage_rows(
    *,
    prediction_rows: list[dict[str, Any]],
    memories_by_sample: dict[str, Any] | None = None,
    memories_by_question: dict[str, Any] | None = None,
    retrieve_ranks: list[dict[str, Any]] | None = None,
    ingest_rows: list[dict[str, Any]] | None = None,
    writer_calls: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One lineage row per injected memory item: question → item → writer.

    Prefers retrieve ranks (selected only), then graph ``edge_id`` joined
    to the writer call for that session, then ``Memory.source_ids``.
    """
    ranks_by = _index_retrieve_ranks(retrieve_ranks or [])
    edge_prov = graph_edge_provenance(ingest_rows or [], writer_calls)
    calls = writer_calls or []
    rows: list[dict[str, Any]] = []
    for pred in prediction_rows:
        qid = str(pred.get("question_id") or "")
        sample_id = str(pred.get("sample_id") or "")
        memory = _memory_for_question(
            question_id=qid,
            sample_id=sample_id,
            memories_by_question=memories_by_question,
            memories_by_sample=memories_by_sample,
        )
        source_ids = list(getattr(memory, "source_ids", None) or [])
        if getattr(memory, "memory_type", None) == "full_context":
            # The complete conversation is one injected payload. Expanding every
            # turn for every question creates millions of duplicate audit rows.
            rows.append(
                {
                    "question_id": qid,
                    "sample_id": sample_id,
                    "item_id": f"full_context:{sample_id}",
                    "item_kind": "full_context_payload",
                    "rank": 1,
                    "score": None,
                    "session_id": None,
                    "text_preview": preview(str(getattr(memory, "text", "") or "")),
                    "proposed_by": [],
                    "votes": None,
                    "source_id_count": len(source_ids),
                }
            )
            continue
        rank_row = _rank_row_for(ranks_by, sample_id=sample_id, question_id=qid)
        if rank_row:
            for cand in rank_row.get("candidates") or []:
                if not cand.get("selected"):
                    continue
                rows.append(
                    {
                        "question_id": qid,
                        "sample_id": sample_id,
                        "item_id": cand.get("item_id"),
                        "item_kind": cand.get("item_kind") or rank_row.get("retriever"),
                        "rank": cand.get("rank"),
                        "score": cand.get("score"),
                        "session_id": cand.get("session_id"),
                        "text_preview": cand.get("text_preview") or preview(str(cand.get("text") or "")),
                        "proposed_by": list(cand.get("proposed_by") or []),
                        "votes": cand.get("votes"),
                    }
                )
            continue
        if source_ids and any(
            edge_prov.get((sample_id, str(sid))) is not None for sid in source_ids
        ):
            for item_id in source_ids:
                prov = edge_prov.get((sample_id, str(item_id)))
                if prov is None:
                    rows.append(
                        {
                            "question_id": qid,
                            "sample_id": sample_id,
                            "item_id": item_id,
                            "item_kind": "graph_edge",
                            "rank": None,
                            "score": None,
                            "session_id": None,
                            "text_preview": "",
                            "proposed_by": [],
                            "votes": None,
                        }
                    )
                    continue
                rows.append(
                    {
                        "question_id": qid,
                        "sample_id": sample_id,
                        **{k: prov[k] for k in (
                            "item_id",
                            "item_kind",
                            "session_id",
                            "text_preview",
                            "proposed_by",
                            "votes",
                            "kept",
                            "source",
                            "relationship",
                            "target",
                        )},
                        "rank": None,
                        "score": None,
                    }
                )
            continue
        for item_id in source_ids:
            session_id = None
            kind = "memory_item"
            if str(item_id).startswith("session_") and (
                str(item_id).endswith("_summary") or str(item_id).endswith("_writer")
            ):
                kind = "session_summary"
                try:
                    session_id = int(str(item_id).split("_")[1])
                except (IndexError, ValueError):
                    session_id = None
            proposers = (
                _writer_for_session(calls, sample_id=sample_id, session_id=session_id)
                if kind == "session_summary"
                else []
            )
            rows.append(
                {
                    "question_id": qid,
                    "sample_id": sample_id,
                    "item_id": item_id,
                    "item_kind": kind,
                    "rank": None,
                    "score": None,
                    "session_id": session_id,
                    "text_preview": "",
                    "proposed_by": proposers,
                    "votes": len(proposers) or None,
                }
            )
    return rows


# Sandwich layer for each LLM role (write = writer, read = answer, judge = autorater).
ROLE_LAYER = {
    "reader": "read",
    "agent": "read",
    "writer": "write",
    "graph": "write",
    "autorater": "judge",
}

# Cap human ATTRIBUTION.md; JSONL stays complete.
ATTRIBUTION_MD_CALL_CAP = 12


def infer_llm_role(row: dict[str, Any]) -> str:
    """Sandwich role for one trace row when ``role`` was omitted on older dumps.

    Prefer the logged field. Fallback: triples → ``graph``, a writer
    session → ``writer``, ``question_id`` → ``reader``.
    """
    raw = row.get("role")
    if raw:
        return str(raw)
    if row.get("relations") or row.get("n_relations"):
        return "graph"
    if row.get("writer_id") or row.get("session_id") is not None:
        return "writer"
    if row.get("question_id"):
        return "reader"
    return "writer"


def _injected_questions_for_triple(
    lineage: list[dict[str, Any]],
    *,
    writer_id: str,
    sample_id: str,
    session_id: Any,
    rel: dict[str, Any],
) -> list[str]:
    """Question ids whose injected graph edge is this writer's triple."""
    src, relationship, tgt = relation_key(rel)
    qids: list[str] = []
    seen: set[str] = set()
    for row in lineage:
        if str(row.get("sample_id") or "") != str(sample_id):
            continue
        proposers = [str(t) for t in (row.get("proposed_by") or [])]
        if writer_id not in proposers:
            continue
        if row.get("session_id") is not None and session_id is not None:
            if row.get("session_id") != session_id and str(row.get("session_id")) != str(
                session_id
            ):
                continue
        if not (row.get("source") or row.get("relationship") or row.get("target")):
            continue
        rsrc, rrel, rtgt = relation_key(row)
        if (rsrc, rrel, rtgt) != (src, relationship, tgt):
            continue
        qid = str(row.get("question_id") or "")
        if qid and qid not in seen:
            seen.add(qid)
            qids.append(qid)
    return qids


def _injected_questions_for_summary(
    lineage: list[dict[str, Any]],
    *,
    writer_id: str,
    sample_id: str,
    session_id: Any,
) -> list[str]:
    """Question ids that injected this writer's session summary."""
    qids: list[str] = []
    seen: set[str] = set()
    for row in lineage:
        if str(row.get("sample_id") or "") != str(sample_id):
            continue
        kind = str(row.get("item_kind") or "")
        if kind != "session_summary":
            continue
        proposers = [str(t) for t in (row.get("proposed_by") or [])]
        if writer_id and writer_id not in proposers:
            continue
        if session_id is not None and row.get("session_id") is not None:
            if row.get("session_id") != session_id and str(row.get("session_id")) != str(
                session_id
            ):
                continue
        qid = str(row.get("question_id") or "")
        if qid and qid not in seen:
            seen.add(qid)
            qids.append(qid)
    return qids


def _graph_claims_for_call(
    call: dict[str, Any],
    *,
    lineage: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Graph-triple claims for one writer call. Ingested triples are kept."""
    writer_id = str(call.get("writer_id") or "")
    sample_id = str(call.get("sample_id") or "")
    session_id = call.get("session_id")
    rels = list(call.get("relations") or [])
    claims: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for rel in rels:
        key = relation_key(rel)
        if not key[0] or not key[2] or key in seen:
            continue
        seen.add(key)
        src, relationship, tgt = key
        claims.append(
            {
                "claim_kind": "graph_triple",
                "source": src,
                "relationship": relationship,
                "target": tgt,
                "text_preview": preview(f"{src} -- {relationship} -- {tgt}"),
                "kept": True,
                "votes": None,
                "proposed_by": [writer_id] if writer_id else [],
                "injected_question_ids": _injected_questions_for_triple(
                    lineage,
                    writer_id=writer_id,
                    sample_id=sample_id,
                    session_id=session_id,
                    rel={"source": src, "relationship": relationship, "target": tgt},
                ),
            }
        )
    return claims


def _writer_call_row(
    call: dict[str, Any],
    *,
    index: int,
    lineage: list[dict[str, Any]],
) -> dict[str, Any]:
    """One attribution row for a write-path call (triples or a session summary)."""
    role = infer_llm_role(call)
    writer_id = str(call.get("writer_id") or "")
    sample_id = str(call.get("sample_id") or "")
    session_id = call.get("session_id")
    if role == "graph" or call.get("relations") or call.get("n_relations"):
        role = "graph" if role in ("writer", "graph") else role
        claims = _graph_claims_for_call(call, lineage=lineage)
    else:
        text = str(call.get("output_text") or "")
        claims = []
        if text.strip():
            claims.append(
                {
                    "claim_kind": "session_summary",
                    "text_preview": preview(text),
                    "session_id": session_id,
                    "kept": True,
                    "injected_question_ids": _injected_questions_for_summary(
                        lineage,
                        writer_id=writer_id,
                        sample_id=sample_id,
                        session_id=session_id,
                    ),
                }
            )
    prompt, completion, total = _usage_tokens(call.get("usage"))
    return {
        "call_id": ":".join(
            [
                role,
                writer_id or "writer",
                sample_id or "-",
                str(session_id if session_id is not None else "-"),
                str(index),
            ]
        ),
        "role": role,
        "layer": ROLE_LAYER.get(role, "write"),
        "provider": call.get("provider"),
        "model": call.get("model"),
        "writer_id": writer_id or None,
        "sample_id": sample_id or None,
        "session_id": session_id,
        "question_id": None,
        "parse": call.get("parse"),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "n_claims": len(claims),
        "n_kept": sum(1 for c in claims if c.get("kept") is True),
        "n_injected": sum(1 for c in claims if c.get("injected_question_ids")),
        "claims": claims,
        "used_memory_items": [],
    }


def _reader_call_row(
    trace: dict[str, Any],
    *,
    index: int,
    lineage: list[dict[str, Any]],
) -> dict[str, Any]:
    """One attribution row for the answer LLM: predicted answer + used memory items."""
    qid = str(trace.get("question_id") or "")
    sample_id = str(trace.get("sample_id") or "")
    answer = str(trace.get("predicted_answer") or "")
    used = []
    for row in lineage:
        if str(row.get("question_id") or "") != qid:
            continue
        row_sid = str(row.get("sample_id") or "")
        if sample_id and row_sid and row_sid != sample_id:
            continue
        used.append(
            {
                "item_id": row.get("item_id"),
                "item_kind": row.get("item_kind"),
                "proposed_by": list(row.get("proposed_by") or []),
                "text_preview": row.get("text_preview") or "",
            }
        )
    claims = []
    if answer or qid:
        claims.append(
            {
                "claim_kind": "predicted_answer",
                "text_preview": preview(answer) if answer else "",
                "question_id": qid or None,
                "n_memory_items": len(used),
            }
        )
    prompt, completion, total = _usage_tokens(trace.get("usage"))
    role = infer_llm_role(trace) if trace.get("role") else "reader"
    if role not in ("reader", "agent"):
        role = "reader"
    return {
        "call_id": ":".join([role, qid or "-", str(index)]),
        "role": role,
        "layer": ROLE_LAYER.get(role, "read"),
        "provider": trace.get("provider"),
        "model": trace.get("model"),
        "writer_id": None,
        "sample_id": trace.get("sample_id"),
        "session_id": None,
        "question_id": qid or None,
        "parse": None,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "n_claims": len(claims),
        "n_kept": None,
        "n_injected": len(used),
        "claims": claims,
        "used_memory_items": used,
    }


def attribution_call_rows(
    *,
    reader_traces: list[dict[str, Any]],
    writer_calls: list[dict[str, Any]],
    lineage: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One row per LLM call: role + claims that call made.

    Writer: graph triples or session summaries, joined to lineage when the
    injected question ids exist.
    Reader: predicted answer plus the memory items that question used.
    """
    lineage = lineage or []
    rows: list[dict[str, Any]] = []
    for i, call in enumerate(writer_calls):
        rows.append(_writer_call_row(call, index=i, lineage=lineage))
    for i, trace in enumerate(reader_traces):
        rows.append(_reader_call_row(trace, index=i, lineage=lineage))
    return rows


def attribution_role_summary(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Counts per pipeline role / model for SUMMARY and ATTRIBUTION.md tables.

    Software rollup of call rows, not an LLM judge.
    """
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    order: list[tuple[str, ...]] = []
    for call in calls:
        key = (
            str(call.get("role") or ""),
            str(call.get("layer") or ""),
            str(call.get("writer_id") or ""),
            str(call.get("provider") or ""),
            str(call.get("model") or ""),
        )
        rec = grouped.get(key)
        if rec is None:
            rec = {
                "role": call.get("role"),
                "layer": call.get("layer"),
                "writer_id": call.get("writer_id"),
                "provider": call.get("provider"),
                "model": call.get("model"),
                "n_calls": 0,
                "n_claims": 0,
                "n_kept": 0,
                "n_injected": 0,
            }
            grouped[key] = rec
            order.append(key)
        rec["n_calls"] += 1
        rec["n_claims"] += int(call.get("n_claims") or 0)
        rec["n_kept"] += int(call.get("n_kept") or 0)
        rec["n_injected"] += int(call.get("n_injected") or 0)
    return [grouped[k] for k in order]


def render_attribution_md(
    *,
    run_id: str,
    calls: list[dict[str, Any]],
    roles: list[dict[str, Any]],
) -> str:
    """Human report: LLM call → pipeline role → claims made."""
    lines = [
        f"# Attribution and LLM roles: `{run_id}`",
        "",
        "Each row is one LLM **call**: the pipeline **role** it played, and the",
        "**claims** that call produced. Teachers propose memory claims (triples",
        "or session summaries). The reader proposes a QA answer from injected",
        "`{memory}`. This file is the join; `attribution.jsonl` is the machine copy.",
        "",
        "## Roles this run",
        "",
        "| role | layer | actor | model | calls | claims made | kept | used / injected |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    if roles:
        for rec in roles:
            actor = rec.get("writer_id") or rec.get("role")
            lines.append(
                f"| `{rec.get('role')}` | {rec.get('layer')} | `{actor}` | "
                f"`{rec.get('model')}` | {rec.get('n_calls')} | {rec.get('n_claims')} | "
                f"{rec.get('n_kept')} | {rec.get('n_injected')} |"
            )
    else:
        lines.append("| (none) | | | | 0 | 0 | 0 | 0 |")
    lines.extend(["", "## Write-path calls", ""])
    writers = [c for c in calls if c.get("layer") == "write"]
    if not writers:
        lines.extend(["(no writer calls this run)", ""])
    else:
        for call in writers[:ATTRIBUTION_MD_CALL_CAP]:
            loc = f"sample `{call.get('sample_id')}` session `{call.get('session_id')}`"
            lines.append(
                f"### `{call.get('role')}` · `{call.get('writer_id')}` · "
                f"`{call.get('model')}` · {loc}"
            )
            lines.append("")
            if not call.get("claims"):
                lines.append("- (no parsed claims)")
            for claim in call.get("claims") or []:
                kept = claim.get("kept")
                kept_s = "kept" if kept is True else ("discarded" if kept is False else "kept=?")
                qids = claim.get("injected_question_ids") or []
                inj = f" · injected on {', '.join(f'`{q}`' for q in qids)}" if qids else ""
                lines.append(f"- {claim.get('text_preview') or ''} · {kept_s}{inj}")
            lines.append("")
        extra = len(writers) - ATTRIBUTION_MD_CALL_CAP
        if extra > 0:
            lines.append(f"- … {extra} more writer calls in `attribution.jsonl`")
            lines.append("")
    lines.extend(["## Read-path calls (reader)", ""])
    readers = [c for c in calls if c.get("role") == "reader"]
    if not readers:
        lines.extend(["(no reader LLM calls this run)", ""])
    else:
        for call in readers[:ATTRIBUTION_MD_CALL_CAP]:
            lines.append(
                f"### reader · `{call.get('model')}` · question `{call.get('question_id')}`"
            )
            lines.append("")
            for claim in call.get("claims") or []:
                lines.append(f"- predicted: {claim.get('text_preview') or '—'}")
            used = call.get("used_memory_items") or []
            if used:
                for item in used[:8]:
                    writers = ",".join(item.get("proposed_by") or []) or "—"
                    lines.append(
                        f"- used `{item.get('item_id')}` ({item.get('item_kind')}) "
                        f"proposed_by={writers} {item.get('text_preview') or ''}"
                    )
                if len(used) > 8:
                    lines.append(f"- … {len(used) - 8} more injected items in `attribution.jsonl`")
            else:
                lines.append("- used memory items: (none linked)")
            lines.append("")
        extra = len(readers) - ATTRIBUTION_MD_CALL_CAP
        if extra > 0:
            lines.append(f"- … {extra} more reader calls in `attribution.jsonl`")
            lines.append("")
    return "\n".join(lines)


def ingest_summary(ingest_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count MERGE ops (add_edge / invalidate / skip_dup / node) for SUMMARY.md."""
    counts: dict[str, int] = defaultdict(int)
    counts["n_sessions"] = len(ingest_rows)
    for row in ingest_rows:
        for op in row.get("ops") or []:
            counts[str(op.get("op") or "unknown")] += 1
    return dict(counts)


def render_summary_md(
    *,
    run_id: str,
    meta: dict[str, Any],
    metrics: dict[str, Any],
    cost: dict[str, Any],
    quality: dict[str, Any] | None,
    ingest_rows: list[dict[str, Any]],
    retrieve_ranks: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    n_writer_calls: int,
    n_predictions: int,
    attribution_roles: list[dict[str, Any]] | None = None,
) -> str:
    """Human-first claim audit for ``SUMMARY.md``. Pointers, not a second metrics paper.

    Includes a short LLM-roles table; the full call → claims join is
    ``ATTRIBUTION.md``.
    """
    locomo = (metrics.get("metrics") or metrics or {}).get("locomo_f1")
    overall = metrics.get("metrics") or {}
    quality = quality or {}
    by_writer = quality.get("by_writer") or {}
    ingest = ingest_summary(ingest_rows)
    total = cost.get("total") or {}
    lines = [
        f"# Run audit: `{run_id}`",
        "",
        "This folder is an **audit of claims** (what entered `{memory}` and who",
        "proposed it), not only an audit of API calls.",
        "",
        "## Sandwich",
        "",
        f"- **memory:** `{meta.get('memory_type')}`",
        f"- **reader:** `{meta.get('reader_provider')}/{meta.get('reader_model')}` "
        f"(prompt `{meta.get('prompt_version')}`)",
        (
            f"- **agent:** `{meta.get('agent')}` persist={meta.get('agent_persist')} "
            f"tools={meta.get('agent_tools')}"
            if meta.get("agent")
            else "- **agent:** (none; one-shot reader)"
        ),
        f"- **writer:** `{meta.get('writer_provider')}/{meta.get('writer_model')}`"
        if meta.get("writer_model")
        else "- **writer:** (none)",
        f"- **n questions:** {n_predictions}",
        f"- **LoCoMo F1:** {locomo}",
        f"- **token F1 / EM:** {overall.get('token_f1')} / {overall.get('exact_match')}",
        "",
        "## LLM roles",
        "",
        "- `ATTRIBUTION.md` — each LLM call, the role it played, and the claims it made",
        "- `attribution.jsonl` — machine join (call → role → claims)",
        "",
    ]
    if attribution_roles:
        lines.extend(
            [
                "| role | layer | model | calls | claims |",
                "|---|---|---|---:|---:|",
            ]
        )
        for rec in attribution_roles:
            lines.append(
                f"| `{rec.get('role')}` | {rec.get('layer')} | `{rec.get('model')}` | "
                f"{rec.get('n_calls')} | {rec.get('n_claims')} |"
            )
        lines.append("")
    contract = meta.get("comparison_contract") or {}
    if contract:
        lines.extend(
            [
                "## Agent comparison controls",
                "",
                f"- status: **{contract.get('status')}** · contract `{contract.get('sha256')}`",
                f"- backbone: `{(contract.get('backbone') or {}).get('adapter')}` / "
                f"`{(contract.get('backbone') or {}).get('model')}` snapshot="
                f"`{(contract.get('backbone') or {}).get('model_snapshot')}`",
                f"- prompt sha256: `{(contract.get('task_prompt') or {}).get('sha256')}`",
                f"- context workspace sha256: `{(contract.get('context') or {}).get('workspace_manifest_sha256')}`",
                f"- retrieval: `{contract.get('retrieval')}`",
                f"- memory write: `{contract.get('memory_write')}`",
                f"- judge: `{contract.get('judge')}`",
                f"- tool budget: `{contract.get('tool_budget')}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Frozen config",
        "",
        "- `config.source.yaml` — YAML file used for this run",
        "- `config.resolved.yaml` — YAML plus CLI overrides that actually ran",
        f"- git `{meta.get('code_git_hash')}` · data sha256 `{str(meta.get('data_sha256') or '')[:12]}…`",
        "",
        "## Cost rollup",
        "",
        f"- reader tokens: {cost.get('reader', {}).get('total_tokens')} "
        f"(usd={cost.get('reader', {}).get('usd')}; "
        f"reasoning={cost.get('reader', {}).get('reasoning_tokens')})",
        f"- writer tokens: {cost.get('writer', {}).get('total_tokens')} "
        f"(usd={cost.get('writer', {}).get('usd')}; "
        f"reasoning={cost.get('writer', {}).get('reasoning_tokens')})",
        f"- **total tokens:** {total.get('total_tokens')} · **usd:** {total.get('usd')} "
        f"· reasoning={total.get('reasoning_tokens')}",
        f"- pricing pins as of {cost.get('pricing_as_of')}; unknown models are token-only",
        "",
        ]
    )
    if by_writer:
        lines.extend(
            [
                "## Writer quality",
                "",
                "| writer | calls | parse ok | entities | relations |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for tid, rec in sorted(by_writer.items()):
            lines.append(
                f"| `{tid}` | {rec.get('n_calls')} | {rec.get('n_parse_ok')} | "
                f"{rec.get('n_entities')} | {rec.get('n_relations')} |"
            )
        lines.append("")
    else:
        lines.extend(["## Writer quality", "", "(no writer calls this run)", ""])

    if ingest_rows:
        lines.extend(
            [
                "## Graph ingest",
                "",
                f"- sessions ingested: {ingest.get('n_sessions')}",
                f"- add_edge: {ingest.get('add_edge', 0)} · invalidate: {ingest.get('invalidate', 0)} · "
                f"skip_dup: {ingest.get('skip_dup', 0)}",
                f"- new_node: {ingest.get('new_node', 0)} · reuse_node: {ingest.get('reuse_node', 0)}",
                "- details: `memory/graph/ingest.jsonl`",
                "",
            ]
        )
    if retrieve_ranks:
        n_cand = sum(int(r.get("n_candidates") or 0) for r in retrieve_ranks)
        n_sel = sum(int(r.get("n_selected") or 0) for r in retrieve_ranks)
        lines.extend(
            [
                "## Retrieve ranks",
                "",
                f"- questions with ranks: {len(retrieve_ranks)} · "
                f"candidates {n_cand} · selected (injected) {n_sel}",
                "- full lists (losers included): `memory/retrieve_ranks.jsonl`",
                "",
            ]
        )
    lines.extend(
        [
            "## How to audit one claim",
            "",
            "1. `ATTRIBUTION.md` / `attribution.jsonl` — LLM call → role → claims made.",
            "2. Pick a `question_id` in `predictions.jsonl` (or `SUMMARY` lineage sample below).",
            "3. `memory/lineage.jsonl` — which memory items were injected and which writer proposed them.",
            "4. `memory/writer/sessions/` — the session text the writer saw.",
            "5. `memory/graph/ingest.jsonl` — MERGE / invalidate (graph conditions).",
            "6. `memory/retrieve_ranks.jsonl` — candidates that lost to the injected winners.",
            "",
            f"- writer calls this run: {n_writer_calls}",
            f"- lineage rows (injected items): {len(lineage)}",
            "",
        ]
    )
    sample = lineage[:8]
    if sample:
        lines.extend(["## Lineage sample (first injected items)", ""])
        for row in sample:
            writers = ",".join(row.get("proposed_by") or []) or "—"
            lines.append(
                f"- `{row.get('question_id')}` ← `{row.get('item_id')}` "
                f"({row.get('item_kind')}) proposed_by={writers} "
                f"{row.get('text_preview') or ''}"
            )
        lines.append("")
    return "\n".join(lines)
