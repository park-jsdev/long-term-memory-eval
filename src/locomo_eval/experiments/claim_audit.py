"""Claim-level sandwich audit: lineage, ranks, teacher quality, cost, SUMMARY.

Call traces (reader/teachers JSONL) record that an LLM ran. These helpers
record **what entered {memory}** and who is responsible for each item, so a
reviewer can audit a QA claim without re-running models.

Pure functions. ``audit_writer`` dumps the dicts; ``run.py`` collects inputs.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

PREVIEW_CHARS = 200

# USD per 1M tokens. Rollup pins, not invoices. Unknown models stay token-only.
PRICING_AS_OF = "2026-09-10"
USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def preview(text: str, n: int = PREVIEW_CHARS) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= n:
        return compact
    return compact[:n] + "…"


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int, int]:
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return prompt, completion, total


def estimate_usd(model: str | None, prompt_tokens: int, completion_tokens: int) -> float | None:
    if not model:
        return None
    rates = USD_PER_MILLION.get(str(model).strip())
    if rates is None:
        return None
    prompt_rate, completion_rate = rates
    return round(
        (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000.0,
        6,
    )


def add_costs(*values: float | None) -> float | None:
    known = [v for v in values if v is not None]
    if not known:
        return None
    return round(sum(known), 6)


def ranked_candidates(
    items: list[dict[str, Any]],
    *,
    selected_ids: set[str],
    id_key: str = "item_id",
) -> list[dict[str, Any]]:
    """Attach 1-based rank and selected flag. ``items`` must already be score-sorted."""
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


def teacher_quality_stats(
    calls: list[dict[str, Any]],
    fusion_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse/yield/keep rates per teacher. Software stats, not an LLM judge."""
    by_teacher: dict[str, dict[str, Any]] = {}
    for call in calls:
        tid = str(call.get("teacher_id") or "teacher")
        rec = by_teacher.setdefault(
            tid,
            {
                "teacher_id": tid,
                "provider": call.get("provider"),
                "model": call.get("model"),
                "n_calls": 0,
                "n_parse_ok": 0,
                "n_parse_fallback": 0,
                "n_parse_unset": 0,
                "n_entities": 0,
                "n_relations": 0,
                "n_proposed_kept": 0,
                "latency_sum_s": 0.0,
                "n_latency": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
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

    kept_by_teacher: dict[str, int] = defaultdict(int)
    n_proposed = 0
    n_kept = 0
    n_unanimous = 0
    n_discarded = 0
    n_sessions = len(fusion_rows)
    for session in fusion_rows:
        n_teachers = max(1, len(session.get("teacher_ids") or []))
        for rel in session.get("relations") or []:
            n_proposed += 1
            proposers = [str(t) for t in (rel.get("proposed_by") or [])]
            if rel.get("kept"):
                n_kept += 1
                if len(proposers) >= n_teachers and n_teachers > 1:
                    n_unanimous += 1
                for tid in proposers:
                    kept_by_teacher[tid] += 1
            else:
                n_discarded += 1

    for tid, rec in by_teacher.items():
        rec["n_proposed_kept"] = int(kept_by_teacher.get(tid, 0))
        rec["keep_rate"] = (
            round(rec["n_proposed_kept"] / rec["n_relations"], 4)
            if rec["n_relations"]
            else None
        )
        rec["mean_latency_s"] = (
            round(rec["latency_sum_s"] / rec["n_latency"], 4) if rec["n_latency"] else None
        )
        rec["usd"] = estimate_usd(rec.get("model"), rec["prompt_tokens"], rec["completion_tokens"])
        rec.pop("latency_sum_s", None)
        rec.pop("n_latency", None)

    return {
        "by_teacher": by_teacher,
        "fusion": {
            "n_sessions": n_sessions,
            "n_triples_proposed": n_proposed,
            "n_triples_kept": n_kept,
            "n_triples_discarded": n_discarded,
            "n_unanimous_kept": n_unanimous,
            "keep_rate": round(n_kept / n_proposed, 4) if n_proposed else None,
        },
    }


def cost_rollup(
    *,
    reader_traces: list[dict[str, Any]],
    teacher_calls: list[dict[str, Any]],
    extra_usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Token + optional USD totals. Mock traces with empty usage cost $0."""

    def _bucket(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
        prompt = completion = total = 0
        usd: float | None = 0.0
        any_priced = False
        by_model: dict[str, dict[str, Any]] = {}
        for row in rows:
            model = str(row.get("model") or "")
            p, c, t = _usage_tokens(row.get("usage"))
            prompt += p
            completion += c
            total += t
            rec = by_model.setdefault(
                model or "(unknown)",
                {
                    "model": model or None,
                    "role": role,
                    "n_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "usd": None,
                },
            )
            rec["n_calls"] += 1
            rec["prompt_tokens"] += p
            rec["completion_tokens"] += c
            rec["total_tokens"] += t
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
            "usd": usd,
            "by_model": by_model,
        }

    reader = _bucket(reader_traces, "reader")
    teacher = _bucket(teacher_calls, "teacher")
    extra = _bucket(extra_usage or [], "other")
    total_usd = add_costs(reader["usd"], teacher["usd"], extra["usd"])
    if reader["usd"] is None or teacher["usd"] is None:
        if (reader["prompt_tokens"] + teacher["prompt_tokens"]) == 0:
            total_usd = 0.0
        elif reader["usd"] is None and teacher["usd"] is None and extra["usd"] is None:
            total_usd = None
    return {
        "pricing_as_of": PRICING_AS_OF,
        "pricing_note": (
            "USD uses pinned list prices for known OpenAI ids only. "
            "Unknown models contribute tokens but usd=null. Not an invoice."
        ),
        "reader": reader,
        "teacher": teacher,
        "other": extra,
        "total": {
            "prompt_tokens": reader["prompt_tokens"] + teacher["prompt_tokens"] + extra["prompt_tokens"],
            "completion_tokens": (
                reader["completion_tokens"] + teacher["completion_tokens"] + extra["completion_tokens"]
            ),
            "total_tokens": reader["total_tokens"] + teacher["total_tokens"] + extra["total_tokens"],
            "usd": total_usd,
        },
    }


def _fusion_by_triple(
    fusion_rows: list[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for session in fusion_rows:
        sample_id = session.get("sample_id")
        session_id = session.get("session_id")
        for rel in session.get("relations") or []:
            key = (
                str(sample_id),
                session_id,
                str(rel.get("source") or ""),
                str(rel.get("relationship") or ""),
                str(rel.get("target") or ""),
            )
            out[key] = rel
    return out


def graph_edge_provenance(
    ingest_rows: list[dict[str, Any]],
    fusion_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """edge_id → ingest session + fusion proposed_by (claim provenance)."""
    fusion = _fusion_by_triple(fusion_rows)
    by_edge: dict[str, dict[str, Any]] = {}
    for session in ingest_rows:
        sample_id = session.get("sample_id")
        session_id = session.get("session_id")
        for op in session.get("ops") or []:
            if op.get("op") != "add_edge":
                continue
            edge_id = str(op.get("edge_id") or "")
            if not edge_id:
                continue
            key = (
                str(sample_id),
                session_id,
                str(op.get("source") or ""),
                str(op.get("relationship") or ""),
                str(op.get("target") or ""),
            )
            rel = fusion.get(key) or {}
            by_edge[edge_id] = {
                "item_id": edge_id,
                "item_kind": "graph_edge",
                "sample_id": sample_id,
                "session_id": session_id,
                "source": op.get("source"),
                "relationship": op.get("relationship"),
                "target": op.get("target"),
                "proposed_by": list(rel.get("proposed_by") or []),
                "votes": rel.get("votes"),
                "kept": rel.get("kept", True),
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
    if memories_by_question and question_id in memories_by_question:
        return memories_by_question[question_id]
    if memories_by_sample and sample_id in memories_by_sample:
        return memories_by_sample[sample_id]
    return None


def _teacher_for_session(
    calls: list[dict[str, Any]],
    *,
    sample_id: str,
    session_id: Any,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for call in calls:
        if str(call.get("sample_id")) != str(sample_id):
            continue
        if call.get("session_id") != session_id and str(call.get("session_id")) != str(session_id):
            continue
        tid = str(call.get("teacher_id") or "")
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
    fusion_rows: list[dict[str, Any]] | None = None,
    teacher_calls: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One row per injected memory item: question → item → teacher(s)."""
    ranks_by_qid = {
        str(row.get("question_id")): row for row in (retrieve_ranks or []) if row.get("question_id")
    }
    edge_prov = graph_edge_provenance(ingest_rows or [], fusion_rows or [])
    calls = teacher_calls or []
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
        rank_row = ranks_by_qid.get(qid)
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
        if source_ids and any(sid in edge_prov for sid in source_ids):
            for item_id in source_ids:
                prov = edge_prov.get(item_id)
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
            if str(item_id).startswith("session_") and str(item_id).endswith("_teacher"):
                kind = "teacher_summary"
                try:
                    session_id = int(str(item_id).split("_")[1])
                except (IndexError, ValueError):
                    session_id = None
            elif str(item_id).endswith("_summary"):
                kind = "session_summary"
            proposers = (
                _teacher_for_session(calls, sample_id=sample_id, session_id=session_id)
                if kind == "teacher_summary"
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


def ingest_summary(ingest_rows: list[dict[str, Any]]) -> dict[str, int]:
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
    n_teacher_calls: int,
    n_predictions: int,
) -> str:
    """Human-first claim audit. Pointers, not a second metrics paper."""
    locomo = (metrics.get("metrics") or metrics or {}).get("locomo_f1")
    overall = metrics.get("metrics") or {}
    quality = quality or {}
    by_teacher = quality.get("by_teacher") or {}
    fusion = quality.get("fusion") or {}
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
        f"- **teacher:** `{meta.get('teacher_provider')}/{meta.get('teacher_model')}`"
        if meta.get("teacher_model")
        else "- **teacher:** (none)",
        f"- **pool / fusion:** `{meta.get('teacher_pool')}` / `{meta.get('teacher_fusion')}`"
        if meta.get("teacher_pool")
        else "- **pool / fusion:** (n/a)",
        f"- **n questions:** {n_predictions}",
        f"- **LoCoMo F1:** {locomo}",
        f"- **token F1 / EM:** {overall.get('token_f1')} / {overall.get('exact_match')}",
        "",
        "## Frozen config",
        "",
        "- `config.source.yaml` — YAML file used for this run",
        "- `config.resolved.yaml` — YAML plus CLI overrides that actually ran",
        f"- git `{meta.get('code_git_hash')}` · data sha256 `{str(meta.get('data_sha256') or '')[:12]}…`",
        "",
        "## Cost rollup",
        "",
        f"- reader tokens: {cost.get('reader', {}).get('total_tokens')} "
        f"(usd={cost.get('reader', {}).get('usd')})",
        f"- teacher tokens: {cost.get('teacher', {}).get('total_tokens')} "
        f"(usd={cost.get('teacher', {}).get('usd')})",
        f"- **total tokens:** {total.get('total_tokens')} · **usd:** {total.get('usd')}",
        f"- pricing pins as of {cost.get('pricing_as_of')}; unknown models are token-only",
        "",
    ]
    if by_teacher:
        lines.extend(
            [
                "## Teacher quality",
                "",
                "| teacher | calls | parse ok | entities | relations | kept | keep rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tid, rec in sorted(by_teacher.items()):
            lines.append(
                f"| `{tid}` | {rec.get('n_calls')} | {rec.get('n_parse_ok')} | "
                f"{rec.get('n_entities')} | {rec.get('n_relations')} | "
                f"{rec.get('n_proposed_kept')} | {rec.get('keep_rate')} |"
            )
        lines.extend(
            [
                "",
                f"- fusion sessions: {fusion.get('n_sessions')} · "
                f"proposed {fusion.get('n_triples_proposed')} · "
                f"kept {fusion.get('n_triples_kept')} · "
                f"discarded {fusion.get('n_triples_discarded')} · "
                f"unanimous kept {fusion.get('n_unanimous_kept')}",
                "",
            ]
        )
    else:
        lines.extend(["## Teacher quality", "", "(no teacher calls this run)", ""])

    if ingest_rows:
        lines.extend(
            [
                "## Graph ingest after fusion",
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
            "1. Pick a `question_id` in `predictions.jsonl` (or `SUMMARY` lineage sample below).",
            "2. `memory/lineage.jsonl` — which memory items were injected and which teacher proposed them.",
            "3. `memory/teachers/sessions/` — the session text that teacher actually saw.",
            "4. `memory/teachers/fusion.jsonl` — `proposed_by` / `kept` for that triple.",
            "5. `memory/graph/ingest.jsonl` — MERGE / invalidate after fusion (graph conditions).",
            "6. `memory/retrieve_ranks.jsonl` — candidates that lost to the injected winners.",
            "",
            f"- teacher calls this run: {n_teacher_calls}",
            f"- lineage rows (injected items): {len(lineage)}",
            "",
        ]
    )
    sample = lineage[:8]
    if sample:
        lines.extend(["## Lineage sample (first injected items)", ""])
        for row in sample:
            teachers = ",".join(row.get("proposed_by") or []) or "—"
            lines.append(
                f"- `{row.get('question_id')}` ← `{row.get('item_id')}` "
                f"({row.get('item_kind')}) proposed_by={teachers} "
                f"{row.get('text_preview') or ''}"
            )
        lines.append("")
    return "\n".join(lines)
