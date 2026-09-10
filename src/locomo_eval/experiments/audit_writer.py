"""Dump sandwich-layer folders into ``experiments/<run_id>/`` during a run.

Called from ``run.py`` (reader/QA), ``memory_log.py`` (teachers), and the
autorater CLI (judge traces). Analysis code should not import this module —
use ``audit_loader`` on a finished directory instead.

Writes (each optional except reader on a completed QA run):

- ``reader/`` — frozen answer-LLM traces + predictions
- ``memory/teachers/`` — write-path teacher calls + fusion audit + session text
- ``memory/graph/`` — fused Mem0g snapshot + ingest ops after fusion
- ``memory/lineage.jsonl`` — question → injected item → teacher
- ``memory/retrieve_ranks.jsonl`` — full ranked candidates, not just winners
- ``config.*.yaml`` / ``cost.json`` / ``SUMMARY.md`` — frozen run + human audit
- ``autorater/traces.jsonl`` — judge reasoning next to verdicts
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..report import write_json, write_jsonl
from .audit_layout import AUDIT_LAYOUT_VERSION, AuditPaths, teacher_dir_name
from .claim_audit import (
    cost_rollup,
    lineage_rows,
    render_summary_md,
    sha256_text,
    teacher_quality_stats,
)


def audit_graph_dict(graph: Any, *, sample_id: str) -> dict[str, Any]:
    """Slim Mem0g snapshot (no embeddings) for claim audit."""
    nodes = getattr(graph, "nodes", {}) or {}
    edges = getattr(graph, "edges", []) or []
    node_rows = []
    for node in nodes.values() if hasattr(nodes, "values") else nodes:
        node_rows.append(
            {
                "node_id": getattr(node, "node_id", None),
                "name": getattr(node, "name", None),
                "entity_type": getattr(node, "entity_type", None),
            }
        )
    edge_rows = []
    n_valid = 0
    for edge in edges:
        valid = bool(getattr(edge, "valid", True))
        if valid:
            n_valid += 1
        edge_rows.append(
            {
                "edge_id": getattr(edge, "edge_id", None),
                "source": getattr(edge, "source", None),
                "relationship": getattr(edge, "relationship", None),
                "target": getattr(edge, "target", None),
                "valid": valid,
                "timestamp": getattr(edge, "timestamp", "") or "",
            }
        )
    return {
        "sample_id": sample_id,
        "n_nodes": len(node_rows),
        "n_edges": len(edge_rows),
        "n_valid_edges": n_valid,
        "nodes": node_rows,
        "edges": edge_rows,
    }


def write_reader_module(
    run_dir: Path,
    *,
    prediction_rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> Path:
    """Write experiments/<run_id>/reader/ (answer LLM traces + predictions)."""
    paths = AuditPaths.from_run_dir(run_dir)
    paths.reader_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.reader_predictions, prediction_rows)
    write_jsonl(paths.reader_traces, traces)
    if summary is not None:
        write_json(paths.reader_metrics, summary)
    write_json(
        paths.reader_dir / "schema.json",
        {
            "schema_version": AUDIT_LAYOUT_VERSION,
            "role": "reader",
            "notes": (
                "Frozen answer LLM. traces.jsonl is the Chat Completions audit "
                "(reasoning usually empty under reasoning_effort=none / gpt-4o-mini). "
                "predictions.jsonl is the LoCoMo QA pack (gold is scorer-only)."
            ),
            "files": {
                "predictions.jsonl": "one row per question (same as run-root copy)",
                "traces.jsonl": "one LLM call per question: output, reasoning, usage",
                "metrics.json": "LoCoMo / SPEC scores copied from the run",
            },
        },
    )
    (paths.reader_dir / "README.md").write_text(
        "\n".join(
            [
                "# Reader (answer LLM)",
                "",
                "Frozen sandwich bottom: `{memory}` + question → predicted answer.",
                "Gold answers are in `predictions.jsonl` for scoring only; they are",
                "not sent to the reader.",
                "",
                "- `traces.jsonl` — reasoning / usage per question",
                "- `predictions.jsonl` — LoCoMo QA rows (also copied at run root)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths.reader_dir


def write_teacher_module(
    run_dir: Path,
    *,
    calls: list[dict[str, Any]],
    fusion_rows: list[dict[str, Any]] | None = None,
    session_texts: list[dict[str, Any]] | None = None,
) -> Path | None:
    """Write experiments/<run_id>/memory/teachers/ plus compat teacher_calls.jsonl."""
    if not calls and not fusion_rows and not session_texts:
        return None
    paths = AuditPaths.from_run_dir(run_dir)
    paths.teachers_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.teacher_calls, calls)
    write_jsonl(paths.teacher_calls_compat, calls)
    write_teacher_sessions(run_dir, session_texts or [])

    index_rows = []
    by_teacher: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        tid = str(call.get("teacher_id") or "teacher")
        by_teacher[tid].append(call)
        reasoning = str(call.get("reasoning") or "")
        index_rows.append(
            {
                "sample_id": call.get("sample_id"),
                "session_id": call.get("session_id"),
                "session_index": call.get("session_index"),
                "teacher_id": tid,
                "provider": call.get("provider"),
                "model": call.get("model"),
                "role": call.get("role"),
                "thinking": call.get("thinking"),
                "thinking_supported": call.get("thinking_supported"),
                "n_entities": call.get("n_entities"),
                "n_relations": call.get("n_relations"),
                "n_reasoning_chars": len(reasoning),
                "parse": call.get("parse"),
                "session_text_sha256": call.get("session_text_sha256"),
                "n_session_chars": call.get("n_session_chars"),
                "path": f"memory/teachers/by_teacher/{teacher_dir_name(tid)}/calls.jsonl",
            }
        )
    write_jsonl(paths.teacher_index, index_rows)
    for tid, rows in sorted(by_teacher.items()):
        write_jsonl(paths.teacher_calls_path(tid), rows)

    if fusion_rows:
        write_jsonl(paths.teacher_fusion, fusion_rows)

    write_json(
        paths.teachers_dir / "schema.json",
        {
            "schema_version": AUDIT_LAYOUT_VERSION,
            "role": "teacher",
            "notes": (
                "Write-path LLM traces. Index by teacher_id + sample_id + session_id. "
                "fusion.jsonl records which teacher proposed each triple and whether "
                "pool/majority kept it. Gold never enters these prompts."
            ),
            "files": {
                "index.jsonl": "one row per teacher call (lookup keys, no long reasoning)",
                "calls.jsonl": "full call: reasoning, output_text, entities, relations",
                "by_teacher/<id>/calls.jsonl": "same rows partitioned by teacher_id",
                "fusion.jsonl": "per session: proposed_by[], votes, kept",
                "sessions/by_sample/<id>/session_<k>.txt": "exact teacher input text",
                "sessions.jsonl": "index of session texts (sha256, path)",
                "quality.json": "parse/yield/keep rates per teacher",
            },
        },
    )
    (paths.teachers_dir / "README.md").write_text(
        "\n".join(
            [
                "# Teachers (write-path LLMs)",
                "",
                "Attribute memory construction to a teacher:",
                "",
                "1. `index.jsonl` — which teacher was called for which session",
                "2. `by_teacher/<teacher_id>/calls.jsonl` — that model's reasoning + triples",
                "3. `sessions/by_sample/<id>/session_<k>.txt` — the session text the teacher saw",
                "4. `fusion.jsonl` — `proposed_by` / `kept` for each graph triple",
                "5. `quality.json` — parse / yield / keep-rate stats",
                "",
                "Compat copy: `memory/teacher_calls.jsonl` is the same as `calls.jsonl`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths.teachers_dir


def write_graph_module(run_dir: Path, graphs_by_sample: dict[str, Any]) -> Path | None:
    """Write experiments/<run_id>/memory/graph/by_sample/<id>.json."""
    if not graphs_by_sample:
        return None
    paths = AuditPaths.from_run_dir(run_dir)
    sample_dir = paths.graph_dir / "by_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for sample_id, graph in sorted(graphs_by_sample.items()):
        payload = audit_graph_dict(graph, sample_id=str(sample_id))
        rel = f"memory/graph/by_sample/{sample_id}.json"
        write_json(paths.graph_sample_path(str(sample_id)), payload)
        index_rows.append(
            {
                "sample_id": sample_id,
                "n_nodes": payload["n_nodes"],
                "n_edges": payload["n_edges"],
                "n_valid_edges": payload["n_valid_edges"],
                "full_graph_path": rel,
            }
        )
    write_jsonl(paths.graph_index, index_rows)
    write_json(
        paths.graph_dir / "schema.json",
        {
            "schema_version": AUDIT_LAYOUT_VERSION,
            "role": "memory_graph",
            "notes": (
                "Fused Mem0g snapshot after pool/fusion. Embeddings omitted. "
                "ingest.jsonl is the MERGE/invalidate trail (claim audit), not the LLM call log."
            ),
            "files": {
                "index.jsonl": "per-sample node/edge counts",
                "by_sample/<id>.json": "nodes + edges after all sessions",
                "ingest.jsonl": "per-session ops after fusion (add_edge, invalidate, skip_dup, node merge)",
            },
        },
    )
    return paths.graph_dir


def write_autorater_traces(out_dir: Path, scored: list[dict[str, Any]]) -> Path:
    """Write autorater/traces.jsonl (judge reasoning) next to verdicts."""
    traces = []
    for row in scored:
        traces.append(
            {
                "sample_id": row.get("sample_id"),
                "question_id": row.get("question_id"),
                "role": "autorater",
                "provider": row.get("autorater_provider") or row.get("provider"),
                "model": row.get("autorater_model") or row.get("model"),
                "label": row.get("label"),
                "skipped": row.get("skipped"),
                "reasoning": row.get("reasoning") or "",
                "raw_text": row.get("raw_text") or "",
                "latency_s": row.get("autorater_latency_s") or row.get("latency_s"),
                "usage": row.get("usage") or {},
            }
        )
    path = Path(out_dir) / "traces.jsonl"
    write_jsonl(path, traces)
    write_json(
        Path(out_dir) / "schema.json",
        {
            "schema_version": AUDIT_LAYOUT_VERSION,
            "role": "autorater",
            "notes": (
                "Judge LLM traces. Gold is visible to this module (unlike the reader). "
                "autorater_verdicts.jsonl keeps the full scored row; traces.jsonl is "
                "the LLM-call subset."
            ),
        },
    )
    return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


def write_frozen_config(
    run_dir: Path,
    *,
    cfg: dict[str, Any],
    overrides: Any,
    source_config: str | Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Copy the source YAML and dump YAML + CLI overrides that actually ran."""
    paths = AuditPaths.from_run_dir(run_dir)
    raw = source_config or getattr(overrides, "config", None)
    src: Path | None = None
    if raw:
        candidate = Path(str(raw))
        if candidate.is_file():
            src = candidate
        elif repo_root is not None and (Path(repo_root) / candidate).is_file():
            src = Path(repo_root) / candidate
    if src is not None:
        shutil.copy2(src, paths.config_source)
    cli = {}
    if overrides is not None:
        for key, value in vars(overrides).items():
            if value is None:
                continue
            cli[key] = _jsonable(value)
    payload = {
        "source_config": str(src) if src is not None else (str(raw) if raw else None),
        "cli_overrides": cli,
        "yaml": _jsonable(cfg),
    }
    paths.config_resolved.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return paths.config_resolved


def write_teacher_sessions(run_dir: Path, session_texts: list[dict[str, Any]]) -> Path | None:
    """Dump the exact session text each teacher saw (once per sample/session)."""
    if not session_texts:
        return None
    paths = AuditPaths.from_run_dir(run_dir)
    index_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in session_texts:
        sample_id = str(row.get("sample_id") or "")
        session_id = row.get("session_id")
        key = (sample_id, str(session_id))
        if key in seen:
            continue
        seen.add(key)
        text = str(row.get("text") or "")
        dest = paths.teacher_session_text_path(sample_id, session_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        rel = dest.relative_to(paths.run_dir).as_posix()
        index_rows.append(
            {
                "sample_id": sample_id,
                "session_id": session_id,
                "session_index": row.get("session_index"),
                "n_chars": len(text),
                "text_sha256": row.get("text_sha256") or sha256_text(text),
                "path": rel,
            }
        )
    write_jsonl(paths.teacher_sessions_index, index_rows)
    return paths.teacher_sessions_index


def write_graph_ingest(run_dir: Path, ingest_rows: list[dict[str, Any]]) -> Path | None:
    if not ingest_rows:
        return None
    paths = AuditPaths.from_run_dir(run_dir)
    paths.graph_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.graph_ingest, ingest_rows)
    return paths.graph_ingest


def write_claim_audit(
    run_dir: Path,
    *,
    prediction_rows: list[dict[str, Any]],
    reader_traces: list[dict[str, Any]],
    teacher_calls: list[dict[str, Any]],
    fusion_rows: list[dict[str, Any]],
    ingest_rows: list[dict[str, Any]] | None = None,
    retrieve_ranks: list[dict[str, Any]] | None = None,
    memories_by_sample: dict[str, Any] | None = None,
    memories_by_question: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Lineage, ranks, teacher quality, cost, and human SUMMARY.md."""
    paths = AuditPaths.from_run_dir(run_dir)
    ingest_rows = ingest_rows or []
    retrieve_ranks = retrieve_ranks or []
    written: dict[str, str] = {}

    if ingest_rows:
        write_graph_ingest(run_dir, ingest_rows)
        written["graph_ingest"] = str(paths.graph_ingest)
    if retrieve_ranks:
        write_jsonl(paths.retrieve_ranks, retrieve_ranks)
        written["retrieve_ranks"] = str(paths.retrieve_ranks)

    lineage = lineage_rows(
        prediction_rows=prediction_rows,
        memories_by_sample=memories_by_sample,
        memories_by_question=memories_by_question,
        retrieve_ranks=retrieve_ranks,
        ingest_rows=ingest_rows,
        fusion_rows=fusion_rows,
        teacher_calls=teacher_calls,
    )
    if lineage:
        write_jsonl(paths.lineage, lineage)
        written["lineage"] = str(paths.lineage)

    quality = teacher_quality_stats(teacher_calls, fusion_rows)
    if teacher_calls or fusion_rows:
        write_json(paths.teacher_quality, quality)
        written["teacher_quality"] = str(paths.teacher_quality)

    cost = cost_rollup(reader_traces=reader_traces, teacher_calls=teacher_calls)
    write_json(paths.cost, cost)
    written["cost"] = str(paths.cost)

    summary = render_summary_md(
        run_id=str((meta or {}).get("run_id") or paths.run_dir.name),
        meta=meta or {},
        metrics=metrics or {},
        cost=cost,
        quality=quality if (teacher_calls or fusion_rows) else None,
        ingest_rows=ingest_rows,
        retrieve_ranks=retrieve_ranks,
        lineage=lineage,
        n_teacher_calls=len(teacher_calls),
        n_predictions=len(prediction_rows),
    )
    paths.summary.write_text(summary, encoding="utf-8")
    written["summary"] = str(paths.summary)
    return written
