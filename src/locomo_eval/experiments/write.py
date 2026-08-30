"""Write experiments/<run_id>/ module folders. Used by run.py, not by eval."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..report import write_json, write_jsonl
from .layout import AUDIT_LAYOUT_VERSION, PackPaths, teacher_dir_name


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
    paths = PackPaths.from_run_dir(run_dir)
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
) -> Path | None:
    """Write experiments/<run_id>/memory/teachers/ plus compat teacher_calls.jsonl."""
    if not calls and not fusion_rows:
        return None
    paths = PackPaths.from_run_dir(run_dir)
    paths.teachers_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.teacher_calls, calls)
    write_jsonl(paths.teacher_calls_compat, calls)

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
                "3. `fusion.jsonl` — `proposed_by` / `kept` for each graph triple",
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
    paths = PackPaths.from_run_dir(run_dir)
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
            "notes": "Fused Mem0g snapshot after pool/fusion. Embeddings omitted.",
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
