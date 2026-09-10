"""Path contract for one sandwich run: ``experiments/<run_id>/``.

This module names folders and files. It does not read or write them.

A sandwich run freezes data + reader/eval and varies memory. The on-disk
tree is the **audit** of those layers (reader, teachers, memory graph, judge)
so later analysis can map a condition to its results without re-running LLMs.

``AuditPaths`` is the shared map so the dump side (``audit_writer``) and the
analysis side (``audit_loader``) never invent different filenames.

Do not import teachers, readers, or ``run.py`` from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

AUDIT_LAYOUT_VERSION = "audit_pack.v2"

READER = "reader"
MEMORY = "memory"
TEACHERS = "memory/teachers"
GRAPH = "memory/graph"
AUTORATER = "autorater"
COMPAT_PREDICTIONS = "predictions.jsonl"
COMPAT_TEACHER_CALLS = "memory/teacher_calls.jsonl"
LINEAGE = "memory/lineage.jsonl"
RETRIEVE_RANKS = "memory/retrieve_ranks.jsonl"
GRAPH_INGEST = "memory/graph/ingest.jsonl"
TEACHER_QUALITY = "memory/teachers/quality.json"
TEACHER_SESSIONS = "memory/teachers/sessions.jsonl"
COST = "cost.json"
SUMMARY = "SUMMARY.md"
CONFIG_SOURCE = "config.source.yaml"
CONFIG_RESOLVED = "config.resolved.yaml"


def audit_layout_meta() -> dict[str, str]:
    """Value stored on run_meta.json['audit_layout']."""
    return {
        "version": AUDIT_LAYOUT_VERSION,
        "reader": f"{READER}/",
        "memory": f"{MEMORY}/",
        "teachers": f"{TEACHERS}/",
        "graph": f"{GRAPH}/",
        "autorater": f"{AUTORATER}/",
        "compat_predictions": COMPAT_PREDICTIONS,
        "lineage": LINEAGE,
        "retrieve_ranks": RETRIEVE_RANKS,
        "graph_ingest": GRAPH_INGEST,
        "teacher_quality": TEACHER_QUALITY,
        "teacher_sessions": TEACHER_SESSIONS,
        "cost": COST,
        "summary": SUMMARY,
        "config_source": CONFIG_SOURCE,
        "config_resolved": CONFIG_RESOLVED,
    }


def teacher_dir_name(teacher_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(teacher_id))
    return safe or "teacher"


@dataclass(frozen=True)
class AuditPaths:
    """Absolute paths for one sandwich-run directory. Missing files are still listed."""

    run_dir: Path
    run_meta: Path
    metrics: Path
    predictions_root: Path
    reader_dir: Path
    reader_predictions: Path
    reader_traces: Path
    reader_metrics: Path
    memory_dir: Path
    teachers_dir: Path
    teacher_index: Path
    teacher_calls: Path
    teacher_fusion: Path
    teacher_calls_compat: Path
    teacher_quality: Path
    teacher_sessions_index: Path
    graph_dir: Path
    graph_index: Path
    graph_ingest: Path
    lineage: Path
    retrieve_ranks: Path
    autorater_dir: Path
    autorater_verdicts: Path
    autorater_traces: Path
    cost: Path
    summary: Path
    config_source: Path
    config_resolved: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> AuditPaths:
        root = Path(run_dir)
        reader = root / READER
        teachers = root / TEACHERS
        graph = root / GRAPH
        autorater = root / AUTORATER
        return cls(
            run_dir=root,
            run_meta=root / "run_meta.json",
            metrics=root / "metrics.json",
            predictions_root=root / COMPAT_PREDICTIONS,
            reader_dir=reader,
            reader_predictions=reader / "predictions.jsonl",
            reader_traces=reader / "traces.jsonl",
            reader_metrics=reader / "metrics.json",
            memory_dir=root / MEMORY,
            teachers_dir=teachers,
            teacher_index=teachers / "index.jsonl",
            teacher_calls=teachers / "calls.jsonl",
            teacher_fusion=teachers / "fusion.jsonl",
            teacher_calls_compat=root / COMPAT_TEACHER_CALLS,
            teacher_quality=root / TEACHER_QUALITY,
            teacher_sessions_index=root / TEACHER_SESSIONS,
            graph_dir=graph,
            graph_index=graph / "index.jsonl",
            graph_ingest=root / GRAPH_INGEST,
            lineage=root / LINEAGE,
            retrieve_ranks=root / RETRIEVE_RANKS,
            autorater_dir=autorater,
            autorater_verdicts=autorater / "autorater_verdicts.jsonl",
            autorater_traces=autorater / "traces.jsonl",
            cost=root / COST,
            summary=root / SUMMARY,
            config_source=root / CONFIG_SOURCE,
            config_resolved=root / CONFIG_RESOLVED,
        )

    def teacher_calls_path(self, teacher_id: str) -> Path:
        return self.teachers_dir / "by_teacher" / teacher_dir_name(teacher_id) / "calls.jsonl"

    def graph_sample_path(self, sample_id: str) -> Path:
        return self.graph_dir / "by_sample" / f"{sample_id}.json"

    def teacher_session_text_path(self, sample_id: str, session_id: str | int) -> Path:
        return (
            self.teachers_dir
            / "sessions"
            / "by_sample"
            / str(sample_id)
            / f"session_{session_id}.txt"
        )
