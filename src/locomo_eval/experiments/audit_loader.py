"""Load a finished sandwich audit from ``experiments/<run_id>/``.

No LLM, no teachers, no ``run.py``. Compare scripts and eval branches should
import this module (and ``audit_layout``) rather than ``audit_writer``,
``run``, or ``teachers``. Missing optional layers (teachers, autorater)
return [].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit_layout import AuditPaths

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def predictions_jsonl(run_dir: str | Path) -> Path | None:
    """Prefer run-root predictions.jsonl, then reader/predictions.jsonl."""
    paths = AuditPaths.from_run_dir(run_dir)
    if paths.predictions_root.is_file():
        return paths.predictions_root
    if paths.reader_predictions.is_file():
        return paths.reader_predictions
    return None


def resolve_predictions_jsonl(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve a run dir, run id, or JSONL file to a predictions.jsonl.

    Same search rules as the compare CLI (cwd + repo-root experiments/).
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    raw = Path(path)
    candidates: list[Path] = []

    def _add(item: Path) -> None:
        resolved = item.resolve() if item.exists() else item
        if resolved not in candidates:
            candidates.append(resolved)

    _add(raw)
    if not raw.is_absolute():
        _add(root / raw)
        _add(root / "experiments" / raw)
        _add(root / "experiments" / raw.name)

    existing_dirs = [c for c in candidates if c.is_dir()]
    for directory in existing_dirs:
        found = predictions_jsonl(directory)
        if found is not None:
            return found
        raise FileNotFoundError(
            f"Run directory {directory} has no predictions.jsonl. "
            f"Finish the pipeline for that --run-id first."
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(c) for c in candidates)
    hint_id = raw.name if raw.suffix != ".jsonl" else raw.stem
    raise FileNotFoundError(
        f"No predictions found for {path!s}. Tried: {tried}. "
        f"Create a run pack first, for example:\n"
        f"  python -m src.locomo_eval.run --config configs/raw_chunks.yaml "
        f"--reader mock --max-questions 5 --run-id {hint_id}"
    )


def load_qa_pack(run_dir: str | Path) -> dict[str, Any]:
    """QA/reader slice of a sandwich audit (stable keys for compare_full_runs)."""
    paths = AuditPaths.from_run_dir(run_dir)
    if not paths.metrics.is_file():
        raise FileNotFoundError(f"Missing {paths.metrics}")
    metrics = load_json(paths.metrics)
    meta = load_json(paths.run_meta)
    pred_path = predictions_jsonl(paths.run_dir)
    preds = load_jsonl(pred_path) if pred_path is not None else []
    by_qid = {row["question_id"]: row for row in preds if row.get("question_id")}
    return {
        "dir": str(paths.run_dir),
        "run_id": meta.get("run_id") or paths.run_dir.name,
        "metrics": metrics,
        "meta": meta,
        "predictions": preds,
        "by_qid": by_qid,
    }


@dataclass
class SandwichAudit:
    """One sandwich run on disk: reader, optional teachers/graph, optional judge.

    Optional writer/judge files are empty lists when that layer was not run.
    """

    paths: AuditPaths
    meta: dict[str, Any]
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]
    reader_traces: list[dict[str, Any]] = field(default_factory=list)
    teacher_index: list[dict[str, Any]] = field(default_factory=list)
    teacher_calls: list[dict[str, Any]] = field(default_factory=list)
    fusion: list[dict[str, Any]] = field(default_factory=list)
    lineage: list[dict[str, Any]] = field(default_factory=list)
    retrieve_ranks: list[dict[str, Any]] = field(default_factory=list)
    graph_ingest: list[dict[str, Any]] = field(default_factory=list)
    teacher_quality: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    autorater_verdicts: list[dict[str, Any]] = field(default_factory=list)
    autorater_traces: list[dict[str, Any]] = field(default_factory=list)

    def teacher_calls_for(self, teacher_id: str) -> list[dict[str, Any]]:
        wanted = str(teacher_id)
        rows = [row for row in self.teacher_calls if str(row.get("teacher_id")) == wanted]
        if rows:
            return rows
        return load_jsonl(self.paths.teacher_calls_path(wanted))

    def fusion_kept_for(self, *, sample_id: str | None = None) -> list[dict[str, Any]]:
        """Flatten fusion.jsonl to kept triples, optionally one sample."""
        out: list[dict[str, Any]] = []
        for session in self.fusion:
            if sample_id is not None and str(session.get("sample_id")) != str(sample_id):
                continue
            for rel in session.get("relations") or []:
                if not rel.get("kept"):
                    continue
                out.append(
                    {
                        "sample_id": session.get("sample_id"),
                        "session_id": session.get("session_id"),
                        **rel,
                    }
                )
        return out

    def lineage_for(self, *, question_id: str | None = None) -> list[dict[str, Any]]:
        if question_id is None:
            return list(self.lineage)
        wanted = str(question_id)
        return [row for row in self.lineage if str(row.get("question_id")) == wanted]

    def retrieve_ranks_for(self, *, question_id: str) -> dict[str, Any] | None:
        wanted = str(question_id)
        for row in self.retrieve_ranks:
            if str(row.get("question_id")) == wanted:
                return row
        return None


def load_sandwich_audit(run_dir: str | Path) -> SandwichAudit:
    """Load reader + optional teacher/graph/autorater traces for analysis."""
    paths = AuditPaths.from_run_dir(run_dir)
    pred_path = predictions_jsonl(paths.run_dir)
    calls = load_jsonl(paths.teacher_calls)
    if not calls:
        calls = load_jsonl(paths.teacher_calls_compat)
    return SandwichAudit(
        paths=paths,
        meta=load_json(paths.run_meta),
        metrics=load_json(paths.metrics),
        predictions=load_jsonl(pred_path) if pred_path is not None else [],
        reader_traces=load_jsonl(paths.reader_traces),
        teacher_index=load_jsonl(paths.teacher_index),
        teacher_calls=calls,
        fusion=load_jsonl(paths.teacher_fusion),
        lineage=load_jsonl(paths.lineage),
        retrieve_ranks=load_jsonl(paths.retrieve_ranks),
        graph_ingest=load_jsonl(paths.graph_ingest),
        teacher_quality=load_json(paths.teacher_quality),
        cost=load_json(paths.cost),
        autorater_verdicts=load_jsonl(paths.autorater_verdicts),
        autorater_traces=load_jsonl(paths.autorater_traces),
    )
