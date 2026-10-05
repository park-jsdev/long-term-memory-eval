"""Load a finished experiment pack from ``experiments/<run_id>/``.

The function name ``load_sandwich_audit`` is unchanged.

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
    """Read a JSON object, or ``{}`` when the optional layer was not dumped."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows, or ``[]`` when the optional layer was not dumped."""
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
    """Locate predictions inside an already-resolved run directory.

    Prefers the run-root copy, then ``reader/predictions.jsonl``, so compare
    scripts work on packs written before the reader/ split.
    """
    paths = AuditPaths.from_run_dir(run_dir)
    if paths.predictions_root.is_file():
        return paths.predictions_root
    if paths.reader_predictions.is_file():
        return paths.reader_predictions
    return None


def resolve_predictions_jsonl(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve a run dir, run id, or JSONL file to a predictions.jsonl.

    Same search rules as the compare CLI (cwd + repo-root ``experiments/``).
    Empty decoy directories are skipped so a leftover cwd folder cannot hide
    ``experiments/<run_id>``. Distinct from ``predictions_jsonl``, which
    assumes ``path`` is already a run directory.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    raw = Path(path)
    candidates: list[Path] = []

    def _add(item: Path) -> None:
        """Dedup search candidates; resolve only paths that already exist."""
        resolved = item.resolve() if item.exists() else item
        if resolved not in candidates:
            candidates.append(resolved)

    _add(raw)
    if not raw.is_absolute():
        _add(root / raw)
        _add(root / "experiments" / raw)
        _add(root / "experiments" / raw.name)

    existing_dirs = [c for c in candidates if c.is_dir()]
    empty_dirs: list[Path] = []
    for directory in existing_dirs:
        found = predictions_jsonl(directory)
        if found is not None:
            return found
        empty_dirs.append(directory)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if empty_dirs:
        directory = empty_dirs[0]
        raise FileNotFoundError(
            f"Run directory {directory} has no predictions.jsonl. "
            f"Finish the pipeline for that --run-id first."
        )

    tried = ", ".join(str(c) for c in candidates)
    hint_id = raw.name if raw.suffix != ".jsonl" else raw.stem
    raise FileNotFoundError(
        f"No predictions found for {path!s}. Tried: {tried}. "
        f"Create a run pack first, for example:\n"
        f"  python -m src.locomo_eval.run --config configs/writers/raw_chunks.yaml "
        f"--reader mock --max-questions 5 --run-id {hint_id}"
    )


def load_qa_pack(run_dir: str | Path) -> dict[str, Any]:
    """QA/reader slice only (metrics + predictions + run_meta).

    Return keys match ``scripts.compare_full_runs.load_pack`` so compare
    stays independent of teachers/lineage/attribution.
    """
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
    """One experiment pack on disk: reader, optional writer/graph, optional judge.

    Optional writer/judge files are empty lists when that layer was not run.
    """

    paths: AuditPaths
    meta: dict[str, Any]
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]
    reader_traces: list[dict[str, Any]] = field(default_factory=list)
    writer_index: list[dict[str, Any]] = field(default_factory=list)
    writer_calls: list[dict[str, Any]] = field(default_factory=list)
    lineage: list[dict[str, Any]] = field(default_factory=list)
    retrieve_ranks: list[dict[str, Any]] = field(default_factory=list)
    graph_ingest: list[dict[str, Any]] = field(default_factory=list)
    writer_quality: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    attribution: list[dict[str, Any]] = field(default_factory=list)
    autorater_verdicts: list[dict[str, Any]] = field(default_factory=list)
    autorater_traces: list[dict[str, Any]] = field(default_factory=list)

    def writer_calls_for(self, writer_id: str) -> list[dict[str, Any]]:
        """Calls for one writer_id.

        Prefers the in-memory ``calls.jsonl`` load; falls back to the
        ``by_writer/<id>/`` partition if the combined file is empty.
        """
        wanted = str(writer_id)
        rows = [row for row in self.writer_calls if str(row.get("writer_id")) == wanted]
        if rows:
            return rows
        return load_jsonl(self.paths.writer_calls_path(wanted))

    def lineage_for(
        self,
        *,
        question_id: str | None = None,
        sample_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Injected memory items for one question and/or sample (or the whole run)."""
        rows = list(self.lineage)
        if question_id is not None:
            wanted = str(question_id)
            rows = [row for row in rows if str(row.get("question_id")) == wanted]
        if sample_id is not None:
            wanted = str(sample_id)
            rows = [row for row in rows if str(row.get("sample_id")) == wanted]
        return rows

    def retrieve_ranks_for(
        self,
        *,
        question_id: str,
        sample_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Full ranked candidate list for one question, including retrieve losers."""
        wanted_q = str(question_id)
        wanted_s = str(sample_id) if sample_id is not None else None
        for row in self.retrieve_ranks:
            if str(row.get("question_id")) != wanted_q:
                continue
            if wanted_s is not None and str(row.get("sample_id") or "") != wanted_s:
                continue
            return row
        return None

    def attribution_for(
        self,
        *,
        role: str | None = None,
        writer_id: str | None = None,
        question_id: str | None = None,
        sample_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """LLM calls with role + claims they made.

        ``question_id`` matches reader rows and teacher claims whose
        ``injected_question_ids`` include that question. ``sample_id``
        keeps sibling conversations from leaking into the same filter.
        """
        rows = list(self.attribution)
        if role is not None:
            wanted = str(role)
            rows = [row for row in rows if str(row.get("role")) == wanted]
        if writer_id is not None:
            wanted = str(writer_id)
            rows = [row for row in rows if str(row.get("writer_id")) == wanted]
        if question_id is not None:
            wanted = str(question_id)
            rows = [
                row
                for row in rows
                if str(row.get("question_id") or "") == wanted
                or any(
                    wanted in (claim.get("injected_question_ids") or [])
                    for claim in (row.get("claims") or [])
                )
            ]
        if sample_id is not None:
            wanted = str(sample_id)
            rows = [row for row in rows if str(row.get("sample_id") or "") == wanted]
        return rows


def load_sandwich_audit(run_dir: str | Path) -> SandwichAudit:
    """Load the full experiment pack for analysis (no LLM).

    Missing optional layers (writer, graph, autorater, attribution) load
    as empty lists. Writer calls fall back to the compat path.
    """
    paths = AuditPaths.from_run_dir(run_dir)
    pred_path = predictions_jsonl(paths.run_dir)
    calls = load_jsonl(paths.writer_calls)
    if not calls:
        calls = load_jsonl(paths.writer_calls_compat)
    return SandwichAudit(
        paths=paths,
        meta=load_json(paths.run_meta),
        metrics=load_json(paths.metrics),
        predictions=load_jsonl(pred_path) if pred_path is not None else [],
        reader_traces=load_jsonl(paths.reader_traces),
        writer_index=load_jsonl(paths.writer_index),
        writer_calls=calls,
        lineage=load_jsonl(paths.lineage),
        retrieve_ranks=load_jsonl(paths.retrieve_ranks),
        graph_ingest=load_jsonl(paths.graph_ingest),
        writer_quality=load_json(paths.writer_quality),
        cost=load_json(paths.cost),
        attribution=load_jsonl(paths.attribution),
        autorater_verdicts=load_jsonl(paths.autorater_verdicts),
        autorater_traces=load_jsonl(paths.autorater_traces),
    )
