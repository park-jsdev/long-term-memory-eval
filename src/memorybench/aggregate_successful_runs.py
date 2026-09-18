"""Third wave: collect finished cells into ``experiments/<name>/aggregate/``.

Pulls packs from GCS when ``storage.backend`` is ``gcs``, concatenates Parquet
for notebooks, and copies a small audit catalog (SUMMARY / TRACE / ATTRIBUTION)
so analysis does not hunt per-run folders. Safe to rerun; always regenerates.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.memorybench.completed_run_skip import autorater_success_path, qa_success_path
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.gcs_run_workspace import (
    aggregate_prefix,
    collected_prefix,
    download_experiment_runs,
    gcs_enabled,
    remote_run_prefix,
    upload_aggregate_dir,
    upload_collected_dir,
)
from src.memorybench.load_experiment_yaml import load_experiment_yaml
from src.memorybench.open_configured_store import local_experiments_root

# Human/machine audit files copied into aggregate/by_run/<id>/. Full memory/
# dumps stay on the run prefix (too large to duplicate). GCS catalog download
# is the same set plus parquet: pulling whole packs (2800+ blobs) OOMs the
# 4Gi aggregate job (Cloud Run SIGKILL / signal 9).
CATALOG_FILES = (
    "SUMMARY.md",
    "TRACE.md",
    "ATTRIBUTION.md",
    "attribution.jsonl",
    "cost.json",
    "metrics.json",
    "metrics_by_category.csv",
    "run_meta.json",
    "run.json",
    "predictions.csv",
    "errors.jsonl",
    "autorater/SUMMARY.md",
    "autorater/autorater_metrics.json",
    "autorater/_SUCCESS",
    "_SUCCESS",
)
CATALOG_DOWNLOAD_FILES = CATALOG_FILES + (
    "examples.parquet",
    "summary.parquet",
    # Search times live here on older cell parquet. Downloaded to patch
    # aggregate examples; not copied into by_run (memory text is large).
    "predictions.jsonl",
)


def collect_experiment_results(
    config_path: str | Path,
    *,
    store: Any | None = None,
    mode: str = "catalog",
) -> dict[str, Path]:
    """Download (if GCS), write experiment folder, upload. No LLM.

    ``mode="catalog"`` (default): parquet + thin SUMMARY/TRACE/ATTRIBUTION copies.
    ``mode="full"``: entire run packs including ``memory/`` and traces (on-demand).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required for aggregate") from exc

    if mode not in ("catalog", "full"):
        raise ValueError(f"mode must be 'catalog' or 'full', got {mode!r}")
    cfg = load_experiment_yaml(config_path)
    specs = expand_run_matrix(cfg)
    name = str((cfg.get("experiment") or {}).get("name") or "experiment")
    out_root = local_experiments_root(cfg)
    if gcs_enabled(cfg):
        downloaded = download_experiment_runs(
            cfg,
            [spec.run_id for spec in specs],
            out_root,
            store=store,
            rel_allowlist=CATALOG_DOWNLOAD_FILES if mode == "catalog" else None,
        )
        n_blobs = sum(downloaded.values())
        print(f"downloaded {n_blobs} blobs from GCS for {len(specs)} cell(s)")

    folder = "aggregate" if mode == "catalog" else "collected"
    agg_dir = out_root / name / folder
    if agg_dir.exists():
        shutil.rmtree(agg_dir)
    agg_dir.mkdir(parents=True)

    cells: list[dict[str, Any]] = []
    example_tables = []
    summary_tables = []
    for spec in specs:
        run_dir = out_root / spec.run_id
        cell = _cell_record(cfg, spec, run_dir, mode=mode)
        cells.append(cell)
        if mode == "catalog":
            _copy_catalog(run_dir, agg_dir / "by_run" / spec.run_id)
        else:
            _copy_full_run(run_dir, agg_dir / "runs" / spec.run_id)
        if not cell["qa_success"]:
            continue
        examples = run_dir / "examples.parquet"
        summary = run_dir / "summary.parquet"
        if examples.is_file():
            example_tables.append(_examples_table_with_search(examples, run_dir, pa, pq))
        if summary.is_file():
            summary_tables.append(pq.read_table(summary))

    written: dict[str, Path] = {"aggregate_dir": agg_dir}
    cells_path = agg_dir / "cells.jsonl"
    cells_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cells),
        encoding="utf-8",
    )
    written["cells"] = cells_path

    status = {
        "experiment_name": name,
        "mode": mode,
        "expected_runs": len(specs),
        "qa_completed": sum(1 for row in cells if row["qa_success"]),
        "autorater_completed": sum(1 for row in cells if row["autorater_success"]),
        "failed_or_incomplete": sum(1 for row in cells if row["stage"] == "failed"),
        "not_started": sum(1 for row in cells if row["stage"] == "not_started"),
        "gcs": gcs_enabled(cfg),
        "aggregate_prefix": (
            (aggregate_prefix(cfg) if mode == "catalog" else collected_prefix(cfg))
            if gcs_enabled(cfg)
            else None
        ),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "cells": cells,
    }
    status_path = agg_dir / "status.json"
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    written["status"] = status_path

    examples_out = agg_dir / "examples.parquet"
    runs_out = agg_dir / "runs.parquet"
    if example_tables:
        pq.write_table(_concat(example_tables, pa), examples_out)
        written["examples"] = examples_out
    if summary_tables:
        pq.write_table(_concat(summary_tables, pa), runs_out)
        written["runs"] = runs_out

    summary_md = agg_dir / "SUMMARY.md"
    summary_md.write_text(_render_summary(name, status, mode=mode), encoding="utf-8")
    written["summary"] = summary_md
    (agg_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    written["success"] = agg_dir / "_SUCCESS"

    if mode == "catalog":
        uploaded = upload_aggregate_dir(cfg, agg_dir, store=store)
        remote = aggregate_prefix(cfg)
    else:
        uploaded = upload_collected_dir(cfg, agg_dir, store=store)
        remote = collected_prefix(cfg)
    if uploaded:
        print(f"uploaded {uploaded} blobs -> {remote}")
    return written


def collect_full_run_packs(
    config_path: str | Path, *, store: Any | None = None
) -> dict[str, Path]:
    """On-demand third wave: copy entire run packs including memory dumps."""
    return collect_experiment_results(config_path, store=store, mode="full")


def aggregate_successful_runs(
    config_path: str | Path, *, store: Any | None = None
) -> dict[str, Path]:
    """Compat name for the third-wave collect (parquet + catalog + GCS)."""
    return collect_experiment_results(config_path, store=store)


def _cell_record(
    cfg: dict[str, Any], spec: Any, run_dir: Path, *, mode: str
) -> dict[str, Any]:
    qa_ok = qa_success_path(run_dir).is_file()
    judge_ok = autorater_success_path(run_dir).is_file()
    has_error = (run_dir / "errors.jsonl").is_file()
    has_any = run_dir.is_dir() and any(run_dir.iterdir())
    if qa_ok and judge_ok:
        stage = "autorater_completed"
    elif qa_ok:
        stage = "qa_completed"
    elif has_error or has_any:
        stage = "failed"
    else:
        stage = "not_started"
    return {
        "run_index": spec.run_index,
        "run_id": spec.run_id,
        "memory_method": spec.memory_method,
        "reader_model": spec.reader.api_model_id,
        "reader_provider": spec.reader.provider,
        "seed": spec.seed,
        "stage": stage,
        "qa_success": qa_ok,
        "autorater_success": judge_ok,
        "gcs_prefix": remote_run_prefix(cfg, spec.run_id) if gcs_enabled(cfg) else None,
        "local_run_dir": str(run_dir),
        "catalog_dir": (
            f"by_run/{spec.run_id}" if mode == "catalog" else f"runs/{spec.run_id}"
        ),
    }


def _examples_table_with_search(examples: Path, run_dir: Path, pa: Any, pq: Any):
    """Concat-ready table; fill search seconds from predictions.jsonl if parquet omitted them."""
    table = pq.read_table(examples)
    names = set(table.schema.names)
    search_present = "search_latency_seconds" in names
    if search_present:
        col = table.column("search_latency_seconds")
        if col.null_count < len(col):
            return table
    from scripts.analysis.campaign_tables import overlay_search_latency_from_run

    frame = overlay_search_latency_from_run(table.to_pandas(), run_dir)
    return pa.Table.from_pandas(frame, preserve_index=False)


def _copy_catalog(run_dir: Path, dest_dir: Path) -> None:
    if not run_dir.is_dir():
        return
    for rel in CATALOG_FILES:
        src = run_dir / rel
        if not src.is_file():
            continue
        dest = dest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _copy_full_run(run_dir: Path, dest_dir: Path) -> None:
    if not run_dir.is_dir():
        return
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(run_dir, dest_dir)


def _render_summary(name: str, status: dict[str, Any], *, mode: str) -> str:
    lines = [
        f"# Experiment collect `{name}` ({mode})",
        "",
        (
            "Default third wave: parquet + thin audit catalog. "
            "Trigger `collect-full` for complete `memory/` dumps."
            if mode == "catalog"
            else "On-demand full collect: complete run packs including `{memory}` dumps and traces."
        ),
        "",
        f"- expected: {status['expected_runs']}",
        f"- QA `_SUCCESS`: {status['qa_completed']}",
        f"- autorater `_SUCCESS`: {status['autorater_completed']}",
        f"- failed / incomplete: {status['failed_or_incomplete']}",
        f"- not started: {status['not_started']}",
        f"- collected_at: `{status['collected_at']}`",
        "",
        "## Notebooks",
        "",
        "- `runs.parquet` — one row per completed QA cell",
        "- `examples.parquet` — one row per question (judge columns if autorater ran)",
        "- `cells.jsonl` / `status.json` — every matrix row, including missing cells",
        "",
        "## Cells",
        "",
        "| index | run_id | memory | reader | stage | catalog |",
        "|------:|--------|--------|--------|-------|---------|",
    ]
    for row in status["cells"]:
        lines.append(
            f"| {row['run_index']} | `{row['run_id']}` | `{row['memory_method']}` | "
            f"`{row['reader_model']}` | {row['stage']} | `{row['catalog_dir']}` |"
        )
    if mode == "catalog":
        pack_note = (
            "Per-cell `by_run/<run_id>/` holds SUMMARY / TRACE / ATTRIBUTION / metrics / cost. "
            "Full `{memory}` dumps: run `python -m src.memorybench collect-full <config>` "
            "(Cloud Run job `memorybench-collect-full`)."
        )
    else:
        pack_note = (
            "Per-cell `runs/<run_id>/` is the complete audit pack (`memory/`, `reader/`, "
            "predictions, plots, autorater). Same hashed `run_id` as QA and autorater."
        )
    lines.extend(["", pack_note, ""])
    if status.get("aggregate_prefix"):
        lines.append(f"GCS prefix: `{status['aggregate_prefix']}/`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _concat(tables, pa):
    try:
        return pa.concat_tables(tables, promote_options="default")
    except TypeError:
        return pa.concat_tables(tables)
