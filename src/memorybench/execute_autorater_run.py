"""Second Cloud Run stage: Mem0 autorater on a finished QA pack.

Requires ``_SUCCESS`` from ``execute_qa_run``. Does not call the answer LLM.
"""

from __future__ import annotations

from pathlib import Path

from src.memorybench.completed_run_skip import (
    autorater_success_path,
    qa_success_path,
    should_skip_completed,
    write_success_marker,
)
from src.memorybench.execute_qa_run import select_run_spec
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.gcs_run_workspace import (
    autorater_success_remote,
    download_run_dir,
    gcs_blob_exists,
    gcs_enabled,
    qa_success_remote,
    upload_run_dir,
)
from src.memorybench.load_experiment_yaml import load_experiment_yaml
from src.memorybench.open_configured_store import local_experiments_root
from src.memorybench.write_analysis_parquet import (
    load_autorater_verdicts,
    load_prediction_rows,
    qa_summary_row,
    write_examples_parquet,
    write_summary_parquet,
)


def execute_autorater_run(
    config_path: str | Path,
    *,
    run_index: int | None = None,
    run_id: str | None = None,
    force: bool = False,
    allow_unconfirmed: bool = False,
) -> Path:
    from scripts.analysis.run_benchmark import main as autorater_main

    cfg = load_experiment_yaml(config_path)
    specs = expand_run_matrix(cfg)
    spec = select_run_spec(specs, run_index=run_index, run_id=run_id)
    if spec.status != "runnable" and not allow_unconfirmed:
        raise SystemExit(
            f"{spec.run_id} status={spec.status}. Pin model_snapshot or pass --allow-unconfirmed"
        )
    out_root = local_experiments_root(cfg)
    run_dir = out_root / spec.run_id
    if gcs_enabled(cfg) and not qa_success_path(run_dir).is_file():
        downloaded = download_run_dir(cfg, spec.run_id, run_dir)
        print(f"downloaded {downloaded} blobs from GCS for {spec.run_id}")
    if (
        gcs_enabled(cfg)
        and not force
        and gcs_blob_exists(cfg, autorater_success_remote(cfg, spec.run_id))
    ):
        print(f"skip autorater {spec.run_id}: GCS autorater/_SUCCESS exists")
        return run_dir
    if not qa_success_path(run_dir).is_file():
        hint = ""
        if gcs_enabled(cfg) and not gcs_blob_exists(cfg, qa_success_remote(cfg, spec.run_id)):
            hint = " GCS also has no QA _SUCCESS."
        raise SystemExit(
            f"QA is not complete for {spec.run_id} (missing {qa_success_path(run_dir)}). "
            f"Run execute-qa first.{hint}"
        )
    marker = autorater_success_path(run_dir)
    if should_skip_completed(marker, force=force):
        print(f"skip autorater {spec.run_id}: {marker} exists")
        return run_dir

    execution = cfg.get("execution") or {}
    judge_provider = str(
        execution.get("judge_provider") or spec.judge_provider or "openai"
    )
    argv = [
        "--run",
        str(run_dir),
        "--autorater",
        judge_provider,
        "--model",
        spec.judge_model,
    ]
    autorater_main(argv)
    rows = load_prediction_rows(run_dir)
    verdicts = load_autorater_verdicts(run_dir)
    write_examples_parquet(
        run_dir / "examples.parquet",
        spec=spec,
        prediction_rows=rows,
        verdicts_by_qid=verdicts,
    )
    import json

    meta = {}
    metrics = {}
    meta_path = run_dir / "run_meta.json"
    metrics_path = run_dir / "metrics.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    write_summary_parquet(
        run_dir / "summary.parquet",
        qa_summary_row(
            spec,
            run_dir=run_dir,
            n_examples=len(rows),
            metrics=metrics,
            meta=meta,
            status="completed",
        ),
    )
    write_success_marker(marker)
    uploaded = upload_run_dir(cfg, spec.run_id, run_dir)
    extra = f" uploaded={uploaded} blobs" if uploaded else ""
    print(f"autorater complete {spec.run_id}{extra}")
    return run_dir
