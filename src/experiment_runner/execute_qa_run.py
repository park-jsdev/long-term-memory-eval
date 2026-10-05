"""Execute one run spec: wrap ``locomo_eval.run``, then write Parquet + ``_SUCCESS``.

Does not run the autorater. Judge is a later ``execute_autorater_run`` job.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_config
from src.locomo_eval.run import build_parser, run_locomo_pipeline_with_memory_config
from src.experiment_runner.completed_run_skip import (
    qa_success_path,
    should_skip_completed,
    write_success_marker,
)
from src.experiment_runner.expand_run_matrix import expand_run_matrix
from src.experiment_runner.experiment_run_spec import ExperimentRunSpec
from src.experiment_runner.load_experiment_yaml import load_experiment_yaml
from src.experiment_runner.gcs_run_workspace import (
    ensure_dataset_local,
    ensure_shared_index_local,
    gcs_blob_exists,
    gcs_enabled,
    qa_success_remote,
    upload_run_dir,
)
from src.experiment_runner.open_configured_store import local_experiments_root
from src.experiment_runner.resolve_task_index import resolve_task_index
from src.experiment_runner.write_analysis_parquet import (
    load_prediction_rows,
    qa_summary_row,
    write_examples_parquet,
    write_summary_parquet,
)

ROOT = Path(__file__).resolve().parents[2]


def select_run_spec(
    specs: list[ExperimentRunSpec],
    *,
    run_index: int | None = None,
    run_id: str | None = None,
) -> ExperimentRunSpec:
    if run_id:
        for spec in specs:
            if spec.run_id == run_id:
                return spec
        raise SystemExit(f"run_id {run_id!r} is not in the expanded matrix")
    index = resolve_task_index(run_index)
    if index < 0 or index >= len(specs):
        raise SystemExit(f"run-index {index} out of range 0..{len(specs) - 1}")
    return specs[index]


def execute_qa_run(
    config_path: str | Path,
    *,
    run_index: int | None = None,
    run_id: str | None = None,
    force: bool = False,
    allow_unconfirmed: bool = False,
) -> Path:
    cfg = load_experiment_yaml(config_path)
    specs = expand_run_matrix(cfg)
    spec = select_run_spec(specs, run_index=run_index, run_id=run_id)
    if spec.status != "runnable" and not allow_unconfirmed:
        raise SystemExit(
            f"{spec.run_id} status={spec.status}. Pin model_snapshot or pass --allow-unconfirmed"
        )
    out_root = local_experiments_root(cfg)
    run_dir = out_root / spec.run_id
    if (
        gcs_enabled(cfg)
        and not force
        and gcs_blob_exists(cfg, qa_success_remote(cfg, spec.run_id))
    ):
        print(f"skip qa {spec.run_id}: GCS _SUCCESS exists (pass --force to regenerate)")
        return run_dir
    marker = qa_success_path(run_dir)
    if should_skip_completed(marker, force=force):
        print(f"skip qa {spec.run_id}: {marker} exists (pass --force to regenerate)")
        return run_dir

    work_cfg = dict(cfg)
    work_cfg["benchmark"] = dict(cfg.get("benchmark") or {})
    work_cfg["benchmark"]["dataset_path"] = str(ensure_dataset_local(cfg))

    execution = cfg.get("execution") or {}
    reader_override = execution.get("reader_provider")
    try:
        _ensure_shared_indexes(cfg, spec, out_root)
        _require_shared_indexes(spec, out_root)
        argv = _qa_argv(spec, work_cfg, out_root, reader_override)
        args = build_parser().parse_args(argv)
        locomo_cfg = load_config(args.config)
        if spec.agent_comparison:
            locomo_cfg["agent"] = dict(locomo_cfg.get("agent") or {})
            locomo_cfg["agent"]["comparison"] = dict(spec.agent_comparison)
        run_locomo_pipeline_with_memory_config(locomo_cfg, args)
        _write_qa_artifacts(spec, run_dir, status="completed")
        write_success_marker(marker)
        uploaded = upload_run_dir(cfg, spec.run_id, run_dir)
        extra = f" uploaded={uploaded} blobs" if uploaded else ""
        print(f"qa complete {spec.run_id} -> {run_dir}{extra}")
        return run_dir
    except Exception as exc:
        _write_error(run_dir, spec, stage="qa", exc=exc)
        upload_run_dir(cfg, spec.run_id, run_dir)
        raise


def _qa_argv(
    spec: ExperimentRunSpec,
    cfg: dict[str, Any],
    out_root: Path,
    reader_override: str | None,
) -> list[str]:
    reader = str(reader_override or spec.reader.provider)
    argv = [
        "--config",
        str(ROOT / spec.method_yaml),
        "--run-id",
        spec.run_id,
        "--output-dir",
        str(out_root),
        "--memory",
        spec.memory_method,
        "--reader",
        reader,
        "--model",
        spec.reader.api_model_id,
        "--prompt",
        spec.prompt_path,
    ]
    data = (cfg.get("benchmark") or {}).get("dataset_path")
    if data:
        argv.extend(["--data", str(data)])
    if spec.max_questions is not None:
        argv.extend(["--max-questions", str(spec.max_questions)])
    if spec.max_samples is not None:
        argv.extend(["--max-samples", str(spec.max_samples)])
    if spec.sample_id:
        argv.extend(["--sample-id", spec.sample_id])
    if spec.mem0_index_run_id:
        argv.extend(["--mem0-index-run-id", spec.mem0_index_run_id])
    if spec.rag_index_run_id:
        argv.extend(["--rag-index-run-id", spec.rag_index_run_id])
    if spec.writer is not None:
        argv.extend(
            [
                "--writer",
                spec.writer.provider,
                "--writer-model",
                spec.writer.api_model_id,
            ]
        )
        if spec.writer.thinking is False:
            argv.extend(["--thinking", "off"])
        elif spec.writer.thinking is True:
            argv.extend(["--thinking", "on"])
        if spec.writer.max_tokens is not None:
            argv.extend(["--writer-max-tokens", str(spec.writer.max_tokens)])
    elif spec.reader.thinking is False:
        argv.extend(["--reader-thinking", "off"])
    elif spec.reader.thinking is True:
        argv.extend(["--reader-thinking", "on"])
    if spec.writer is None and spec.reader.max_tokens is not None:
        argv.extend(["--max-tokens", str(spec.reader.max_tokens)])
    agent = None
    execution = cfg.get("execution") or {}
    if execution.get("agent_provider"):
        agent = str(execution.get("agent_provider"))
    elif spec.agent:
        agent = str(spec.agent)
    if agent and agent.lower() not in ("none", "null", ""):
        argv.extend(["--agent", agent])
        if spec.agent_persist is False:
            argv.extend(["--agent-persist", "off"])
        elif spec.agent_persist is True:
            argv.extend(["--agent-persist", "on"])
        if spec.agent_sessions:
            argv.extend(["--agent-sessions", str(spec.agent_sessions)])
        if spec.agent_tools:
            argv.extend(["--agent-tools", str(spec.agent_tools)])
        if spec.agent_prompt_mode:
            argv.extend(["--agent-prompt-mode", str(spec.agent_prompt_mode)])
    return argv


def _ensure_shared_indexes(
    cfg: dict[str, Any], spec: ExperimentRunSpec, out_root: Path
) -> None:
    if spec.rag_index_run_id:
        ensure_shared_index_local(cfg, spec.rag_index_run_id, "rag_index", out_root)
    if spec.mem0_index_run_id:
        ensure_shared_index_local(cfg, spec.mem0_index_run_id, "mem0_index", out_root)


def _require_shared_indexes(spec: ExperimentRunSpec, out_root: Path) -> None:
    if spec.mem0_index_run_id:
        path = out_root / spec.mem0_index_run_id / "mem0_index"
        if not path.is_dir():
            raise SystemExit(
                f"Missing shared Mem0 index at {path}. Build once with "
                "python -m src.locomo_eval.mem0.run_index --config configs/writers/mem0.yaml "
                f"--run-id {spec.mem0_index_run_id}"
            )
    if spec.rag_index_run_id:
        path = out_root / spec.rag_index_run_id / "rag_index"
        if not path.is_dir():
            raise SystemExit(
                f"Missing shared RAG index at {path}. Build once with "
                "python -m src.locomo_eval.rag.run_index --config configs/writers/rag.yaml "
                f"--run-id {spec.rag_index_run_id}"
            )


def _write_qa_artifacts(spec: ExperimentRunSpec, run_dir: Path, status: str) -> None:
    rows = load_prediction_rows(run_dir)
    meta = _read_json(run_dir / "run_meta.json")
    metrics = _read_json(run_dir / "metrics.json")
    write_examples_parquet(
        run_dir / "examples.parquet", spec=spec, prediction_rows=rows
    )
    summary = qa_summary_row(
        spec,
        run_dir=run_dir,
        n_examples=len(rows),
        metrics=metrics,
        meta=meta,
        status=status,
    )
    write_summary_parquet(run_dir / "summary.parquet", summary)
    run_json = {
        **spec.to_manifest_row(),
        "status": status,
        "git_commit": meta.get("code_git_hash"),
        "data_sha256": meta.get("data_sha256"),
        "prompt_path": spec.prompt_path,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "audit_predictions": "predictions.jsonl",
        "examples_parquet": "examples.parquet",
        "summary_parquet": "summary.parquet",
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_error(run_dir: Path, spec: ExperimentRunSpec, *, stage: str, exc: BaseException) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": spec.run_id,
        "example_id": None,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "attempt": 1,
        "traceback": traceback.format_exc(),
    }
    path = run_dir / "errors.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {
        **spec.to_manifest_row(),
        "status": "failed",
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
    (run_dir / "run.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
