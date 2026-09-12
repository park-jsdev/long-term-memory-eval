"""Status from object storage / the local experiments directory. No database."""

from __future__ import annotations

from pathlib import Path

from src.memorybench.completed_run_skip import autorater_success_path, qa_success_path
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.gcs_run_workspace import (
    autorater_success_remote,
    gcs_blob_exists,
    gcs_enabled,
    qa_success_remote,
    remote_run_prefix,
)
from src.memorybench.load_experiment_yaml import load_experiment_yaml
from src.memorybench.open_configured_store import local_experiments_root


def report_experiment_status(config_path: str | Path) -> dict[str, int | list[str]]:
    cfg = load_experiment_yaml(config_path)
    specs = expand_run_matrix(cfg)
    out_root = local_experiments_root(cfg)
    completed: list[str] = []
    judged: list[str] = []
    failed: list[str] = []
    not_started: list[str] = []
    for spec in specs:
        run_dir = out_root / spec.run_id
        if gcs_enabled(cfg):
            if gcs_blob_exists(cfg, qa_success_remote(cfg, spec.run_id)):
                completed.append(spec.run_id)
                if gcs_blob_exists(cfg, autorater_success_remote(cfg, spec.run_id)):
                    judged.append(spec.run_id)
                continue
            prefix = remote_run_prefix(cfg, spec.run_id)
            if gcs_blob_exists(cfg, f"{prefix}/errors.jsonl") or gcs_blob_exists(
                cfg, f"{prefix}/run.json"
            ):
                failed.append(spec.run_id)
            else:
                not_started.append(spec.run_id)
            continue
        if qa_success_path(run_dir).is_file():
            completed.append(spec.run_id)
            if autorater_success_path(run_dir).is_file():
                judged.append(spec.run_id)
            continue
        if run_dir.exists():
            failed.append(spec.run_id)
        else:
            not_started.append(spec.run_id)
    return {
        "expected_runs": len(specs),
        "qa_completed": len(completed),
        "autorater_completed": len(judged),
        "failed_or_incomplete": len(failed),
        "not_started": len(not_started),
        "qa_completed_ids": completed,
        "failed_ids": failed,
        "not_started_ids": not_started,
    }
