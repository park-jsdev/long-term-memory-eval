"""Concatenate completed-run Parquet files. Ignore directories without ``_SUCCESS``."""

from __future__ import annotations

from pathlib import Path

from src.memorybench.completed_run_skip import qa_success_path
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.load_experiment_yaml import load_experiment_yaml
from src.memorybench.open_configured_store import local_experiments_root


def aggregate_successful_runs(config_path: str | Path) -> dict[str, Path]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required for aggregate") from exc

    cfg = load_experiment_yaml(config_path)
    specs = expand_run_matrix(cfg)
    name = str((cfg.get("experiment") or {}).get("name") or "experiment")
    out_root = local_experiments_root(cfg)
    agg_dir = out_root / name / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    example_tables = []
    summary_tables = []
    for spec in specs:
        run_dir = out_root / spec.run_id
        if not qa_success_path(run_dir).is_file():
            continue
        examples = run_dir / "examples.parquet"
        summary = run_dir / "summary.parquet"
        if examples.is_file():
            example_tables.append(pq.read_table(examples))
        if summary.is_file():
            summary_tables.append(pq.read_table(summary))

    written: dict[str, Path] = {}
    examples_out = agg_dir / "examples.parquet"
    runs_out = agg_dir / "runs.parquet"
    if example_tables:
        pq.write_table(_concat(example_tables, pa), examples_out)
        written["examples"] = examples_out
    if summary_tables:
        pq.write_table(_concat(summary_tables, pa), runs_out)
        written["runs"] = runs_out
    return written


def _concat(tables, pa):
    try:
        return pa.concat_tables(tables, promote_options="default")
    except TypeError:
        return pa.concat_tables(tables)
