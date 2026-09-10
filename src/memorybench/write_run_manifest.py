"""Write ``manifest/runs.jsonl`` — the durable list of intended Cloud Run tasks."""

from __future__ import annotations

import json
from pathlib import Path

from src.memorybench.experiment_run_spec import ExperimentRunSpec


def write_run_manifest(specs: list[ExperimentRunSpec], dest: str | Path) -> Path:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec.to_manifest_row(), ensure_ascii=False) + "\n")
    return path


def load_run_manifest(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
