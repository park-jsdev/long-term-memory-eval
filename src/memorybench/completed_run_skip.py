"""Idempotent skip for Cloud Run retries vs local regenerate.

Default: if ``_SUCCESS`` exists, do not call the LLM again.
``--force`` deletes the marker and lets locomo_eval wipe the audit pack.

This is the opposite of a bare ``python -m src.locomo_eval.run`` invocation,
which always clears and regenerates. The harness owns skip; locomo_eval still
regenerates whenever it actually runs.
"""

from __future__ import annotations

from pathlib import Path

QA_SUCCESS = "_SUCCESS"
AUTORATER_SUCCESS = "autorater/_SUCCESS"


def qa_success_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / QA_SUCCESS


def autorater_success_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "autorater" / "_SUCCESS"


def should_skip_completed(marker: str | Path, *, force: bool) -> bool:
    if force:
        return False
    return Path(marker).is_file()


def write_success_marker(marker: str | Path) -> Path:
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")
    return path
