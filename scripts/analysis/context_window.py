"""CLI: context-window utilization vs LoCoMo memory coverage.

Offline. Does not launch QA jobs.

    python -m scripts.analysis.context_window
    python -m scripts.analysis.context_window --out experiments/_design/context_window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_runner.analysis.context_window import (  # noqa: E402
    DEFAULT_AUDIT_RUN,
    DEFAULT_LOCAL_RUNS,
    DEFAULT_PACKS,
    DEFAULT_PREPROCESS,
    render_context_window_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit full-context injection size vs published model windows."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repo root (experiments/ lives here).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Report directory (default experiments/_design/context_window).",
    )
    parser.add_argument(
        "--packs",
        nargs="*",
        default=None,
        help="Experiment pack names under experiments/. Default: 2025/2026 reader+writer campaigns.",
    )
    parser.add_argument(
        "--audit-run",
        default=DEFAULT_AUDIT_RUN,
        help="Run id with memory/index.jsonl (default full_context_qa).",
    )
    parser.add_argument(
        "--preprocess",
        default=DEFAULT_PREPROCESS,
        help="Preprocess run id for session counts.",
    )
    parser.add_argument(
        "--local-runs",
        nargs="*",
        default=None,
        help="Single locomo_eval run ids (predictions.jsonl). Default: full_context_qa rag_k2_256_qa mem0_qa.",
    )
    args = parser.parse_args(argv)
    report = render_context_window_report(
        root=args.root,
        out_dir=args.out,
        packs=args.packs if args.packs else list(DEFAULT_PACKS),
        audit_run=args.audit_run,
        preprocess=args.preprocess,
        local_runs=args.local_runs if args.local_runs else list(DEFAULT_LOCAL_RUNS),
    )
    print(f"wrote {report.out_dir}")
    if report.summary_path:
        print(f"  {report.summary_path}")
    for path in report.csv_paths:
        print(f"  {path}")
    for path in report.plot_paths:
        print(f"  {path}")
    if report.skipped_packs:
        print("skipped:", ", ".join(report.skipped_packs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
