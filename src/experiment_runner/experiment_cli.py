"""Experiment runner CLI: manifest, execute-qa, execute-autorater, aggregate, status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.locomo_eval.env import load_env
from src.experiment_runner.aggregate_successful_runs import (
    collect_experiment_results,
    collect_full_run_packs,
)
from src.experiment_runner.execute_autorater_run import execute_autorater_run
from src.experiment_runner.execute_qa_run import execute_qa_run
from src.experiment_runner.expand_run_matrix import expand_run_matrix
from src.experiment_runner.load_experiment_yaml import load_experiment_yaml
from src.experiment_runner.open_configured_store import local_experiments_root
from src.experiment_runner.report_experiment_status import report_experiment_status
from src.experiment_runner.write_run_manifest import write_run_manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="experiment_runner",
        description="Expand a YAML matrix and run one eval cell per task.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    man = sub.add_parser("write-manifest", help="Expand YAML to manifest/runs.jsonl")
    man.add_argument("config")
    man.add_argument("--output", default=None)

    qa = sub.add_parser("execute-qa", help="One matrix cell: answer LLM + string metrics")
    _add_run_flags(qa)

    judge = sub.add_parser(
        "execute-autorater",
        help="One matrix cell: Mem0 judge on stored predictions (QA must be done)",
    )
    _add_run_flags(judge)

    agg = sub.add_parser(
        "aggregate",
        help="Default collect: Parquet + thin audit catalog (after QA, ideally after autorater)",
    )
    agg.add_argument("config")

    full = sub.add_parser(
        "collect-full",
        help="On-demand: copy complete run packs including memory dumps",
    )
    full.add_argument("config")

    rep = sub.add_parser(
        "report",
        help="Offline tables/plots from analysis YAML (campaign + experiment)",
    )
    rep.add_argument("config")
    rep.add_argument(
        "--experiment",
        default=None,
        help="Campaign experiment id (smoke|baseline|writers). Omit for all + campaign concat.",
    )

    st = sub.add_parser("status", help="Count completed / failed / not started")
    st.add_argument("config")
    return p


def _add_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("config")
    p.add_argument("--run-index", type=int, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--allow-unconfirmed",
        action="store_true",
        help="Run cells whose model_snapshot is still TO_CONFIRM",
    )


def main(argv: list[str] | None = None) -> None:
    loaded = load_env()
    if loaded is not None:
        print(f"Loaded env from {loaded}")
    args = build_parser().parse_args(argv)
    if args.command == "write-manifest":
        cfg = load_experiment_yaml(args.config)
        specs = expand_run_matrix(cfg)
        dest = Path(args.output) if args.output else (
            local_experiments_root(cfg)
            / str((cfg.get("experiment") or {}).get("name") or "experiment")
            / "manifest"
            / "runs.jsonl"
        )
        path = write_run_manifest(specs, dest)
        print(f"wrote {len(specs)} runs -> {path}")
        return
    if args.command == "execute-qa":
        execute_qa_run(
            args.config,
            run_index=args.run_index,
            run_id=args.run_id,
            force=args.force,
            allow_unconfirmed=args.allow_unconfirmed,
        )
        return
    if args.command == "execute-autorater":
        execute_autorater_run(
            args.config,
            run_index=args.run_index,
            run_id=args.run_id,
            force=args.force,
            allow_unconfirmed=args.allow_unconfirmed,
        )
        return
    if args.command == "aggregate":
        written = collect_experiment_results(args.config)
        print(f"aggregate complete -> {written.get('aggregate_dir')}")
        for key, path in written.items():
            if key == "aggregate_dir":
                continue
            print(f"  {key}: {path}")
        return
    if args.command == "collect-full":
        written = collect_full_run_packs(args.config)
        print(f"collect-full complete -> {written.get('aggregate_dir')}")
        for key, path in written.items():
            if key == "aggregate_dir":
                continue
            print(f"  {key}: {path}")
        return
    if args.command == "report":
        from src.experiment_runner.analysis.report import run_report

        reports = run_report(args.config, experiment_id=args.experiment)
        for report in reports:
            print(f"{report.scope} -> {report.out_dir}")
            if report.missing_packs:
                print(f"  missing: {', '.join(report.missing_packs)}")
            for item in report.results:
                extra = f" skipped={item.skipped}" if item.skipped else f" n={len(item.table)}"
                print(f"  {item.spec.id}{extra}")
        return
    if args.command == "status":
        report = report_experiment_status(args.config)
        print(json.dumps({k: v for k, v in report.items() if not k.endswith("_ids")}, indent=2))
        print(
            f"expected={report['expected_runs']} "
            f"qa_completed={report['qa_completed']} "
            f"autorater_completed={report['autorater_completed']} "
            f"failed={report['failed_or_incomplete']} "
            f"not_started={report['not_started']}"
        )
        return
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
