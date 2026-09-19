"""Offline experiment-validity verifier (configs, prompts, schemas, logs).

No API. One or more run directories, experiment folders, or experiment YAMLs.

  python -m scripts.analysis.verify_experiments experiments/<run_id>
  python -m scripts.analysis.verify_experiments experiments/locomo-mem0-reader-2025-writers-openai-deepseek
  python -m scripts.analysis.verify_experiments --graph-years \\
      experiments/locomo-mem0-reader-2025-writers-openai-deepseek \\
      experiments/locomo-mem0-reader-2026-writers-openai-deepseek
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.locomo_eval.experiments.verify_graph_years import (
    diagnose_graph_year_stagnation,
    render_graph_year_markdown,
)
from src.locomo_eval.experiments.verify_pack import (
    SCHEMA_VERSION,
    discover_run_dirs,
    dump_report_json,
    render_pack_markdown,
    verify_pack,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify sandwich run packs from on-disk audit logs (no API)."
    )
    p.add_argument(
        "paths",
        nargs="+",
        help="Run dir, experiment dir (aggregate/by_run), or hashed smoke dir",
    )
    p.add_argument(
        "--graph-years",
        action="store_true",
        help="Diagnose teacher_graph 2025 vs 2026 stagnation from these dirs",
    )
    p.add_argument("--json", dest="json_out", default=None, help="Write JSON report")
    p.add_argument("--md", dest="md_out", default=None, help="Write Markdown report")
    return p


def resolve_targets(raw_paths: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_absolute():
            cand = ROOT / path
            path = cand if cand.exists() else path
        for run_dir in discover_run_dirs(path):
            resolved = run_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
    return found


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dirs = resolve_targets(args.paths)
    reports = [verify_pack(d) for d in run_dirs]
    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "n_packs": len(reports),
        "packs": [r.to_dict() for r in reports],
    }
    md_parts = [f"# Experiment verifier (`{SCHEMA_VERSION}`)", ""]
    if not reports:
        md_parts.append("No run packs found.")
    for report in reports:
        md_parts.append(render_pack_markdown(report))
    graph = None
    if args.graph_years:
        graph = diagnose_graph_year_stagnation(list(args.paths), pack_reports=reports)
        payload["graph_year"] = graph.to_dict()
        md_parts.append(render_graph_year_markdown(graph))
    md = "\n".join(md_parts).rstrip() + "\n"
    print(md)
    if args.json_out:
        Path(args.json_out).write_text(dump_report_json(payload), encoding="utf-8")
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if any(r.verdict == "invalid" for r in reports):
        return 1
    if graph is not None and graph.technical and not reports:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
