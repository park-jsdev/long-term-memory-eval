"""Aggregate N autorater / QA seed packs into mean ± std and 95% CIs.

Paper protocol: J is the mean of 10 independent judge runs ± 1 std.
This script never calls an LLM; it only reads finished packs.

    python -m scripts.analysis.aggregate_seeds \\
      --packs experiments/rag_qa/autorater_seeds/seed_00 experiments/rag_qa/autorater_seeds/seed_01 \\
      --out experiments/rag_qa/autorater
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.locomo_eval.report import write_json
from src.locomo_eval.stats import summarize_seed_metrics


def _load_metrics(pack: Path) -> dict[str, Any]:
    for name in ("autorater_metrics.json", "metrics.json"):
        path = Path(pack) / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No autorater_metrics.json or metrics.json in {pack}")


def _metric_row(metrics_doc: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics_doc.get("metrics") or metrics_doc
    return {
        "llm_judge_pct": metrics.get("llm_judge_pct"),
        "mem0_f1_pct": metrics.get("mem0_f1_pct"),
        "mem0_bleu1_pct": metrics.get("mem0_bleu1_pct"),
        "locomo_f1": metrics.get("locomo_f1")
        or ((metrics_doc.get("metrics") or {}).get("locomo_f1")),
        "metrics": metrics,
    }


def aggregate_autorater_seeds(
    packs: list[Path],
    *,
    out_dir: Path | None = None,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    rows = [_metric_row(_load_metrics(p)) for p in packs]
    summary = summarize_seed_metrics(rows, bootstrap_seed=bootstrap_seed)
    j = summary.get("llm_judge_pct") or {}
    report = {
        "n_seeds": len(packs),
        "packs": [str(p) for p in packs],
        "paper_protocol": (
            "Chhikara et al. arXiv:2504.19413 report mean ± std of 10 judge runs. "
            "This aggregate is that table locally; it is not a Platform re-run."
        ),
        "metrics": summary,
        "headline": {
            "J_mean": j.get("mean"),
            "J_std": j.get("std"),
            "J_ci95": [j.get("ci95_low"), j.get("ci95_high")],
        },
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "seed_aggregate.json", report)
        lines = [
            "# Multi-seed autorater aggregate",
            "",
            f"- n_seeds: {len(packs)}",
            f"- J mean ± std: **{j.get('mean')} ± {j.get('std')}**",
            f"- J 95% CI: [{j.get('ci95_low')}, {j.get('ci95_high')}]",
            "",
            "Paper reports mean ± 1 std of 10 independent LLM-as-a-Judge runs.",
            "",
        ]
        (out / "SEED_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Aggregate autorater/QA seed packs (mean, std, 95% CI). No API."
    )
    p.add_argument("--packs", nargs="+", required=True, help="Seed pack directories")
    p.add_argument("--out", default=None, help="Write seed_aggregate.json here")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    args = p.parse_args(argv)
    report = aggregate_autorater_seeds(
        [Path(x) for x in args.packs],
        out_dir=Path(args.out) if args.out else None,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(report["headline"], indent=2))


if __name__ == "__main__":
    main()
