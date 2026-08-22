"""Offline scorer CLI: recompute EM / token F1 / LoCoMo F1 from stored predictions.

This module is **metrics-only**. It never calls an LLM. Use it when the
string scorer or plots change and you want new numbers on the same
``predictions.jsonl`` without re-billing.

Not an LLM-as-judge / autorater. Autoraters (model grades the
answer) belongs in a separate module so this path stays deterministic
and gold-reference-based.

Memory, prompt, or model changes still need ``run_locomo_pipeline_with_memory_config``
in ``run.py`` (those alter the stored predictions). Holding predictions fixed
keeps new scores comparable and avoids rate limits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.locomo_eval.metrics import summarize_predictions
from src.locomo_eval.report import write_category_csv, write_json, make_plots


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Offline rescore of predictions.jsonl "
            "(EM / token F1 / LoCoMo F1 only; no API, not an LLM autorater)"
        )
    )
    p.add_argument("--predictions", required=True, help="predictions.jsonl from a run")
    p.add_argument("--output-dir", default=None, help="Where to write metrics (default: beside predictions)")
    args = p.parse_args(argv)

    pred_path = Path(args.predictions)
    rows = load_jsonl(pred_path)
    summary = summarize_predictions(rows)
    if rows:
        summary["memory_type"] = rows[0].get("memory_type")
        summary["reader_model"] = rows[0].get("reader_model")
        summary["prompt_version"] = rows[0].get("prompt_version")
        if rows[0].get("teacher_model"):
            summary["teacher_model"] = rows[0].get("teacher_model")
            summary["teacher_provider"] = rows[0].get("teacher_provider")

    out_dir = Path(args.output_dir) if args.output_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", summary)
    write_category_csv(out_dir / "metrics_by_category.csv", summary.get("by_category") or {})
    plots = make_plots(summary, out_dir / "plots")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"Wrote {out_dir / 'metrics.json'}")
    for plot in plots:
        print(f"Plot: {plot}")


if __name__ == "__main__":
    main()
