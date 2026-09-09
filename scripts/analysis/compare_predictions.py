"""Figure 2 kernel: compare **exactly two** prediction JSONLs (offline LoCoMo F1).

This is **not** the sandwich experiment report. Use ``scripts/compare_full_runs.py``
when you have two full ``experiments/<run_id>/`` packs and want SUMMARY /
category tables / memory-distinctness checks. That script calls the plots
from this module.

This CLI is the reusable two-pack analysis: pair on ``question_id``, write
overall LoCoMo F1 bars (colors = conditions), boxplot, and histograms.
``--a`` / ``--b`` may be a run directory or a ``predictions.jsonl`` path.
Paths are resolved from cwd and the repo root.

    python -m src.locomo_eval.run --config configs/raw_chunks.yaml --reader mock --max-questions 5 --run-id cmp_raw_chunks
    python -m src.locomo_eval.run --config configs/session_summaries.yaml --reader mock --max-questions 5 --run-id cmp_session_summaries
    python -m scripts.analysis.compare_predictions --a cmp_raw_chunks --b cmp_session_summaries --out experiments/compare_raw_chunks_session_summaries
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.plots import (
    write_locomo_f1_boxplot,
    write_locomo_f1_histograms,
    write_locomo_f1_overall_bars,
)
from src.locomo_eval.experiments.audit_loader import resolve_predictions_jsonl
from src.locomo_eval.metrics import score_row


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load prediction rows from a run dir, run id, or ``predictions.jsonl``."""
    p = resolve_predictions_jsonl(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def locomo_f1_of_row(row: dict[str, Any]) -> float:
    """Use a stored LoCoMo F1 if present; otherwise score prediction vs gold."""
    if row.get("locomo_f1") is not None:
        return float(row["locomo_f1"])
    return float(
        score_row(
            str(row.get("predicted_answer", "")),
            str(row.get("reference_answer", "")),
            int(row.get("category", 0)),
        )["locomo_f1"]
    )


def pair_predictions(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inner-join two prediction lists on ``question_id``; attach LoCoMo F1 + delta."""
    by_a = {r["question_id"]: r for r in rows_a if r.get("question_id")}
    by_b = {r["question_id"]: r for r in rows_b if r.get("question_id")}
    paired: list[dict[str, Any]] = []
    for qid in sorted(set(by_a) & set(by_b)):
        ra, rb = by_a[qid], by_b[qid]
        fa, fb = locomo_f1_of_row(ra), locomo_f1_of_row(rb)
        paired.append(
            {
                "question_id": qid,
                "category": ra.get("category"),
                "question": ra.get("question"),
                "locomo_f1_a": fa,
                "locomo_f1_b": fb,
                "delta_locomo_f1_b_minus_a": round(fb - fa, 6),
            }
        )
    return paired


def write_paired_locomo_csv(path: Path, paired: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not paired:
        path.write_text("", encoding="utf-8")
        return
    fields = list(paired[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(paired)


def write_compare_prediction_plots(
    paired: list[dict[str, Any]],
    *,
    label_a: str,
    label_b: str,
    out_dir: Path,
) -> list[Path]:
    """Boxplot, histograms, and overall LoCoMo F1 bars (colors = conditions)."""
    scores_a = [float(r["locomo_f1_a"]) for r in paired]
    scores_b = [float(r["locomo_f1_b"]) for r in paired]
    written: list[Path] = []
    n = len(paired)
    mean_a = sum(scores_a) / n if n else 0.0
    mean_b = sum(scores_b) / n if n else 0.0
    overall = write_locomo_f1_overall_bars(
        mean_a,
        mean_b,
        label_a=label_a,
        label_b=label_b,
        path=out_dir / "locomo_f1_overall.png",
    )
    box = write_locomo_f1_boxplot(
        scores_a,
        scores_b,
        label_a=label_a,
        label_b=label_b,
        path=out_dir / "locomo_f1_boxplot.png",
    )
    hist = write_locomo_f1_histograms(
        scores_a,
        scores_b,
        label_a=label_a,
        label_b=label_b,
        path=out_dir / "locomo_f1_histograms.png",
    )
    if overall is not None:
        written.append(overall)
    if box is not None:
        written.append(box)
    if hist is not None:
        written.append(hist)
    return written


def compare_predictions(
    path_a: str | Path,
    path_b: str | Path,
    *,
    out_dir: str | Path,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict[str, Any]:
    """Load two prediction sets, pair them, write CSV + LoCoMo F1 plots."""
    rows_a = load_prediction_rows(path_a)
    rows_b = load_prediction_rows(path_b)
    label_a = label_a or Path(path_a).name
    label_b = label_b or Path(path_b).name
    paired = pair_predictions(rows_a, rows_b)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_paired_locomo_csv(out / "paired_locomo_f1.csv", paired)
    plots = write_compare_prediction_plots(
        paired, label_a=label_a, label_b=label_b, out_dir=out / "plots"
    )
    n = len(paired)
    mean_a = round(sum(r["locomo_f1_a"] for r in paired) / n, 4) if n else None
    mean_b = round(sum(r["locomo_f1_b"] for r in paired) / n, 4) if n else None
    summary = {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": n,
        "mean_locomo_f1_a": mean_a,
        "mean_locomo_f1_b": mean_b,
        "mean_delta_b_minus_a": round(mean_b - mean_a, 4) if n else None,
        "plots": [str(p) for p in plots],
    }
    (out / "compare_predictions.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compare exactly two prediction sets (LoCoMo F1 overall bars, boxplot, histograms). "
            "Offline string metrics only; not an LLM autorater."
        )
    )
    p.add_argument(
        "--a",
        required=True,
        help="Condition A: experiments/<run_id> dir, run id, or predictions.jsonl",
    )
    p.add_argument(
        "--b",
        required=True,
        help="Condition B: experiments/<run_id> dir, run id, or predictions.jsonl",
    )
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    args = p.parse_args(argv)
    summary = compare_predictions(
        args.a,
        args.b,
        out_dir=args.out,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
