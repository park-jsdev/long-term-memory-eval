"""Human-auditable CSV/JSON reports and simple plots."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Flatten for spreadsheet audit: drop very long memory by default but keep a clip.
    fieldnames = [
        "sample_id",
        "question_id",
        "category",
        "question",
        "reference_answer",
        "predicted_answer",
        "exact_match",
        "token_f1",
        "locomo_f1",
        "memory_type",
        "reader_model",
        "teacher_model",
        "prompt_version",
        "cached",  # LlmResponseHash hit if wired; currently always false from run.py
        "memory_chars",
        "memory_preview",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            mem = str(r.get("memory_text", ""))
            w.writerow(
                {
                    **r,
                    "memory_chars": len(mem),
                    "memory_preview": mem[:200].replace("\n", " "),
                }
            )


def write_category_csv(path: Path, by_category: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "category_id", "n", "exact_match", "token_f1", "locomo_f1"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, stats in by_category.items():
            w.writerow(
                {
                    "category": name,
                    "category_id": stats.get("category_id"),
                    "n": stats.get("n"),
                    "exact_match": stats.get("exact_match"),
                    "token_f1": stats.get("token_f1"),
                    "locomo_f1": stats.get("locomo_f1"),
                }
            )


def make_plots(summary: dict[str, Any], out_dir: Path) -> list[Path]:
    """Write simple bar charts; no-op gracefully if matplotlib missing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: list[Path] = []
    by_cat = summary.get("by_category") or {}
    names, locomo_f1, token_f1, counts = [], [], [], []
    for name, stats in by_cat.items():
        if not stats.get("n"):
            continue
        names.append(name)
        locomo_f1.append(stats.get("locomo_f1") or 0.0)
        token_f1.append(stats.get("token_f1") or 0.0)
        counts.append(stats.get("n") or 0)

    if names:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(names))
        width = 0.35
        ax.bar([i - width / 2 for i in x], locomo_f1, width, label="LoCoMo F1")
        ax.bar([i + width / 2 for i in x], token_f1, width, label="Token F1")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("QA score by LoCoMo category")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "f1_by_category.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(names, counts, color="#4C78A8")
        ax.set_ylabel("# questions")
        ax.set_title("Question counts by category")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        p = out_dir / "counts_by_category.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    metrics = summary.get("metrics") or {}
    overall_names = [k for k in ("locomo_f1", "token_f1", "exact_match") if metrics.get(k) is not None]
    if overall_names:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vals = [metrics[k] for k in overall_names]
        ax.bar(overall_names, vals, color=["#54A24B", "#F58518", "#E45756"][: len(vals)])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Overall metrics (n={summary.get('number_of_questions')})")
        ax.set_ylabel("Score")
        fig.tight_layout()
        p = out_dir / "overall_metrics.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    return written


def write_run_report(
    run_dir: Path,
    prediction_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, str]:
    """Write the full audit package for one run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions_jsonl": str(run_dir / "predictions.jsonl"),
        "predictions_csv": str(run_dir / "predictions.csv"),
        "metrics_json": str(run_dir / "metrics.json"),
        "meta_json": str(run_dir / "run_meta.json"),
        "category_csv": str(run_dir / "metrics_by_category.csv"),
        "plots_dir": str(run_dir / "plots"),
    }

    # Attach per-row scores for CSV
    from .metrics import score_row

    scored = []
    for r in prediction_rows:
        s = score_row(r["predicted_answer"], r["reference_answer"], int(r["category"]))
        scored.append({**r, **s})

    write_jsonl(Path(paths["predictions_jsonl"]), prediction_rows)
    write_predictions_csv(Path(paths["predictions_csv"]), scored)
    write_json(Path(paths["metrics_json"]), summary)
    write_json(Path(paths["meta_json"]), meta)
    write_category_csv(Path(paths["category_csv"]), summary.get("by_category") or {})
    plots = make_plots(summary, Path(paths["plots_dir"]))
    paths["plots"] = [str(p) for p in plots]
    return paths
