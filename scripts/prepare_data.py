"""Flatten LoCoMo into JSONL + CSV for inspection (and optional KD prep)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.locomo import iter_qa_examples, load_locomo

# Full context blows CSV size (multi‑100MB). Keep a short audit snippet.
CONTEXT_PREVIEW_CHARS = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="all", choices=["all", "train", "eval"])
    parser.add_argument("--out", default=None, help="Output JSONL path")
    parser.add_argument(
        "--csv",
        default=None,
        help="Output CSV path (default: same stem as JSONL with .csv)",
    )
    parser.add_argument(
        "--jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write full JSONL with context (default: true)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = load_locomo(cfg["data"]["raw_path"])
    holdout = cfg["data"].get("holdout_sample_ids") or []

    out_jsonl = Path(args.out or Path(cfg["data"]["processed_dir"]) / f"qa_{args.split}.jsonl")
    out_csv = Path(args.csv or out_jsonl.with_suffix(".csv"))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "qa_index",
        "question_id",
        "category",
        "category_name",
        "question",
        "answer",
        "evidence",
        "context_chars",
        "context_preview",
    ]

    n = 0
    f_jsonl = out_jsonl.open("w", encoding="utf-8") if args.jsonl else None
    try:
        with out_csv.open("w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for ex in iter_qa_examples(samples, holdout_sample_ids=holdout, split=args.split):
                qid = f"{ex['sample_id']}-q-{ex['qa_index']}"
                ctx = ex.get("context") or ""
                row = {
                    "sample_id": ex["sample_id"],
                    "qa_index": ex["qa_index"],
                    "question_id": qid,
                    "category": ex["category"],
                    "category_name": ex.get("category_name", ""),
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "evidence": "|".join(ex.get("evidence") or []),
                    "context_chars": len(ctx),
                    "context_preview": ctx[:CONTEXT_PREVIEW_CHARS].replace("\n", " "),
                }
                writer.writerow(row)
                if f_jsonl is not None:
                    full = {**ex, "question_id": qid}
                    f_jsonl.write(json.dumps(full, ensure_ascii=False) + "\n")
                n += 1
    finally:
        if f_jsonl is not None:
            f_jsonl.close()

    print(f"Wrote {n} examples -> {out_csv}")
    if args.jsonl:
        print(f"Wrote {n} examples -> {out_jsonl}")


if __name__ == "__main__":
    main()
