"""Flatten LoCoMo into a simple JSONL of QA examples for distillation / inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.locomo import iter_qa_examples, load_locomo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="all", choices=["all", "train", "eval"])
    parser.add_argument("--out", default=None, help="Output JSONL path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = load_locomo(cfg["data"]["raw_path"])
    holdout = cfg["data"].get("holdout_sample_ids") or []

    out = Path(args.out or Path(cfg["data"]["processed_dir"]) / f"qa_{args.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for ex in iter_qa_examples(samples, holdout_sample_ids=holdout, split=args.split):
            # Drop huge context duplication option: keep context for train stubs.
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} examples -> {out}")


if __name__ == "__main__":
    main()
