"""Minimal training entrypoint stub for knowledge distillation on LoCoMo QA."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.locomo import iter_qa_examples, load_locomo
from src.logging_csv import append_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)

    raw_path = Path(cfg["data"]["raw_path"])
    if not raw_path.exists():
        raise SystemExit(
            f"Missing {raw_path}. Run: python scripts/fetch_locomo.py"
        )

    samples = load_locomo(raw_path)
    holdout = cfg["data"].get("holdout_sample_ids") or []
    examples = list(
        iter_qa_examples(samples, holdout_sample_ids=holdout, split="train")
    )
    print(f"Loaded {len(examples)} train QA examples from {len(samples)} conversations")

    run_name = cfg["train"]["run_name"]
    out_dir = Path(cfg["train"]["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = cfg["train"].get("log_csv", "logs/train.csv")

    teacher = cfg["model"].get("teacher")
    student = cfg["model"].get("student")
    if not teacher or not student:
        print(
            "Stub mode: set model.teacher and model.student in the YAML to train for real."
        )
        append_csv(
            log_csv,
            {
                "run_name": run_name,
                "step": 0,
                "loss": None,
                "n_examples": len(examples),
                "note": "stub_no_models",
            },
        )
        print(f"Logged stub row -> {log_csv}")
        print(f"Run dir: {out_dir}")
        return

    # Placeholder for the real loop: load models, tokenize context+question,
    # teacher soft labels, student KD+CE via src.distill.losses.kd_ce_loss.
    raise NotImplementedError(
        "Wire teacher/student forward passes here once model IDs are set."
    )


if __name__ == "__main__":
    main()
