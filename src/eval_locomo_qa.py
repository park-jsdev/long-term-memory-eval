"""Score LoCoMo QA predictions with apples-to-apples official metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.locomo import load_locomo, load_predictions
from src.logging_csv import append_csv
from src.metrics.locomo_qa import aggregate_by_category, eval_question_answering


def flatten_qa(
    pred_samples: list[dict],
    gold_by_id: dict[str, dict],
    prediction_key: str,
) -> list[dict]:
    """Build a flat QA list with gold answer/category + model prediction."""
    flat = []
    for sample in pred_samples:
        sid = sample["sample_id"]
        gold = gold_by_id[sid]
        gold_qa = gold["qa"]
        pred_qa = sample["qa"]
        if len(pred_qa) != len(gold_qa):
            raise ValueError(
                f"{sid}: prediction qa len {len(pred_qa)} != gold {len(gold_qa)}"
            )
        for g, p in zip(gold_qa, pred_qa):
            if prediction_key not in p:
                raise KeyError(
                    f"{sid}: missing '{prediction_key}' on a QA item. "
                    f"keys={list(p.keys())}"
                )
            flat.append(
                {
                    "sample_id": sid,
                    "question": g["question"],
                    "answer": g["answer"],
                    "category": g["category"],
                    "evidence": g.get("evidence", []),
                    prediction_key: p[prediction_key],
                }
            )
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/processed_export.yaml")
    parser.add_argument("--predictions", default=None, help="Override eval.predictions_path")
    parser.add_argument("--prediction-key", default=None)
    parser.add_argument("--run-name", default="eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_path = args.predictions or cfg["eval"].get("predictions_path")
    if not pred_path:
        raise SystemExit(
            "Set eval.predictions_path in the YAML or pass --predictions. "
            "Expected a LoCoMo-style JSON list of samples with qa[].<prediction_key>."
        )
    prediction_key = args.prediction_key or cfg["eval"].get("prediction_key", "prediction")

    gold = load_locomo(cfg["data"]["raw_path"])
    gold_by_id = {s["sample_id"]: s for s in gold}
    preds = load_predictions(pred_path)

    flat = flatten_qa(preds, gold_by_id, prediction_key)
    scores = eval_question_answering(flat, eval_key=prediction_key)
    summary = aggregate_by_category(flat, scores)

    # Per-example CSV
    log_csv = cfg["eval"].get("log_csv", "logs/eval_qa.csv")
    for row, score in zip(flat, scores):
        append_csv(
            log_csv,
            {
                "run_name": args.run_name,
                "sample_id": row["sample_id"],
                "category": row["category"],
                "question": row["question"],
                "answer": row["answer"],
                "prediction": row[prediction_key],
                "f1": round(score, 4),
            },
        )

    out_dir = Path(cfg["eval"].get("output_dir") or "experiments")
    out_json = out_dir / args.run_name / "locomo_qa_scores.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Per-example log: {log_csv}")
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()
