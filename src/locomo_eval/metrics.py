"""Scoring helpers for baseline reports.

Two layers:
1. Simple token EM / F1 (SPEC_v1) — easy to explain in writeups.
2. LoCoMo category-aware F1 (src.metrics.locomo_qa) — apples-to-apples with paper.

Called from run_locomo_pipeline_with_memory_config after predictions are stored,
and from offline_evaluate.py to rescore JSONL with no API. This is not an LLM
autorater; a model-as-judge path would be a separate module.
"""

from __future__ import annotations

import string
from collections import Counter
from typing import Any

import regex

from src.metrics.locomo_qa import (
    CATEGORY_NAMES,
    aggregate_by_category,
    eval_question_answering,
    normalize_answer as locomo_normalize,
    score_prediction as locomo_score,
)


def normalize_answer(s: str) -> str:
    """SPEC-style normalize (subset of LoCoMo; articles a/an/the only)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = regex.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    ref_toks = normalize_answer(reference).split()
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(pred_toks)
    recall = n / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def score_row(prediction: str, reference: str, category: int) -> dict[str, float]:
    return {
        "exact_match": exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "locomo_f1": float(locomo_score(prediction, reference, category)),
    }


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: dicts with predicted_answer, reference_answer, category."""
    if not rows:
        return {
            "number_of_questions": 0,
            "metrics": {
                "exact_match": None,
                "token_f1": None,
                "locomo_f1": None,
            },
            "by_category": {},
        }

    ems, f1s = [], []
    qa_for_locomo = []
    for r in rows:
        pred = str(r["predicted_answer"])
        ref = str(r["reference_answer"])
        cat = int(r["category"])
        ems.append(exact_match(pred, ref))
        f1s.append(token_f1(pred, ref))
        qa_for_locomo.append(
            {
                "answer": ref,
                "category": cat,
                "prediction": pred,
            }
        )

    locomo_scores = eval_question_answering(qa_for_locomo, eval_key="prediction")
    locomo_summary = aggregate_by_category(qa_for_locomo, locomo_scores)

    # Simple metrics by category (SPEC categories = LoCoMo ids)
    from collections import defaultdict

    cat_em: dict[int, list[float]] = defaultdict(list)
    cat_f1: dict[int, list[float]] = defaultdict(list)
    for r, em, f1 in zip(rows, ems, f1s):
        c = int(r["category"])
        cat_em[c].append(em)
        cat_f1[c].append(f1)

    by_category = {}
    for cat, name in CATEGORY_NAMES.items():
        n = len(cat_em.get(cat, []))
        by_category[name] = {
            "category_id": cat,
            "n": n,
            "exact_match": round(sum(cat_em[cat]) / n, 4) if n else None,
            "token_f1": round(sum(cat_f1[cat]) / n, 4) if n else None,
            "locomo_f1": locomo_summary["by_category"][name]["f1"],
        }

    n = len(rows)
    return {
        "number_of_questions": n,
        "metrics": {
            "exact_match": round(sum(ems) / n, 4),
            "token_f1": round(sum(f1s) / n, 4),
            "locomo_f1": locomo_summary["overall_f1"],
        },
        "by_category": by_category,
        "locomo_official": locomo_summary,
    }


# Re-export for tests / callers that want LoCoMo normalize
normalize_answer_locomo = locomo_normalize
