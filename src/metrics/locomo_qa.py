"""LoCoMo QA metrics — adapted from snap-research/locomo task_eval/evaluation.py.

Source: https://github.com/snap-research/locomo
Pinned commit: 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376
License: CC BY-NC 4.0

Kept close to upstream so F1 / category handling stays apples-to-apples.
"""

from __future__ import annotations

import string
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import regex
from nltk.stem import PorterStemmer

ps = PorterStemmer()

# Paper category order used in LoCoMo reporting.
CATEGORY_ORDER = [4, 1, 2, 3, 5]
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")

    def remove_articles(text: str) -> str:
        return regex.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1(prediction: str, ground_truth: str) -> float:
    """Multi-answer F1 used for LoCoMo category 1 (multi-hop)."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(
        np.mean(
            [
                max(f1_score(pred, gt) for pred in predictions)
                for gt in ground_truths
            ]
        )
    )


def score_prediction(prediction: str, answer: str, category: int) -> float:
    """Score one QA pair with LoCoMo category rules."""
    prediction = str(prediction)
    answer = str(answer)
    if category == 3:
        answer = answer.split(";")[0].strip()

    if category in (2, 3, 4):
        return float(f1_score(prediction, answer))
    if category == 1:
        return float(f1(prediction, answer))
    if category == 5:
        lower = prediction.lower()
        if "no information available" in lower or "not mentioned" in lower:
            return 1.0
        return 0.0
    raise ValueError(f"Unknown LoCoMo category: {category}")


def eval_question_answering(
    qas: list[dict[str, Any]],
    eval_key: str = "prediction",
) -> list[float]:
    """Mirror of upstream eval_question_answering (F1 path only)."""
    scores: list[float] = []
    for line in qas:
        output = line[eval_key]
        if isinstance(output, list):
            # Upstream accepts list answers in some paths; take first string.
            output = output[0] if output else ""
        scores.append(score_prediction(str(output), str(line["answer"]), int(line["category"])))
    return scores


def aggregate_by_category(
    qas: list[dict[str, Any]],
    scores: list[float],
) -> dict[str, Any]:
    """Mean F1 overall and per category (LoCoMo reporting order)."""
    if len(qas) != len(scores):
        raise ValueError("qas and scores length mismatch")

    totals: dict[int, int] = defaultdict(int)
    sums: dict[int, float] = defaultdict(float)
    for qa, score in zip(qas, scores):
        cat = int(qa["category"])
        totals[cat] += 1
        sums[cat] += score

    by_category = {}
    for cat in CATEGORY_ORDER:
        n = totals.get(cat, 0)
        by_category[CATEGORY_NAMES[cat]] = {
            "category_id": cat,
            "n": n,
            "f1": round(sums[cat] / n, 3) if n else None,
        }

    n_all = len(scores)
    overall = round(float(sum(scores)) / n_all, 3) if n_all else None
    return {"overall_f1": overall, "n": n_all, "by_category": by_category}
