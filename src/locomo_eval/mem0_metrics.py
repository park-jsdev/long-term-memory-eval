"""Mem0-style lexical metrics (F1, BLEU-1) used next to the LLM autorater.

These are **not** LoCoMo category F1 (`src.metrics.locomo_qa`) and not SPEC
token F1 (`metrics.token_f1`). Mem0 reports F1 / BLEU-1 / LLM-as-a-Judge on
LoCoMo categories 1–4 (skips adversarial category 5).

Tokenization is a small local unigram splitter so tests do not need NLTK
``punkt``. Scores are on [0, 1]; literature tables multiply by 100.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def simple_tokenize(text: str) -> list[str]:
    """Lowercase, split punctuation into spaces, then whitespace-split."""
    text = str(text).lower()
    out = []
    buf: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"'", "-"}:
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def mem0_f1(prediction: str, reference: str) -> float:
    """Set-overlap token F1 (Mem0 ``calculate_metrics`` / ``simple_tokenize``)."""
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mem0_bleu1(prediction: str, reference: str) -> float:
    """Unigram BLEU with brevity penalty.

    Close to Mem0's BLEU-1 (nltk ``sentence_bleu`` weights (1,0,0,0)) without
    pulling in ``punkt``. Smoothing: empty prediction → 0.
    """
    pred = simple_tokenize(prediction)
    ref = simple_tokenize(reference)
    if not pred:
        return 0.0
    if not ref:
        return 0.0
    ref_counts = Counter(ref)
    overlap = 0
    for tok in pred:
        if ref_counts[tok] > 0:
            overlap += 1
            ref_counts[tok] -= 1
    precision = overlap / len(pred)
    if len(pred) < len(ref):
        bp = math.exp(1.0 - len(ref) / len(pred))
    else:
        bp = 1.0
    return bp * precision


def score_mem0_lexical(prediction: str, reference: str) -> dict[str, float]:
    return {
        "mem0_f1": mem0_f1(prediction, reference),
        "mem0_bleu1": mem0_bleu1(prediction, reference),
    }


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation percentile. ``p`` is 0–100 (e.g. 50, 95)."""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize_latencies(values: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 4),
        "p50": round(percentile(xs, 50) or 0.0, 4),
        "p95": round(percentile(xs, 95) or 0.0, 4),
    }
