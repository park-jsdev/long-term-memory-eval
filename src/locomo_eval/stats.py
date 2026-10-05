"""Mean, CI, and paired tests for multi-seed and frozen-reader evaluation.

Used by ``scripts.analysis.aggregate_seeds`` and ``scripts.compare_full_runs``.
No scipy: numpy only. These are audit statistics on *already scored* rows,
not a reason to re-call an LLM.

Paper protocol (Chhikara et al., arXiv:2504.19413): J is the mean of 10
independent judge runs, reported with ±1 standard deviation. Local packs
also attach a 95% t-interval so n≠10 is still interpretable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Two-sided 0.975 Student-t critical values by df. n=10 → df=9 → 2.262.
_T_0975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    24: 2.064,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def t_critical_975(df: int) -> float:
    """Two-sided 95% t critical value. Normal 1.96 for large df."""
    if df < 1:
        return float("nan")
    if df in _T_0975:
        return _T_0975[df]
    for key in sorted(_T_0975):
        if df <= key:
            return _T_0975[key]
    return 1.96


def mean_std(values: list[float] | np.ndarray) -> tuple[float | None, float | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None, None
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    return mean, float(np.std(arr, ddof=1))


def mean_ci95(values: list[float] | np.ndarray) -> dict[str, float | None]:
    """Mean ± sample std, plus a 95% t-interval of the mean."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    n = int(arr.size)
    mean, std = mean_std(arr)
    if mean is None or n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    if n == 1 or std is None or std == 0.0:
        return {
            "n": n,
            "mean": round(mean, 6),
            "std": 0.0 if std is not None else None,
            "ci95_low": round(mean, 6),
            "ci95_high": round(mean, 6),
        }
    se = std / math.sqrt(n)
    tcrit = t_critical_975(n - 1)
    half = tcrit * se
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95_low": round(mean - half, 6),
        "ci95_high": round(mean + half, 6),
    }


def bootstrap_ci95(
    values: list[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float | None]:
    """Percentile bootstrap 95% CI of the mean. Seeded for audit."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "n": 1,
            "mean": round(mean, 6),
            "ci95_low": round(mean, 6),
            "ci95_high": round(mean, 6),
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "n": n,
        "mean": round(mean, 6),
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
    }


def wilcoxon_signed_rank(deltas: list[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank on paired deltas (B − A).

    Zero deltas are dropped (Pratt). p-value is the large-sample normal
    approximation with continuity correction. For n<6 the p-value is omitted.
    """
    arr = np.asarray(list(deltas), dtype=np.float64)
    nonzero = arr[arr != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return {"n": 0, "W": None, "z": None, "p_two_sided": None, "note": "all zeros"}
    order = np.argsort(np.abs(nonzero))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    # Average ties.
    abs_sorted = np.abs(nonzero)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_sorted[j + 1] == abs_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    w_pos = float(np.sum(ranks[nonzero > 0]))
    w_neg = float(np.sum(ranks[nonzero < 0]))
    w = min(w_pos, w_neg)
    expected = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return {"n": n, "W": w, "z": None, "p_two_sided": None}
    z = (w - expected) / math.sqrt(var)
    # continuity correction toward the mean
    if w > expected:
        z = (w - 0.5 - expected) / math.sqrt(var)
    elif w < expected:
        z = (w + 0.5 - expected) / math.sqrt(var)
    else:
        z = 0.0
    p = 2.0 * (1.0 - _phi(abs(z)))
    out: dict[str, Any] = {
        "n": n,
        "W": round(w, 4),
        "z": round(float(z), 4),
        "p_two_sided": round(min(1.0, p), 6),
        "mean_delta": round(float(np.mean(arr)), 6),
    }
    if n < 6:
        out["note"] = "n<6; p-value is a normal approximation and is unreliable"
    return out


def mcnemar_test(a_correct: list[bool], b_correct: list[bool]) -> dict[str, Any]:
    """McNemar on paired binary labels (e.g. autorater CORRECT/WRONG).

    Continuity-corrected chi-square (Edwards). Two-sided.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("Paired binary series must be the same length.")
    n01 = n10 = n11 = n00 = 0
    for a, b in zip(a_correct, b_correct):
        if a and b:
            n11 += 1
        elif a and not b:
            n01 += 1
        elif (not a) and b:
            n10 += 1
        else:
            n00 += 1
    discordant = n01 + n10
    if discordant == 0:
        return {
            "n": len(a_correct),
            "n11": n11,
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "chi2": 0.0,
            "p_two_sided": 1.0,
        }
    chi2 = (abs(n01 - n10) - 1) ** 2 / discordant
    # chi-square df=1 survival: erfc(sqrt(x/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "n": len(a_correct),
        "n11": n11,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "chi2": round(chi2, 4),
        "p_two_sided": round(min(1.0, p), 6),
    }


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def summarize_seed_metrics(
    metric_rows: list[dict[str, Any]],
    keys: tuple[str, ...] = ("llm_judge_pct", "mem0_f1_pct", "mem0_bleu1_pct"),
    *,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregate one scalar per seed (paper: mean ± std of J)."""
    out: dict[str, Any] = {"n_seeds": len(metric_rows)}
    for key in keys:
        values = []
        for row in metric_rows:
            val = row.get(key)
            if val is None and isinstance(row.get("metrics"), dict):
                val = row["metrics"].get(key)
            if val is not None:
                values.append(float(val))
        stats = mean_ci95(values)
        stats["bootstrap"] = bootstrap_ci95(values, seed=bootstrap_seed)
        out[key] = stats
    return out
