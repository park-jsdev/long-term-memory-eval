"""Shared matplotlib writers for offline LoCoMo F1 comparisons.

No-op (return []) if matplotlib is missing. Callers still write CSV/JSON.
Condition colors are fixed so overall bars, boxplots, and histograms match.
"""

from __future__ import annotations

from pathlib import Path

# A = blue, B = orange. Used whenever two experiment conditions share a figure.
CONDITION_COLOR_A = "#4C78A8"
CONDITION_COLOR_B = "#F58518"


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def write_locomo_f1_overall_bars(
    mean_a: float,
    mean_b: float,
    *,
    label_a: str,
    label_b: str,
    path: Path,
) -> Path | None:
    """Overall LoCoMo F1: x-axis is the metric, colors are the two conditions.

    Extra metrics (token_f1, exact_match) can be added later as more x ticks
    with the same two condition colors.
    """
    plt = _pyplot()
    if plt is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    x = [0]
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        [mean_a],
        width,
        label=label_a,
        color=CONDITION_COLOR_A,
    )
    ax.bar(
        [i + width / 2 for i in x],
        [mean_b],
        width,
        label=label_b,
        color=CONDITION_COLOR_B,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["locomo_f1"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall LoCoMo F1 by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_locomo_f1_boxplot(
    scores_a: list[float],
    scores_b: list[float],
    *,
    label_a: str,
    label_b: str,
    path: Path,
) -> Path | None:
    """Tukey boxplot (whiskers at 1.5 IQR) of per-question LoCoMo F1 for two conditions."""
    plt = _pyplot()
    if plt is None or not scores_a or not scores_b:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    data = [scores_a, scores_b]
    names = [label_a, label_b]
    try:
        ax.boxplot(data, tick_labels=names, whis=1.5, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=names, whis=1.5, showfliers=True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("LoCoMo F1 (per question)")
    ax.set_title("LoCoMo F1 distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_locomo_f1_histograms(
    scores_a: list[float],
    scores_b: list[float],
    *,
    label_a: str,
    label_b: str,
    path: Path,
    bins: int = 10,
) -> Path | None:
    """Side-by-side histograms of per-question LoCoMo F1 (one panel per condition)."""
    plt = _pyplot()
    if plt is None or not scores_a or not scores_b:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_bins = max(3, min(bins, max(len(scores_a), len(scores_b))))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    for ax, scores, label, color in (
        (axes[0], scores_a, label_a, CONDITION_COLOR_A),
        (axes[1], scores_b, label_b, CONDITION_COLOR_B),
    ):
        ax.hist(scores, bins=n_bins, range=(0.0, 1.0), color=color, edgecolor="white")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("LoCoMo F1")
        ax.set_title(label)
    axes[0].set_ylabel("Count")
    fig.suptitle("LoCoMo F1 by condition", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
