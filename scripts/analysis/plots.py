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
        import matplotlib

        # Reports are file artifacts, never interactive windows. Agg avoids
        # Tk dependencies in CI, servers, and minimal Windows Python installs.
        matplotlib.use("Agg", force=True)
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


def write_histogram(
    values: list[float],
    *,
    path: Path,
    xlabel: str,
    title: str,
    bins: int = 10,
    value_range: tuple[float, float] | None = None,
    color: str = CONDITION_COLOR_A,
) -> Path | None:
    """Single-series histogram. No-op if matplotlib missing or values empty."""
    plt = _pyplot()
    if plt is None or not values:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_bins = max(2, min(bins, max(len(values), 2)))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    kwargs: dict = {"bins": n_bins, "color": color, "edgecolor": "white"}
    if value_range is not None:
        kwargs["range"] = value_range
    ax.hist(values, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_count_bars(
    counts: dict[str, int],
    *,
    path: Path,
    ylabel: str,
    title: str,
) -> Path | None:
    """Bar plot for discrete category counts (for example CORRECT / WRONG)."""
    plt = _pyplot()
    if plt is None or not counts:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(counts)
    values = [int(counts[label]) for label in labels]
    colors = ["#54A24B" if label.upper() == "CORRECT" else "#E45756" for label in labels]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values + [1]) * 1.15)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_boxplot(
    series: dict[str, list[float]],
    *,
    path: Path,
    ylabel: str,
    title: str,
) -> Path | None:
    """Tukey boxplot (whiskers 1.5 IQR) for one or more named series."""
    plt = _pyplot()
    names = [n for n, vals in series.items() if vals]
    data = [series[n] for n in names]
    if plt is None or not data:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.4 * len(names) + 2), 4.2))
    try:
        ax.boxplot(data, tick_labels=names, whis=1.5, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=names, whis=1.5, showfliers=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_grouped_bars(
    labels: list[str],
    series: dict[str, list[float | None]],
    *,
    path: Path,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = (0, 105.0),
) -> Path | None:
    """Grouped bars. ``series`` maps legend name → y values aligned with labels."""
    plt = _pyplot()
    if plt is None or not labels or not series:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(series.keys())
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * n + 2), 4.4))
    x = list(range(n))
    width = 0.8 / max(len(names), 1)
    colors = [CONDITION_COLOR_A, CONDITION_COLOR_B, "#54A24B", "#E45756", "#72B7B2"]
    for i, name in enumerate(names):
        vals = [v if v is not None else 0.0 for v in series[name]]
        ax.bar(
            [xi + i * width for xi in x],
            vals,
            width,
            label=name,
            color=colors[i % len(colors)],
        )
    ax.set_xticks([xi + width * (len(names) - 1) / 2 for xi in x])
    ax.set_xticklabels(labels, rotation=25, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_latency_p50_p95_bars(
    methods: list[str],
    p50: list[float | None],
    p95: list[float | None],
    *,
    path: Path,
    title: str = "Total latency p50 / p95",
    ylabel: str = "Seconds",
) -> Path | None:
    """Side-by-side p50/p95 latency bars (Mem0 Table 2 style)."""
    plt = _pyplot()
    if plt is None or not methods:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(methods) + 2), 4.4))
    x = list(range(len(methods)))
    width = 0.35
    p50_v = [v if v is not None else 0.0 for v in p50]
    p95_v = [v if v is not None else 0.0 for v in p95]
    ax.bar([xi - width / 2 for xi in x], p50_v, width, label="p50", color=CONDITION_COLOR_A)
    ax.bar([xi + width / 2 for xi in x], p95_v, width, label="p95", color=CONDITION_COLOR_B)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_paper_vs_local_bars(
    labels: list[str],
    paper: list[float | None],
    local: list[float | None],
    *,
    path: Path,
    ylabel: str = "J (%)",
    title: str = "LLM-as-a-Judge vs Mem0 paper (Table 2)",
    paper_label: str = "Mem0 paper",
    local_label: str = "Local (best seed)",
) -> Path | None:
    """Grouped paper vs local bars. Missing local scores omit the second bar."""
    plt = _pyplot()
    if plt is None or not labels:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * n + 2.5), 4.8))
    x = list(range(n))
    width = 0.36
    paper_x, paper_y, local_x, local_y = [], [], [], []
    for i, (p_val, l_val) in enumerate(zip(paper, local)):
        if p_val is not None:
            paper_x.append(i - width / 2)
            paper_y.append(p_val)
        if l_val is not None:
            local_x.append(i + width / 2)
            local_y.append(l_val)
    if paper_x:
        paper_bars = ax.bar(
            paper_x, paper_y, width, label=paper_label, color=CONDITION_COLOR_A
        )
        for bar, value in zip(paper_bars, paper_y):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    if local_x:
        local_bars = ax.bar(
            local_x, local_y, width, label=local_label, color=CONDITION_COLOR_B
        )
        for bar, value in zip(local_bars, local_y):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ymax = max([v for v in paper + local if v is not None] + [0.0])
    ax.set_ylim(0, max(105.0, ymax * 1.12))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
