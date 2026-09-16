"""Reusable campaign/experiment bar writers (offline, no LLM).

YAML recipes choose kind / x / hue / y; this module draws the figure.
Legend is always outside the axes. Do not fork this per campaign.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.campaign_tables import GENERATION_AXIS

FAMILY_COLORS = {
    "OpenAI": "#4C78A8",
    "Anthropic": "#F58518",
    "DeepSeek": "#54A24B",
}
FALLBACK_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2")


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _colors(labels: list[str]) -> list[str]:
    out: list[str] = []
    for i, label in enumerate(labels):
        out.append(FAMILY_COLORS.get(str(label), FALLBACK_COLORS[i % len(FALLBACK_COLORS)]))
    return out


def _place_legend_outside(ax) -> None:
    """Put the legend to the right of the axes so it cannot cover bars."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
    )


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _axis_title(col: str) -> str:
    if col == "question_category":
        return "LoCoMo question category"
    if col == "generation":
        return "Generation"
    if col == "thinking":
        return "Thinking"
    if col == "agent_reasoning_tokens":
        return "Reader reasoning tokens"
    if col == "teacher_reasoning_tokens":
        return "Teacher reasoning tokens"
    if col == "agent_latency_seconds":
        return "Reader latency (s)"
    if col == "teacher_latency_seconds":
        return "Teacher latency (s)"
    return col.replace("_", " ")


def write_metrics_grouped_bar(
    table: pd.DataFrame,
    *,
    group_col: str,
    metrics: list[str],
    path: Path,
    title: str,
) -> Path | None:
    """x = metric names; one bar group per ``group_col`` value."""
    plt = _pyplot()
    if plt is None or table.empty:
        return None
    labels = [str(v) for v in table[group_col].tolist()]
    metric_cols = [m for m in metrics if m in table.columns]
    if not metric_cols:
        return None
    x = list(range(len(metric_cols)))
    n = max(len(labels), 1)
    width = min(0.8 / n, 0.25)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = _colors(labels)
    for i, (label, color) in enumerate(zip(labels, colors)):
        ys = [
            float(table.iloc[i][m]) if pd.notna(table.iloc[i][m]) else float("nan")
            for m in metric_cols
        ]
        offset = (i - (n - 1) / 2) * width
        ax.bar([xi + offset for xi in x], ys, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, rotation=20, ha="right")
    ymax = 1.05
    if any(m == "agent_latency_seconds" or str(m).startswith("usd") for m in metric_cols):
        ymax = None
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.set_ylabel("USD" if all(str(m).startswith("usd") for m in metric_cols) else "Score")
    ax.set_title(title)
    _place_legend_outside(ax)
    _save(fig, path)
    plt.close(fig)
    return Path(path)


def write_grouped_bar(
    table: pd.DataFrame,
    *,
    x_col: str,
    hue_col: str,
    metric: str,
    path: Path,
    title: str,
) -> Path | None:
    plt = _pyplot()
    if plt is None or table.empty or metric not in table.columns:
        return None
    xs = [str(v) for v in table[x_col].drop_duplicates().tolist()]
    if x_col == "generation":
        known = [year for year in GENERATION_AXIS if year in xs]
        xs = known + [year for year in xs if year not in GENERATION_AXIS]
    hues = [str(v) for v in table[hue_col].drop_duplicates().tolist()]
    pivot = table.pivot_table(index=x_col, columns=hue_col, values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=xs, columns=hues)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = list(range(len(xs)))
    n = max(len(hues), 1)
    width = min(0.8 / n, 0.25)
    colors = _colors(hues)
    for i, (hue, color) in enumerate(zip(hues, colors)):
        ys = [
            float(v) if pd.notna(v) else float("nan")
            for v in pivot[pivot.columns[i]].tolist()
        ]
        offset = (i - (n - 1) / 2) * width
        ax.bar([xi + offset for xi in x], ys, width, label=hue, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, rotation=20, ha="right")
    ax.set_xlabel(_axis_title(x_col))
    if metric != "agent_latency_seconds" and not str(metric).startswith("usd"):
        ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.set_title(title)
    _place_legend_outside(ax)
    _save(fig, path)
    plt.close(fig)
    return Path(path)


def write_bar(
    table: pd.DataFrame,
    *,
    x_col: str,
    metric: str,
    path: Path,
    title: str,
) -> Path | None:
    plt = _pyplot()
    if plt is None or table.empty or metric not in table.columns:
        return None
    labels = [str(v) for v in table[x_col].tolist()]
    ys = [float(v) if pd.notna(v) else float("nan") for v in table[metric].tolist()]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(labels, ys, color=_colors(labels))
    ax.set_xlabel(_axis_title(x_col))
    ax.set_ylabel(metric)
    ax.set_title(title)
    if metric != "agent_latency_seconds" and not str(metric).startswith("usd"):
        ax.set_ylim(0, 1.05)
    fig.autofmt_xdate(rotation=20)
    _save(fig, path)
    plt.close(fig)
    return Path(path)
