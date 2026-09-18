"""Reusable campaign/experiment bar and line writers (offline, no LLM).

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
GENERATION_COLORS = {
    "2024": "#B8B8B8",
    "2025": "#4C78A8",
    "2026": "#F58518",
}
THINKING_COLORS = {
    "off": "#B8B8B8",
    "on": "#4C78A8",
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
        key = str(label)
        think = THINKING_COLORS.get(key.strip().lower())
        if think:
            out.append(think)
            continue
        year = GENERATION_COLORS.get(key)
        if year:
            out.append(year)
            continue
        out.append(FAMILY_COLORS.get(key, FALLBACK_COLORS[i % len(FALLBACK_COLORS)]))
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
    if col.endswith("_p50"):
        return f"{_axis_title(col[:-4])} p50"
    if col.endswith("_p95"):
        return f"{_axis_title(col[:-4])} p95"
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
        return "Reader generate latency (s)"
    if col == "teacher_latency_seconds":
        return "Teacher latency (s)"
    if col == "search_latency_seconds":
        return "Search latency (s)"
    if col == "total_latency_seconds":
        return "Total latency (s)"
    return col.replace("_", " ")


def unbounded_metric(metric: str) -> bool:
    """Latency, token counts, and USD — not 0–1 scores (``token_f1`` stays a score)."""
    name = str(metric)
    if name.startswith("usd"):
        return True
    if "latency" in name or name.endswith("_seconds"):
        return True
    if name.endswith("_tokens") or "reasoning_tokens" in name:
        return True
    if "_per_" in name:
        return True
    return False


def _apply_ylim(ax, metric: str, values: list[float]) -> None:
    if unbounded_metric(metric):
        finite = [float(v) for v in values if v is not None and pd.notna(v)]
        ymax = max(finite) if finite else 1.0
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
        return
    ax.set_ylim(0, 1.05)


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
    flat = []
    for i in range(len(labels)):
        for m in metric_cols:
            val = table.iloc[i][m]
            if pd.notna(val):
                flat.append(float(val))
    if any(unbounded_metric(m) for m in metric_cols):
        ymax = max(flat) if flat else 1.0
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
        ax.set_ylabel(
            "USD" if all(str(m).startswith("usd") for m in metric_cols) else "Value"
        )
    else:
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
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
    ys_flat = [float(v) for v in pivot.to_numpy().ravel() if pd.notna(v)]
    _apply_ylim(ax, metric, ys_flat)
    ax.set_ylabel(_axis_title(metric))
    ax.set_title(title)
    _place_legend_outside(ax)
    _save(fig, path)
    plt.close(fig)
    return Path(path)


def write_line(
    table: pd.DataFrame,
    *,
    x_col: str,
    hue_col: str,
    metric: str,
    path: Path,
    title: str,
) -> Path | None:
    """Year (or other x) time series; one polyline per hue / joined condition."""
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
    markers = ("o", "s", "D", "^", "v", "P", "X", "*")
    ys_flat: list[float] = []
    for i, hue in enumerate(hues):
        ys = [
            float(v) if pd.notna(v) else float("nan")
            for v in pivot[pivot.columns[i]].tolist()
        ]
        ys_flat.extend(v for v in ys if v == v)
        ax.plot(
            x,
            ys,
            color=_series_color(hue, i),
            marker=markers[i % len(markers)],
            linestyle=_line_style(hue),
            linewidth=1.8,
            markersize=6,
            label=hue,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xs, rotation=20, ha="right")
    ax.set_xlabel(_axis_title(x_col))
    _apply_ylim(ax, metric, ys_flat)
    ax.set_ylabel(_axis_title(metric))
    ax.set_title(title)
    _place_legend_outside(ax)
    _save(fig, path)
    plt.close(fig)
    return Path(path)


def _series_color(label: str, index: int) -> str:
    lower = str(label).lower()
    for family, color in FAMILY_COLORS.items():
        if family.lower() in lower:
            return color
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def _line_style(label: str) -> str:
    key = str(label).strip().lower()
    if key == "off" or key.endswith("off") or " × off" in key:
        return "--"
    return "-"


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
    ax.set_ylabel(_axis_title(metric))
    ax.set_title(title)
    _apply_ylim(ax, metric, ys)
    fig.autofmt_xdate(rotation=20)
    _save(fig, path)
    plt.close(fig)
    return Path(path)
