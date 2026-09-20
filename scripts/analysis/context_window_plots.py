"""Plots for context-window vs coverage (offline, legend outside)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.campaign_plots import (
    _apply_ylim,
    _place_legend_outside,
    _pyplot,
    _save,
    write_bar,
    write_grouped_bar,
    write_line,
)


def write_context_window_plots(
    plots_dir: Path,
    *,
    cell_table: pd.DataFrame,
    conversation_table: pd.DataFrame,
    year_table: pd.DataFrame,
) -> list[Path]:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    live = _live_cells(cell_table)

    path = write_window_fill(live, plots_dir / "window_utilization_full_context.png")
    if path:
        written.append(path)

    method_input = _method_input_table(live)
    path = write_bar(
        method_input,
        x_col="memory_method",
        metric="agent_input_tokens_mean",
        path=plots_dir / "input_tokens_by_method.png",
        title="Mean billed reader input tokens by memory method",
    )
    if path:
        written.append(path)

    if not conversation_table.empty and "agent_input_tokens_mean" in conversation_table.columns:
        conv = conversation_table.dropna(subset=["agent_input_tokens_mean"]).copy()
        if not conv.empty:
            path = write_bar(
                conv,
                x_col="conversation_id",
                metric="agent_input_tokens_mean",
                path=plots_dir / "input_tokens_by_conversation.png",
                title="Billed full-context input tokens by LoCoMo conversation",
            )
            if path:
                written.append(path)

    path = write_scatter(
        live,
        x_col="agent_input_tokens_mean",
        y_col="judge_score",
        hue_col="memory_method",
        path=plots_dir / "coverage_vs_judge_score.png",
        title="Mem0 J vs billed input tokens (coverage, not window fill)",
        xlabel="Mean reader input tokens",
        ylabel="Judge score J (category 5 excluded)",
    )
    if path:
        written.append(path)

    path = write_scatter(
        live,
        x_col="agent_input_tokens_mean",
        y_col="agent_latency_seconds_mean",
        hue_col="memory_method",
        path=plots_dir / "coverage_vs_latency.png",
        title="Reader generate latency vs billed input tokens",
        xlabel="Mean reader input tokens",
        ylabel="Reader generate latency (s)",
    )
    if path:
        written.append(path)

    if "usd_cell" in live.columns and live["usd_cell"].notna().any():
        path = write_scatter(
            live,
            x_col="agent_input_tokens_mean",
            y_col="usd_cell",
            hue_col="memory_method",
            path=plots_dir / "coverage_vs_usd.png",
            title="Pinned-USD cell cost vs billed input tokens",
            xlabel="Mean reader input tokens",
            ylabel="USD (list price, not an invoice)",
        )
        if path:
            written.append(path)

    if not year_table.empty and "window_utilization" in year_table.columns:
        yt = year_table.copy()
        if "hue" not in yt.columns:
            yt["hue"] = _year_hue(yt)
        path = write_line(
            yt,
            x_col="generation",
            hue_col="hue",
            metric="window_utilization",
            path=plots_dir / "year_window_utilization.png",
            title="Full-context window utilization by generation",
        )
        if path:
            written.append(path)
        if "judge_score_per_1k_input" in yt.columns:
            path = write_line(
                yt,
                x_col="generation",
                hue_col="hue",
                metric="judge_score_per_1k_input",
                path=plots_dir / "year_j_per_1k_input.png",
                title="Full-context J per 1k input tokens by generation",
            )
            if path:
                written.append(path)

    if "window_utilization" in live.columns and "judge_score" in live.columns:
        fc = live[live["memory_method"].astype(str) == "full_context"]
        if not fc.empty:
            path = write_scatter(
                fc,
                x_col="window_utilization",
                y_col="judge_score",
                hue_col="generation" if "generation" in fc.columns else "reader_model",
                path=plots_dir / "utilization_vs_j_full_context.png",
                title="Full-context J vs window utilization (idle window is not evidence)",
                xlabel="Window utilization (input / window)",
                ylabel="Judge score J (category 5 excluded)",
            )
            if path:
                written.append(path)

    if "judge_score_per_1k_input" in live.columns:
        path = write_grouped_bar(
            live,
            x_col="memory_method",
            hue_col="generation" if "generation" in live.columns else "pack",
            metric="judge_score_per_1k_input",
            path=plots_dir / "j_per_1k_input_by_method.png",
            title="J per 1k input tokens by memory method",
        )
        if path:
            written.append(path)

    return written


def write_window_fill(cell_table: pd.DataFrame, path: Path) -> Path | None:
    """Horizontal utilization bars for full_context cells (the fill-rate figure)."""
    plt = _pyplot()
    if plt is None or cell_table.empty:
        return None
    fc = cell_table[cell_table["memory_method"].astype(str) == "full_context"].copy()
    need = {"reader_model", "window_utilization", "agent_input_tokens_mean", "context_window"}
    if fc.empty or not need.issubset(fc.columns):
        return None
    fc = fc.dropna(subset=["window_utilization", "context_window"])
    if fc.empty:
        return None
    fc = fc.sort_values("window_utilization", ascending=True)
    labels = []
    for _, row in fc.iterrows():
        think = row.get("thinking") or ""
        extra = f" {think}" if think and str(think) != "nan" else ""
        labels.append(f"{row.get('reader_model')}{extra}")
    utils = [float(v) for v in fc["window_utilization"].tolist()]
    fig, ax = plt.subplots(figsize=(8.2, max(2.8, 0.42 * len(labels) + 1.4)))
    ax.barh(labels, utils, color="#4C78A8")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Window utilization (billed input / published window)")
    ax.set_title("Full-context fill of the model window")
    for i, (_, row) in enumerate(fc.iterrows()):
        inp = float(row["agent_input_tokens_mean"])
        win = float(row["context_window"])
        ax.text(
            min(utils[i] + 0.02, 0.98),
            i,
            f"{inp:,.0f} / {win:,.0f} ({100 * utils[i]:.1f}%)",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
    return Path(path)


def write_scatter(
    table: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    hue_col: str,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path | None:
    plt = _pyplot()
    if plt is None or table.empty:
        return None
    if x_col not in table.columns or y_col not in table.columns:
        return None
    work = table.dropna(subset=[x_col, y_col]).copy()
    if work.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    if hue_col in work.columns:
        hues = [str(v) for v in work[hue_col].drop_duplicates().tolist()]
        colors = _hue_colors(hues)
        for hue, color in zip(hues, colors):
            piece = work[work[hue_col].astype(str) == hue]
            ax.scatter(
                piece[x_col],
                piece[y_col],
                label=hue,
                color=color,
                s=42,
                alpha=0.85,
            )
        _place_legend_outside(ax)
    else:
        ax.scatter(work[x_col], work[y_col], color="#4C78A8", s=42)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ys = [float(v) for v in work[y_col].tolist() if pd.notna(v)]
    _apply_ylim(ax, y_col, ys)
    _save(fig, path)
    plt.close(fig)
    return Path(path)


def _live_cells(cell_table: pd.DataFrame) -> pd.DataFrame:
    if cell_table.empty:
        return cell_table
    if "result_source" not in cell_table.columns:
        return cell_table
    return cell_table[cell_table["result_source"].astype(str) != "paper"].copy()


def _method_input_table(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty or "memory_method" not in cells.columns:
        return pd.DataFrame()
    return (
        cells.groupby("memory_method", dropna=False, as_index=False)["agent_input_tokens_mean"]
        .mean()
        .sort_values("agent_input_tokens_mean", ascending=False)
    )


def _year_hue(table: pd.DataFrame) -> pd.Series:
    family = table["model_family"].astype(str) if "model_family" in table.columns else "model"
    thinking = table["thinking"].astype(str) if "thinking" in table.columns else ""
    source = table["result_source"].astype(str) if "result_source" in table.columns else ""
    return family + " × " + thinking + " (" + source + ")"


def _hue_colors(hues: list[str]) -> list[str]:
    palette = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2", "#B8B8B8")
    return [palette[i % len(palette)] for i in range(len(hues))]
