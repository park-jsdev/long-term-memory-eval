"""Run campaign/experiment analyses from YAML recipes (no LLM).

Tables and figures come from ``scripts.analysis.campaign_tables`` /
``scripts.analysis.campaign_plots``. This module only loads packs and writes
the report directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from scripts.analysis.campaign_plots import (
    _axis_title,
    write_bar,
    write_grouped_bar,
    write_line,
    write_metrics_grouped_bar,
)
from scripts.analysis.campaign_tables import (
    annotate_generation,
    annotate_mem0_latency,
    annotate_model_family,
    annotate_thinking,
    attach_run_judge_score,
    join_search_latency,
    mean_table,
    sort_year_family_table,
)
from src.memorybench.analysis.cost import CostReport, render_cost
from scripts.analysis.campaign_insights import attach_cell_cost, render_insight
from src.memorybench.analysis.load_campaign import (
    AnalysisSpec,
    CampaignConfig,
    ExperimentAnalysisRef,
    InsightSpec,
    PlotSpec,
    load_campaign_yaml,
)

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class AnalysisResult:
    spec: AnalysisSpec
    table: pd.DataFrame
    csv_path: Path | None = None
    plot_paths: list[Path] = field(default_factory=list)
    skipped: str | None = None


@dataclass
class ReportResult:
    scope: str
    out_dir: Path
    results: list[AnalysisResult]
    missing_packs: list[str] = field(default_factory=list)
    cost: CostReport | None = None


def load_pack(root: Path, ref: ExperimentAnalysisRef) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    pack = root / ref.pack if not ref.pack.is_absolute() else ref.pack
    agg = pack / "aggregate"
    examples_path = agg / "examples.parquet"
    runs_path = agg / "runs.parquet"
    if not examples_path.is_file() and not runs_path.is_file():
        return None
    examples = pd.read_parquet(examples_path) if examples_path.is_file() else pd.DataFrame()
    runs = pd.read_parquet(runs_path) if runs_path.is_file() else pd.DataFrame()
    examples = annotate_model_family(examples, ref.family_from)
    runs = annotate_model_family(runs, ref.family_from)
    examples = annotate_generation(examples, ref.family_from)
    runs = annotate_generation(runs, ref.family_from)
    examples = annotate_thinking(examples, pack)
    runs = annotate_thinking(runs, pack)
    examples = join_search_latency(examples, pack)
    runs = join_search_latency(runs, pack)
    examples = annotate_mem0_latency(examples)
    runs = annotate_mem0_latency(runs)
    runs = attach_run_judge_score(runs, examples)
    if not examples.empty and "result_source" not in examples.columns:
        examples = examples.copy()
        examples["result_source"] = "live"
    if not runs.empty and "result_source" not in runs.columns:
        runs = runs.copy()
        runs["result_source"] = "live"
    return runs, examples


def render_analysis(
    spec: AnalysisSpec,
    df: pd.DataFrame,
    out_dir: Path,
    pins: tuple[dict, ...] | list[dict] | None = None,
) -> AnalysisResult:
    work = _prepare_frame(spec, df)
    table = mean_table(work, list(spec.group_by), list(spec.metrics))
    if spec.include_pins and pins:
        pin_table = _pins_as_table(spec, pins)
        if not pin_table.empty:
            table = (
                pd.concat([table, pin_table], ignore_index=True)
                if not table.empty
                else pin_table
            )
    if not table.empty:
        table = sort_year_family_table(table)
    if table.empty:
        return AnalysisResult(spec=spec, table=table, skipped="empty table")
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{spec.id}.csv"
    table.to_csv(csv_path, index=False)
    plot_paths: list[Path] = []
    n_plots = len(spec.plots)
    for plot in spec.plots:
        dest = plots_dir / f"{spec.id}.png"
        if n_plots > 1:
            dest = plots_dir / f"{_plot_stem(spec.id, plot)}.png"
        written = _write_plot(plot, spec, table, dest)
        if written is not None:
            plot_paths.append(written)
    return AnalysisResult(spec=spec, table=table, csv_path=csv_path, plot_paths=plot_paths)


def _write_plot(
    plot: PlotSpec, spec: AnalysisSpec, table: pd.DataFrame, dest: Path
) -> Path | None:
    groups = [g for g in spec.group_by if g in table.columns]
    metrics = [m for m in spec.metrics if m in table.columns]
    if not groups or not metrics:
        return None
    kind = plot.kind
    if kind == "metrics_grouped_bar":
        collapsed = table.copy()
        if plot.hue and plot.hue in collapsed.columns:
            series_col = plot.hue
        elif len(groups) == 1:
            series_col = groups[0]
        else:
            series_col = "_series"
            collapsed[series_col] = collapsed[list(groups)].astype(str).agg(
                " × ".join, axis=1
            )
        mets = _requested_metrics(plot, metrics)
        if not mets:
            return None
        return write_metrics_grouped_bar(
            collapsed,
            group_col=series_col,
            metrics=mets,
            path=dest,
            title=_plot_title(plot, spec, mets[0] if len(mets) == 1 else None),
        )
    if kind in ("grouped_bar", "line"):
        x_col = plot.x if plot.x in table.columns else (groups[-1] if groups else None)
        y = _requested_metric(plot, metrics)
        if not x_col or not y:
            return None
        hue_parts = [g for g in groups if g != x_col]
        if plot.hue and plot.hue in table.columns and plot.hue != x_col:
            hue_parts = [plot.hue] + [g for g in hue_parts if g != plot.hue]
        if not hue_parts:
            if kind == "line":
                plot_table = table.copy()
                plot_table["_series"] = "all"
                return write_line(
                    plot_table,
                    x_col=x_col,
                    hue_col="_series",
                    metric=y,
                    path=dest,
                    title=_plot_title(plot, spec, y),
                )
            return write_bar(
                table,
                x_col=x_col,
                metric=y,
                path=dest,
                title=_plot_title(plot, spec, y),
            )
        plot_table = table
        if len(hue_parts) == 1:
            hue_col = hue_parts[0]
        else:
            hue_col = "_series"
            plot_table = table.copy()
            plot_table[hue_col] = (
                plot_table[hue_parts].astype(str).agg(" × ".join, axis=1)
            )
        writer = write_line if kind == "line" else write_grouped_bar
        return writer(
            plot_table,
            x_col=x_col,
            hue_col=hue_col,
            metric=y,
            path=dest,
            title=_plot_title(plot, spec, y),
        )
    if kind == "bar":
        x_col = plot.x if plot.x in table.columns else groups[0]
        y = _requested_metric(plot, metrics)
        if not y:
            return None
        return write_bar(
            table,
            x_col=x_col,
            metric=y,
            path=dest,
            title=_plot_title(plot, spec, y),
        )
    return None


def _requested_metric(plot: PlotSpec, metrics: list[str]) -> str | None:
    """YAML ``y`` wins; missing ``y`` does not reuse the first table metric."""
    if plot.y:
        return plot.y if plot.y in metrics else None
    return metrics[0] if metrics else None


def _requested_metrics(plot: PlotSpec, metrics: list[str]) -> list[str]:
    if plot.y:
        return [plot.y] if plot.y in metrics else []
    return list(metrics)


def _plot_title(plot: PlotSpec, spec: AnalysisSpec, metric: str | None) -> str:
    """One figure, one metric: do not reuse the analysis section title."""
    if plot.title:
        return plot.title
    if metric:
        return _axis_title(metric)
    return spec.title


def _plot_stem(analysis_id: str, plot: PlotSpec) -> str:
    parts = [analysis_id]
    for bit in (plot.x, plot.hue, plot.y, plot.kind):
        if bit:
            parts.append(str(bit))
    return "_".join(parts)


def render_experiment(
    cfg: CampaignConfig,
    experiment_id: str,
    *,
    root: Path | None = None,
) -> ReportResult:
    root = root or ROOT
    ref = cfg.experiments[experiment_id]
    loaded = load_pack(root, ref)
    out_dir = (root / ref.pack / cfg.output_subdir) if not ref.pack.is_absolute() else (
        ref.pack / cfg.output_subdir
    )
    if loaded is None:
        cost = render_cost(cfg, out_dir, root=root, experiment_id=experiment_id)
        if cost is not None:
            _write_summary(
                out_dir, cfg, f"experiment {experiment_id} ({ref.name})", []
            )
            _append_cost_summary(out_dir / "SUMMARY.md", cost)
        return ReportResult(
            scope=f"experiment:{experiment_id}",
            out_dir=out_dir,
            results=[],
            missing_packs=[ref.name],
            cost=cost,
        )
    _runs, examples = loaded
    results = [
        render_analysis(
            spec,
            _source_frame(spec, _runs, examples),
            out_dir,
            pins=cfg.pins if spec.include_pins else None,
        )
        for spec in ref.analyses
    ]
    for spec in cfg.campaign_analyses:
        if experiment_id not in spec.experiments:
            continue
        results.append(
            render_analysis(
                spec,
                _source_frame(spec, _runs, examples),
                out_dir,
                pins=cfg.pins if spec.include_pins else None,
            )
        )
    _write_summary(out_dir, cfg, f"experiment {experiment_id} ({ref.name})", results)
    cost = render_cost(cfg, out_dir, root=root, experiment_id=experiment_id)
    if cost is not None:
        _append_cost_summary(out_dir / "SUMMARY.md", cost)
    return ReportResult(
        scope=f"experiment:{experiment_id}",
        out_dir=out_dir,
        results=results,
        cost=cost,
    )


def render_campaign(
    cfg: CampaignConfig,
    *,
    root: Path | None = None,
) -> ReportResult:
    root = root or ROOT
    example_frames: dict[str, pd.DataFrame] = {}
    run_frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for exp_id, ref in cfg.experiments.items():
        loaded = load_pack(root, ref)
        if loaded is None:
            missing.append(ref.name)
            continue
        runs, examples = loaded
        examples = examples.copy()
        examples["campaign_experiment"] = exp_id
        runs = runs.copy()
        runs["campaign_experiment"] = exp_id
        example_frames[exp_id] = examples
        run_frames[exp_id] = runs
    out_dir = root / "experiments" / "_campaign" / cfg.id / cfg.output_subdir
    results: list[AnalysisResult] = []
    for spec in cfg.campaign_analyses:
        src = run_frames if spec.source == "runs" else example_frames
        parts = [src[eid] for eid in spec.experiments if eid in src]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if df.empty and not (spec.include_pins and cfg.pins):
            results.append(
                AnalysisResult(spec=spec, table=pd.DataFrame(), skipped="no packs")
            )
            continue
        results.append(
            render_analysis(
                spec, df, out_dir, pins=cfg.pins if spec.include_pins else None
            )
        )
    cost = render_cost(cfg, out_dir, root=root, experiment_id=None)
    results.extend(
        _render_insights(
            cfg,
            results,
            out_dir,
            cost=cost.by_cell if cost is not None else None,
        )
    )
    _write_summary(out_dir, cfg, f"campaign {cfg.id}", results, missing=missing)
    if cost is not None:
        _append_cost_summary(out_dir / "SUMMARY.md", cost)
    return ReportResult(
        scope=f"campaign:{cfg.id}",
        out_dir=out_dir,
        results=results,
        missing_packs=missing,
        cost=cost,
    )


def run_report(
    yaml_path: str | Path,
    *,
    experiment_id: str | None = None,
    root: Path | None = None,
) -> list[ReportResult]:
    cfg = load_campaign_yaml(yaml_path)
    root = root or ROOT
    if experiment_id:
        return [render_experiment(cfg, experiment_id, root=root)]
    out = [render_experiment(cfg, exp_id, root=root) for exp_id in cfg.experiments]
    out.append(render_campaign(cfg, root=root))
    return out


def notebook_show(report: ReportResult) -> None:
    """Display tables and PNGs in a Jupyter notebook (no-op extras if missing)."""
    try:
        from IPython.display import Image, Markdown, display
    except ImportError:
        for item in report.results:
            print(item.spec.id, item.skipped or f"n={len(item.table)}")
            if item.skipped:
                continue
            print(item.table.to_string(index=False))
            for path in item.plot_paths:
                print("plot", path)
        return
    if report.missing_packs:
        display(Markdown("**Missing packs:** " + ", ".join(report.missing_packs)))
    display(Markdown(f"Output `{report.out_dir.as_posix()}`"))
    if not report.results:
        return
    display(Markdown(f"Output `{report.out_dir.as_posix()}`"))
    for item in report.results:
        display(Markdown(f"### {item.spec.title}"))
        if item.skipped:
            display(Markdown(f"_skipped: {item.skipped}_"))
            continue
        display(item.table)
        for path in item.plot_paths:
            display(Image(filename=str(path)))


def _render_insights(
    cfg: CampaignConfig,
    results: list[AnalysisResult],
    out_dir: Path,
    cost: pd.DataFrame | None = None,
) -> list[AnalysisResult]:
    """Year-family robustness CSVs from already-written mean tables."""
    lookup = {item.spec.id: item.table for item in results if not item.skipped}
    extra: list[AnalysisResult] = []
    tables_dir = out_dir / "tables"
    for spec in cfg.insights:
        source = lookup.get(spec.source_analysis)
        if source is None or source.empty:
            extra.append(
                AnalysisResult(
                    spec=_insight_analysis_spec(spec),
                    table=pd.DataFrame(),
                    skipped=f"missing source {spec.source_analysis}",
                )
            )
            continue
        work = source
        if spec.join_cost:
            work = attach_cell_cost(work, cost if cost is not None else pd.DataFrame())
        kwargs: dict = {
            "metrics": spec.metrics,
            "left_family": spec.left_family,
            "right_family": spec.right_family,
        }
        if spec.metric:
            kwargs["metric"] = spec.metric
        if spec.hole_max is not None:
            kwargs["hole_max"] = spec.hole_max
        if spec.move_eps is not None:
            kwargs["move_eps"] = spec.move_eps
        table = render_insight(spec.kind, work, **kwargs)
        tables_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tables_dir / f"{spec.id}.csv"
        if not table.empty:
            table.to_csv(csv_path, index=False)
        extra.append(
            AnalysisResult(
                spec=_insight_analysis_spec(spec),
                table=table,
                csv_path=csv_path if not table.empty else None,
                skipped="empty table" if table.empty else None,
            )
        )
        lookup[spec.id] = table
    return extra


def _insight_analysis_spec(spec: InsightSpec) -> AnalysisSpec:
    return AnalysisSpec(
        id=spec.id,
        title=spec.title,
        group_by=(),
        metrics=spec.metrics,
        plots=(),
        source="examples",
    )


def _source_frame(
    spec: AnalysisSpec, runs: pd.DataFrame, examples: pd.DataFrame
) -> pd.DataFrame:
    if spec.source == "runs":
        return runs
    return examples


def _prepare_frame(spec: AnalysisSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Apply category exclusion and YAML ``where`` before grouping."""
    if df.empty:
        return df
    out = df
    drop = _categories_to_drop(spec)
    if drop and "question_category" in out.columns:
        category = pd.to_numeric(out["question_category"], errors="coerce")
        out = out[~category.isin(set(drop))]
    return _apply_where(out, spec.where)


def _categories_to_drop(spec: AnalysisSpec) -> tuple[int, ...]:
    """Mem0 J / paper clone overalls skip category 5; category plots keep it."""
    if "question_category" in spec.group_by:
        return spec.exclude_question_categories
    return spec.exclude_question_categories or (5,)


def _apply_where(
    df: pd.DataFrame, where: tuple[tuple[str, tuple[str, ...]], ...]
) -> pd.DataFrame:
    out = df
    for col, values in where:
        if col not in out.columns:
            continue
        out = out[out[col].astype(str).isin(values)]
    return out


def _pins_as_table(
    spec: AnalysisSpec, pins: tuple[dict, ...] | list[dict]
) -> pd.DataFrame:
    frame = pd.DataFrame(list(pins))
    if frame.empty:
        return frame
    frame = _apply_where(frame, spec.where)
    groups = [c for c in spec.group_by if c in frame.columns]
    metrics = [m for m in spec.metrics if m in frame.columns]
    if frame.empty or not groups or not metrics:
        return pd.DataFrame()
    keep = list(groups) + list(metrics)
    if "n" in frame.columns:
        keep.append("n")
    out = frame[keep].copy()
    if "n" not in out.columns:
        out["n"] = 1
    return out


def _write_summary(
    out_dir: Path,
    cfg: CampaignConfig,
    heading: str,
    results: list[AnalysisResult],
    missing: list[str] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {heading}",
        "",
        cfg.freeze_note,
        "",
        f"Config: `{cfg.source_path.as_posix()}`",
        "",
    ]
    if missing:
        lines.append("Missing packs: " + ", ".join(missing))
        lines.append("")
    for item in results:
        lines.append(f"## {item.spec.id} — {item.spec.title}")
        if item.skipped:
            lines.append(f"skipped: {item.skipped}")
            lines.append("")
            continue
        if item.csv_path:
            lines.append(f"table: `{item.csv_path.as_posix()}`")
        for plot in item.plot_paths:
            lines.append(f"plot: `{plot.as_posix()}`")
        lines.append("")
        if not item.table.empty:
            lines.append(item.table.to_string(index=False))
            lines.append("")
    dest = out_dir / "SUMMARY.md"
    dest.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return dest


def _append_cost_summary(summary_path: Path, cost: CostReport) -> None:
    lines = ["", "## cost", ""]
    for path in cost.csv_paths:
        lines.append(f"table: `{path.as_posix()}`")
    for path in cost.plot_paths:
        lines.append(f"plot: `{path.as_posix()}`")
    lines.append("")
    if not cost.by_stage.empty:
        lines.append(cost.by_stage.to_string(index=False))
        lines.append("")
    existing = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
    summary_path.write_text(existing.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")
