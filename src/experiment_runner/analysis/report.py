"""Run design and experiment analyses from YAML configurations (no LLM).

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
from scripts.analysis.agent_harness import overlay_collected_agent_audit
from scripts.analysis.campaign_tables import (
    annotate_answer_abstention,
    annotate_generation,
    annotate_gap_stack,
    annotate_gold_token_coverage,
    annotate_mem0_latency,
    annotate_memory_lane,
    annotate_mini_side,
    annotate_model_family,
    annotate_reader_stack,
    annotate_paper_compare,
    annotate_prompt_parity_condition,
    annotate_reader_list_price,
    annotate_system_harness,
    annotate_thinking,
    annotate_writer_harness,
    annotate_workspace_diagnostics,
    attach_agent_identity,
    attach_run_agent_audit,
    attach_run_judge_score,
    COMPARE_SOURCE_AXIS,
    COMPARE_SOURCE_LIVE,
    copy_run_identity_to_examples,
    GENERATION_AXIS,
    join_search_latency,
    LIVE_SOURCE_AXIS,
    mean_table,
    MEMORY_LANE_AXIS,
    PAPER_METHOD_AXIS,
    READER_STACK_AXIS,
    repair_run_harness_status,
    sort_year_family_table,
    SYSTEM_HARNESS_AXIS,
)
from src.experiment_runner.analysis.cost import CostReport, render_cost
from scripts.analysis.campaign_insights import (
    attach_cell_cost,
    filter_where,
    render_insight,
    takeaway_contrast,
)
from src.experiment_runner.analysis.load_design import (
    AnalysisSpec,
    DesignConfig,
    ExperimentAnalysisRef,
    InsightSpec,
    PlotSpec,
    load_design_yaml,
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
    examples = attach_agent_identity(examples, pack)
    runs = attach_agent_identity(runs, pack)
    examples = copy_run_identity_to_examples(examples, runs)
    examples = overlay_collected_agent_audit(examples, pack)
    examples = annotate_workspace_diagnostics(examples)
    runs = attach_run_agent_audit(runs, examples)
    runs = repair_run_harness_status(runs, examples)
    examples = annotate_writer_harness(examples)
    runs = annotate_writer_harness(runs)
    examples = annotate_memory_lane(examples)
    runs = annotate_memory_lane(runs)
    examples = annotate_system_harness(examples)
    runs = annotate_system_harness(runs)
    examples = annotate_reader_stack(examples)
    runs = annotate_reader_stack(runs)
    if not examples.empty and "result_source" not in examples.columns:
        examples = examples.copy()
        examples["result_source"] = "live"
    if not runs.empty and "result_source" not in runs.columns:
        runs = runs.copy()
        runs["result_source"] = "live"
    examples = annotate_paper_compare(examples)
    runs = annotate_paper_compare(runs)
    examples = annotate_mini_side(examples)
    runs = annotate_mini_side(runs)
    examples = annotate_gap_stack(examples)
    runs = annotate_gap_stack(runs)
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
        pin_table = _pins_as_table(spec, pins, live=work)
        if not pin_table.empty:
            table = (
                _concat_frames([table, pin_table])
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
        if n_plots > 1 or plot.split_by:
            dest = plots_dir / f"{_plot_stem(spec.id, plot)}.png"
        for split_dest, split_table, split_title in _plot_frames(
            plot, spec, table, dest
        ):
            written = _write_plot(
                plot, spec, split_table, split_dest, title_suffix=split_title
            )
            if written is not None:
                plot_paths.append(written)
    return AnalysisResult(spec=spec, table=table, csv_path=csv_path, plot_paths=plot_paths)


def _write_plot(
    plot: PlotSpec,
    spec: AnalysisSpec,
    table: pd.DataFrame,
    dest: Path,
    title_suffix: str | None = None,
) -> Path | None:
    groups = [g for g in spec.group_by if g in table.columns]
    metrics = [m for m in spec.metrics if m in table.columns]
    if plot.y == "n" and "n" in table.columns and "n" not in metrics:
        metrics = [*metrics, "n"]
    if not groups or not metrics:
        return None
    kind = plot.kind
    title = lambda metric: _plot_title(plot, spec, metric, suffix=title_suffix)
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
            title=title(mets[0] if len(mets) == 1 else None),
        )
    if kind in ("grouped_bar", "line"):
        x_col = _plot_axis_column(plot.x, table, groups, default_index=-1)
        y = _requested_metric(plot, metrics)
        if not x_col or not y:
            return None
        if plot.hue and plot.hue not in table.columns:
            return None
        hue_parts = [g for g in groups if g != x_col]
        if plot.split_by:
            hue_parts = [g for g in hue_parts if g != plot.split_by]
        if plot.hue and plot.hue != x_col:
            hue_parts = [plot.hue] + [g for g in hue_parts if g != plot.hue]
        hue_parts = [
            g
            for g in hue_parts
            if g in table.columns and table[g].nunique(dropna=False) > 1
        ]
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
                    title=title(y),
                )
            return write_bar(
                table,
                x_col=x_col,
                metric=y,
                path=dest,
                title=title(y),
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
            title=title(y),
        )
    if kind == "bar":
        x_col = _plot_axis_column(plot.x, table, groups, default_index=0)
        y = _requested_metric(plot, metrics)
        if not x_col or not y:
            return None
        return write_bar(
            table,
            x_col=x_col,
            metric=y,
            path=dest,
            title=title(y),
        )
    return None


def _requested_metric(plot: PlotSpec, metrics: list[str]) -> str | None:
    """YAML ``y`` wins; missing ``y`` does not reuse the first table metric."""
    if plot.y:
        return plot.y if plot.y in metrics else None
    return metrics[0] if metrics else None


def _plot_axis_column(
    requested: str | None,
    table: pd.DataFrame,
    groups: list[str],
    *,
    default_index: int,
) -> str | None:
    """Use YAML ``x`` when present; omit it only when the recipe did not set ``x``.

    A set-but-missing ``x`` (``paper_method`` after a collapsed mean table) must
    not fall back to ``model_family`` — that drew one OpenAI bar for every method.
    """
    if requested:
        return requested if requested in table.columns else None
    if not groups:
        return None
    if default_index < 0:
        index = max(len(groups) + default_index, 0)
    else:
        index = min(default_index, len(groups) - 1)
    return groups[index]


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-concat pack frames without dropping all-NA columns.

    pandas currently omits all-NA columns when it unions dtypes, which can
    strip ``live_source`` / ``agent_persist`` from a design concat of
    readers+writers and then collapse ``group_by``.
    """
    nonempty = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    columns: list[str] = []
    seen: set[str] = set()
    for frame in nonempty:
        for col in frame.columns:
            if col not in seen:
                seen.add(col)
                columns.append(col)
    aligned = [frame.reindex(columns=columns) for frame in nonempty]
    return pd.concat(aligned, ignore_index=True)


def _requested_metrics(plot: PlotSpec, metrics: list[str]) -> list[str]:
    if plot.y:
        return [plot.y] if plot.y in metrics else []
    return list(metrics)


def _plot_title(
    plot: PlotSpec,
    spec: AnalysisSpec,
    metric: str | None,
    suffix: str | None = None,
) -> str:
    """One figure, one metric: do not reuse the analysis section title."""
    if plot.title:
        base = plot.title
    elif metric:
        base = _axis_title(metric)
    else:
        base = spec.title
    extra = str(suffix or "").strip()
    if extra:
        return f"{base} — {extra}"
    return base


def _plot_stem(analysis_id: str, plot: PlotSpec) -> str:
    parts = [analysis_id]
    for bit in (plot.x, plot.hue, plot.y, plot.kind):
        if bit:
            parts.append(str(bit))
    return "_".join(parts)


def _plot_frames(
    plot: PlotSpec, spec: AnalysisSpec, table: pd.DataFrame, dest: Path
) -> list[tuple[Path, pd.DataFrame, str | None]]:
    """One PNG for the table, or one PNG per ``split_by`` value."""
    col = plot.split_by
    if not col or col not in table.columns:
        return [(dest, table, None)]
    values = [str(v) for v in table[col].dropna().unique().tolist()]
    frames: list[tuple[Path, pd.DataFrame, str | None]] = []
    for value in _order_split_values(col, values):
        part = table[table[col].astype(str) == value]
        if part.empty:
            continue
        split_dest = dest.with_name(f"{dest.stem}_{_slug_label(value)}{dest.suffix}")
        frames.append((split_dest, part, value))
    return frames or [(dest, table, None)]


def _order_split_values(col: str, values: list[str]) -> list[str]:
    known: tuple[str, ...] = ()
    if col == "generation":
        known = GENERATION_AXIS
    elif col == "compare_source":
        known = COMPARE_SOURCE_AXIS
    elif col == "live_source":
        known = LIVE_SOURCE_AXIS
    elif col == "paper_method":
        known = PAPER_METHOD_AXIS
    elif col == "reader_stack":
        known = READER_STACK_AXIS
    elif col == "memory_lane":
        known = MEMORY_LANE_AXIS
    elif col == "system_harness":
        known = SYSTEM_HARNESS_AXIS
    if not known:
        return values
    present = set(values)
    return [item for item in known if item in present] + [
        item for item in values if item not in known
    ]


def _slug_label(value: str) -> str:
    text = str(value).strip().lower()
    out = []
    for char in text:
        if char.isalnum():
            out.append(char)
        elif char in {" ", "_", "+", "-", "/"}:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "value"


def render_experiment(
    cfg: DesignConfig,
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
    for spec in cfg.design_analyses:
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
    takeaway_results = _render_takeaways(cfg, results, out_dir)
    results = takeaway_results + results
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


def render_design(
    cfg: DesignConfig,
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
        examples["design_experiment"] = exp_id
        runs = runs.copy()
        runs["design_experiment"] = exp_id
        example_frames[exp_id] = examples
        run_frames[exp_id] = runs
    out_dir = root / "experiments" / "_design" / cfg.id / cfg.output_subdir
    results: list[AnalysisResult] = []
    for spec in cfg.design_analyses:
        src = run_frames if spec.source == "runs" else example_frames
        parts = [src[eid] for eid in spec.experiments if eid in src]
        df = _concat_frames(parts)
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
    cost = render_cost(cfg, out_dir, root=root, experiment_id=None        )
    results.extend(
        _render_insights(
            cfg,
            results,
            out_dir,
            cost=cost.by_cell if cost is not None else None,
        )
    )
    takeaway_results = _render_takeaways(cfg, results, out_dir)
    results = takeaway_results + results
    _write_summary(out_dir, cfg, f"design {cfg.id}", results, missing=missing)
    if cost is not None:
        _append_cost_summary(out_dir / "SUMMARY.md", cost)
    return ReportResult(
        scope=f"design:{cfg.id}",
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
    cfg = load_design_yaml(yaml_path)
    root = root or ROOT
    if experiment_id:
        return [render_experiment(cfg, experiment_id, root=root)]
    out = [render_experiment(cfg, exp_id, root=root) for exp_id in cfg.experiments]
    out.append(render_design(cfg, root=root))
    return out


def dataframe_markdown(table: pd.DataFrame) -> str:
    """GitHub-flavored table so Cursor / Jupyter Markdown keep columns aligned.

    ``DataFrame.to_string`` inside Markdown collapses spaces. Pandas HTML
    often fails to render in Cursor notebooks. Pipe tables do both.
    """
    if table is None or table.empty:
        return "_empty table_"
    cols = [str(c) for c in table.columns]
    header = "| " + " | ".join(cols) + " |"
    align = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in table.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(_markdown_cell(v) for v in row) + " |")
    return "\n".join([header, align, *rows])


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("\n", " ").replace("|", "\\|")


def dataframe_html(table: pd.DataFrame) -> str:
    """Standalone HTML table for Cursor notebooks (not mixed into Markdown)."""
    if table is None or table.empty:
        return "<p><em>empty table</em></p>"
    formatted = pd.DataFrame(
        {str(col): [_markdown_cell(v) for v in table[col].tolist()] for col in table.columns}
    )
    body = formatted.to_html(index=False, escape=True, border=1)
    return f'<div style="overflow-x:auto">{body}</div>'


def _display_notebook_table(table: pd.DataFrame) -> None:
    """HTML table as its own output — GFM tables inside prose collapse in Cursor."""
    try:
        from IPython.display import HTML, display

        display(HTML(dataframe_html(table)))
    except ImportError:
        print(dataframe_markdown(table))


def notebook_show(
    report: ReportResult,
    only: tuple[str, ...] | None = None,
    *,
    header: bool = True,
) -> None:
    """Display tables and PNGs in a Jupyter notebook (no-op extras if missing).

    ``only`` keeps those analysis ids, in the given order, so a notebook can
    put Quality, Performance, Retention, and Abstention in separate sections.
    """
    results = _select_results(report, only)
    try:
        from IPython.display import Image, Markdown, display
    except ImportError:
        for item in results:
            if item.spec.id == "takeaways":
                print("\n".join(_takeaways_markdown(item.table, item.csv_path)))
                continue
            print(item.spec.id, item.skipped or f"n={len(item.table)}")
            if item.skipped:
                continue
            print(dataframe_markdown(item.table))
            for path in item.plot_paths:
                print("plot", path)
        return
    if header and report.missing_packs:
        display(Markdown("**Missing packs:** " + ", ".join(report.missing_packs)))
    if header:
        display(Markdown(f"Output `{report.out_dir.as_posix()}`"))
    if not results:
        return
    for item in results:
        if item.spec.id == "takeaways":
            _notebook_show_takeaways(item)
            continue
        display(Markdown(f"### {item.spec.title}"))
        if item.skipped:
            display(Markdown(f"_skipped: {item.skipped}_"))
            continue
        _display_notebook_table(item.table)
        for path in item.plot_paths:
            display(Image(filename=str(path)))


def _select_results(
    report: ReportResult, only: tuple[str, ...] | None
) -> list[AnalysisResult]:
    if only is None:
        return list(report.results)
    order = {name: i for i, name in enumerate(only)}
    chosen = [item for item in report.results if item.spec.id in order]
    chosen.sort(key=lambda item: order[item.spec.id])
    return chosen


def _notebook_show_takeaways(item: AnalysisResult) -> None:
    """Prose Markdown and metric HTML are separate outputs so Cursor can render both."""
    from IPython.display import Markdown, display

    for heading, metrics in _takeaway_sections(item.table, item.csv_path):
        display(Markdown(heading))
        if metrics is not None and not metrics.empty:
            _display_notebook_table(metrics)


def _render_insights(
    cfg: DesignConfig,
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
        if spec.live_generation:
            kwargs["live_generation"] = spec.live_generation
        if spec.series_col:
            kwargs["series_col"] = spec.series_col
        if spec.left_series:
            kwargs["left_series"] = spec.left_series
        if spec.right_series:
            kwargs["right_series"] = spec.right_series
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


def _render_takeaways(
    cfg: DesignConfig,
    results: list[AnalysisResult],
    out_dir: Path,
) -> list[AnalysisResult]:
    """Harness vs model-reader/writer contrasts for SUMMARY / notebook."""
    if not cfg.takeaways:
        return []
    lookup = {item.spec.id: item.table for item in results if not item.skipped}
    records: list[dict] = []
    for spec in cfg.takeaways:
        if not spec.source_analysis:
            piece = takeaway_contrast(
                pd.DataFrame(),
                pd.DataFrame(),
                spec.metrics,
                takeaway_id=spec.id,
                title=spec.title,
                claim=spec.claim,
                finding=spec.finding,
                left_label=spec.left_label,
                right_label=spec.right_label,
            )
        else:
            left_src = lookup.get(spec.source_analysis)
            right_key = spec.right_source or spec.source_analysis
            right_src = lookup.get(right_key)
            if left_src is None or left_src.empty:
                continue
            if right_src is None:
                right_src = pd.DataFrame()
            left = filter_where(left_src, spec.left) if spec.left else left_src
            right = filter_where(right_src, spec.right) if spec.right else right_src
            piece = takeaway_contrast(
                left,
                right,
                spec.metrics,
                takeaway_id=spec.id,
                title=spec.title,
                claim=spec.claim,
                finding=spec.finding,
                left_label=spec.left_label,
                right_label=spec.right_label,
            )
        records.extend(piece.to_dict(orient="records"))
    if not records:
        return []
    table = pd.DataFrame.from_records(records)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "takeaways.csv"
    table.to_csv(csv_path, index=False)
    return [
        AnalysisResult(
            spec=AnalysisSpec(
                id="takeaways",
                title="Harness vs model reader/writer",
                group_by=(),
                metrics=(),
                plots=(),
                source="examples",
            ),
            table=table,
            csv_path=csv_path,
        )
    ]


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
    if "result_source" not in out.columns:
        out = out.copy()
        out["result_source"] = "live"
    out = annotate_writer_harness(out)
    out = annotate_memory_lane(out)
    out = annotate_system_harness(out)
    out = annotate_paper_compare(out)
    out = annotate_mini_side(out)
    out = annotate_gap_stack(out)
    out = annotate_prompt_parity_condition(out)
    if any(metric.startswith("gold_") for metric in spec.metrics):
        out = annotate_gold_token_coverage(out)
    if "reader_usd" in spec.metrics:
        out = annotate_reader_list_price(out)
    if _abstention_analysis(spec):
        out = annotate_answer_abstention(out)
    drop = _categories_to_drop(spec)
    if drop and "question_category" in out.columns:
        category = pd.to_numeric(out["question_category"], errors="coerce")
        out = out[~category.isin(set(drop))]
    return _apply_where(out, spec.where)


_ABSTENTION_RATES = frozenset({"adversarial_refusal", "false_refusal"})


def _abstention_analysis(spec: AnalysisSpec) -> bool:
    """True when the recipe asks for a refusal rate or its denominator."""
    return any(metric.removesuffix("_n") in _ABSTENTION_RATES for metric in spec.metrics)


def _categories_to_drop(spec: AnalysisSpec) -> tuple[int, ...]:
    """Mem0 J / paper clone overalls skip category 5; category plots keep it.

    Failure-mode counts are an audit of what the harness did, so they keep
    adversarial questions the same way category bars do. Refusal rates also
    keep both classes: each rate is missing on the class it does not score.
    """
    if (
        "question_category" in spec.group_by
        or "failure_mode" in spec.group_by
        or _abstention_analysis(spec)
    ):
        return spec.exclude_question_categories
    return spec.exclude_question_categories or (5,)


_PAPER_COMPARE_FILTERS = frozenset(
    {
        "paper_method",
        "live_source",
        "compare_source",
        "reader_stack",
        "memory_lane",
        "system_harness",
    }
)


def _apply_where(
    df: pd.DataFrame, where: tuple[tuple[str, tuple[str, ...]], ...]
) -> pd.DataFrame:
    out = df
    for col, values in where:
        if col not in out.columns:
            # Paper-compare filters are required. Skipping them used to keep
            # every method and then average them in mean_table.
            if col in _PAPER_COMPARE_FILTERS:
                return out.iloc[0:0]
            continue
        out = out[out[col].astype(str).isin(values)]
    return out


def _drop_clone_when_live_exists(
    pins: pd.DataFrame,
    live: pd.DataFrame | None,
    spec: AnalysisSpec,
) -> pd.DataFrame:
    """Methods plot folds clone into live model; drop the clone pin if a pack live-model row exists."""
    if pins.empty or live is None or live.empty:
        return pins
    if "live_source" not in spec.group_by:
        return pins
    if "compare_source" not in live.columns or "paper_method" not in live.columns:
        return pins
    live_methods = {
        str(v)
        for v in live.loc[
            live["compare_source"].astype(str) == COMPARE_SOURCE_LIVE, "paper_method"
        ].tolist()
    }
    if not live_methods:
        return pins
    clone = pins["compare_source"].astype(str) == "local clone"
    overlap = pins["paper_method"].astype(str).isin(live_methods)
    return pins[~(clone & overlap)].copy()


def _pins_as_table(
    spec: AnalysisSpec,
    pins: tuple[dict, ...] | list[dict],
    live: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(list(pins))
    if frame.empty:
        return frame
    frame = annotate_reader_stack(frame)
    frame = annotate_paper_compare(frame)
    frame = _drop_clone_when_live_exists(frame, live, spec)
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
    cfg: DesignConfig,
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
    takeaway = next((item for item in results if item.spec.id == "takeaways"), None)
    if takeaway is not None and not takeaway.table.empty:
        lines.extend(_takeaways_markdown(takeaway.table, takeaway.csv_path))
    for item in results:
        if item.spec.id == "takeaways":
            continue
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
            lines.append(dataframe_markdown(item.table))
            lines.append("")
    dest = out_dir / "SUMMARY.md"
    dest.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return dest


def _takeaway_metric_table(piece: pd.DataFrame) -> pd.DataFrame | None:
    cols = [
        c
        for c in (
            "left_label",
            "right_label",
            "metric",
            "n_left",
            "n_right",
            "left_value",
            "right_value",
            "delta",
        )
        if c in piece.columns
    ]
    if not cols:
        return None
    show = piece[cols]
    if show.empty or "metric" not in show.columns or not show["metric"].notna().any():
        return None
    return show


def _takeaway_sections(
    table: pd.DataFrame, csv_path: Path | None
) -> list[tuple[str, pd.DataFrame | None]]:
    """Heading Markdown separate from the metric frame (notebook HTML vs SUMMARY pipes)."""
    sections: list[tuple[str, pd.DataFrame | None]] = []
    intro = ["## Takeaways — harness vs model reader/writer", ""]
    if csv_path is not None:
        intro.append(f"table: `{csv_path.as_posix()}`")
        intro.append("")
    sections.append(("\n".join(intro).rstrip(), None))
    if table.empty or "takeaway_id" not in table.columns:
        return sections
    order = []
    for tid in table["takeaway_id"].astype(str):
        if tid not in order:
            order.append(tid)
    for tid in order:
        piece = table[table["takeaway_id"].astype(str) == tid]
        row = piece.iloc[0]
        heading = [f"### {row['title']}", f"claim: `{row['claim']}`", ""]
        finding = str(row.get("finding") or "").strip()
        if finding:
            heading.append(finding)
            heading.append("")
        sections.append(("\n".join(heading).rstrip(), _takeaway_metric_table(piece)))
    return sections


def _takeaways_markdown(table: pd.DataFrame, csv_path: Path | None) -> list[str]:
    lines: list[str] = []
    for heading, metrics in _takeaway_sections(table, csv_path):
        lines.append(heading)
        lines.append("")
        if metrics is not None and not metrics.empty:
            lines.append(dataframe_markdown(metrics))
            lines.append("")
    return lines


def _append_cost_summary(summary_path: Path, cost: CostReport) -> None:
    lines = ["", "## cost", ""]
    for path in cost.csv_paths:
        lines.append(f"table: `{path.as_posix()}`")
    for path in cost.plot_paths:
        lines.append(f"plot: `{path.as_posix()}`")
    lines.append("")
    if not cost.by_stage.empty:
        lines.append(dataframe_markdown(cost.by_stage))
        lines.append("")
    existing = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
    summary_path.write_text(existing.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")
