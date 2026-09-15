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
    write_bar,
    write_grouped_bar,
    write_metrics_grouped_bar,
)
from scripts.analysis.campaign_tables import annotate_model_family, mean_table
from src.memorybench.analysis.load_campaign import (
    AnalysisSpec,
    CampaignConfig,
    ExperimentAnalysisRef,
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
    return runs, examples


def render_analysis(
    spec: AnalysisSpec,
    df: pd.DataFrame,
    out_dir: Path,
) -> AnalysisResult:
    table = mean_table(df, list(spec.group_by), list(spec.metrics))
    if table.empty:
        return AnalysisResult(spec=spec, table=table, skipped="empty table")
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{spec.id}.csv"
    table.to_csv(csv_path, index=False)
    plot_paths: list[Path] = []
    for plot in spec.plots:
        dest = plots_dir / f"{spec.id}.png"
        if len(spec.plots) > 1:
            suffix = plot.y or plot.kind
            dest = plots_dir / f"{spec.id}_{suffix}.png"
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
        mets = [plot.y] if plot.y and plot.y in metrics else metrics
        return write_metrics_grouped_bar(
            collapsed,
            group_col=series_col,
            metrics=mets,
            path=dest,
            title=spec.title,
        )
    if kind == "grouped_bar":
        x_col = plot.x if plot.x in table.columns else (groups[-1] if groups else None)
        hue_col = plot.hue if plot.hue in table.columns else (groups[0] if groups else None)
        y = plot.y if plot.y in metrics else (metrics[0] if metrics else None)
        if not x_col or not y:
            return None
        if not hue_col or hue_col == x_col:
            return write_bar(
                table, x_col=x_col, metric=y, path=dest, title=spec.title
            )
        return write_grouped_bar(
            table,
            x_col=x_col,
            hue_col=hue_col,
            metric=y,
            path=dest,
            title=spec.title,
        )
    if kind == "bar":
        x_col = plot.x if plot.x in table.columns else groups[0]
        y = plot.y if plot.y in metrics else metrics[0]
        return write_bar(
            table, x_col=x_col, metric=y, path=dest, title=spec.title
        )
    return None


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
        return ReportResult(
            scope=f"experiment:{experiment_id}",
            out_dir=out_dir,
            results=[],
            missing_packs=[ref.name],
        )
    _runs, examples = loaded
    results = [
        render_analysis(spec, _source_frame(spec, _runs, examples), out_dir)
        for spec in ref.analyses
    ]
    for spec in cfg.campaign_analyses:
        if experiment_id not in spec.experiments:
            continue
        results.append(
            render_analysis(spec, _source_frame(spec, _runs, examples), out_dir)
        )
    _write_summary(out_dir, cfg, f"experiment {experiment_id} ({ref.name})", results)
    return ReportResult(
        scope=f"experiment:{experiment_id}", out_dir=out_dir, results=results
    )


def render_campaign(
    cfg: CampaignConfig,
    *,
    root: Path | None = None,
) -> ReportResult:
    root = root or ROOT
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for exp_id, ref in cfg.experiments.items():
        loaded = load_pack(root, ref)
        if loaded is None:
            missing.append(ref.name)
            continue
        _runs, examples = loaded
        examples = examples.copy()
        examples["campaign_experiment"] = exp_id
        frames[exp_id] = examples
    out_dir = root / "experiments" / "_campaign" / cfg.id / cfg.output_subdir
    results: list[AnalysisResult] = []
    for spec in cfg.campaign_analyses:
        parts = [frames[eid] for eid in spec.experiments if eid in frames]
        if not parts:
            results.append(
                AnalysisResult(spec=spec, table=pd.DataFrame(), skipped="no packs")
            )
            continue
        df = pd.concat(parts, ignore_index=True)
        results.append(render_analysis(spec, df, out_dir))
    _write_summary(out_dir, cfg, f"campaign {cfg.id}", results, missing=missing)
    return ReportResult(
        scope=f"campaign:{cfg.id}",
        out_dir=out_dir,
        results=results,
        missing_packs=missing,
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


def _source_frame(
    spec: AnalysisSpec, runs: pd.DataFrame, examples: pd.DataFrame
) -> pd.DataFrame:
    if spec.source == "runs":
        return runs
    return examples


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
