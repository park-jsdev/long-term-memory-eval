"""Load ``configs/analysis/*.yaml`` (campaign vs experiment report plane)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config import load_config

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PlotSpec:
    """One visualization over a table. ``x`` / ``hue`` / ``y`` are column names.

    Omitted fields fall back to ``group_by`` / ``metrics`` order in the report.
    """

    kind: str
    x: str | None = None
    hue: str | None = None
    y: str | None = None


@dataclass(frozen=True)
class AnalysisSpec:
    """One table+plot recipe. Campaign or experiment scope is the caller."""

    id: str
    title: str
    group_by: tuple[str, ...]
    metrics: tuple[str, ...]
    plots: tuple[PlotSpec, ...]
    source: str
    experiments: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentAnalysisRef:
    """Pack location + which analyses to run for one campaign member."""

    id: str
    name: str
    pack: Path
    notebook: str | None
    role: str
    family_from: str
    subset: str | None
    analyses: tuple[AnalysisSpec, ...]


@dataclass(frozen=True)
class CampaignConfig:
    """What it is: parsed analysis YAML. Who consumes it: report + notebooks."""

    id: str
    title: str
    freeze_note: str
    defaults_metrics: tuple[str, ...]
    output_subdir: str
    campaign_analyses: tuple[AnalysisSpec, ...]
    experiments: dict[str, ExperimentAnalysisRef]
    source_path: Path


def load_campaign_yaml(path: str | Path) -> CampaignConfig:
    raw = load_config(path) if Path(path).suffix in {".yaml", ".yml"} else {}
    if not raw:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    campaign = raw.get("campaign") or {}
    defaults = raw.get("defaults") or {}
    default_metrics = tuple(
        str(m) for m in (defaults.get("metrics") or ["locomo_f1"])
    )
    default_source = str(defaults.get("source") or "examples")
    experiments: dict[str, ExperimentAnalysisRef] = {}
    for exp_id, block in (raw.get("experiments") or {}).items():
        experiments[str(exp_id)] = _parse_experiment(
            str(exp_id), block or {}, default_metrics, default_source
        )
    analyses = tuple(
        _parse_analysis(item, default_metrics, default_source)
        for item in (raw.get("campaign_analyses") or [])
    )
    return CampaignConfig(
        id=str(campaign.get("id") or "campaign"),
        title=str(campaign.get("title") or campaign.get("id") or "campaign"),
        freeze_note=str(campaign.get("freeze_note") or "").strip(),
        defaults_metrics=default_metrics,
        output_subdir=str(defaults.get("output_subdir") or "analysis"),
        campaign_analyses=analyses,
        experiments=experiments,
        source_path=Path(path).resolve(),
    )


def _parse_experiment(
    exp_id: str,
    block: dict[str, Any],
    default_metrics: tuple[str, ...],
    default_source: str,
) -> ExperimentAnalysisRef:
    pack = Path(str(block.get("pack") or f"experiments/{block.get('name') or exp_id}"))
    analyses = tuple(
        _parse_analysis(item, default_metrics, default_source)
        for item in (block.get("analyses") or [])
    )
    return ExperimentAnalysisRef(
        id=exp_id,
        name=str(block.get("name") or exp_id),
        pack=pack,
        notebook=_as_str(block.get("notebook")),
        role=str(block.get("role") or "sweep"),
        family_from=str(block.get("family_from") or "reader"),
        subset=_as_str(block.get("subset")),
        analyses=analyses,
    )


def _parse_analysis(
    item: dict[str, Any],
    default_metrics: tuple[str, ...],
    default_source: str,
) -> AnalysisSpec:
    metrics = item.get("metrics")
    plots = item.get("plots") or ["metrics_grouped_bar"]
    groups = item.get("group_by") or ["reader_display_name"]
    experiments = item.get("experiments") or []
    return AnalysisSpec(
        id=str(item.get("id") or "analysis"),
        title=str(item.get("title") or item.get("id") or "analysis"),
        group_by=tuple(str(g) for g in groups),
        metrics=tuple(str(m) for m in metrics) if metrics else default_metrics,
        plots=tuple(_parse_plot(p) for p in plots),
        source=str(item.get("source") or default_source),
        experiments=tuple(str(e) for e in experiments),
    )


def _parse_plot(item: Any) -> PlotSpec:
    if isinstance(item, str):
        return PlotSpec(kind=item)
    block = item or {}
    return PlotSpec(
        kind=str(block.get("kind") or "metrics_grouped_bar"),
        x=_as_str(block.get("x")),
        hue=_as_str(block.get("hue")),
        y=_as_str(block.get("y")),
    )


def _as_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
