"""Load ``configs/analysis/*.yaml`` (campaign vs experiment report plane)."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    where: tuple[tuple[str, tuple[str, ...]], ...] = ()
    exclude_question_categories: tuple[int, ...] = ()
    include_pins: bool = False


@dataclass(frozen=True)
class PretestSpec:
    """Declared expectations shown before a pack exists (notebook pre-test)."""

    n_cells: int | None = None
    n_questions: int | None = None
    scientific_claim: bool | None = None
    hypotheses: tuple[str, ...] = ()


@dataclass(frozen=True)
class CostConfig:
    """How to price a campaign: pins, prior volumes, parked counterfactuals."""

    pricing: str = "configs/models/pricing.yaml"
    scenario: str = "off_peak"
    volume_from: dict[str, str] = field(default_factory=dict)
    map_models: dict[str, str] = field(default_factory=dict)
    scale: dict[str, dict[str, Any]] = field(default_factory=dict)
    parked: tuple[dict[str, Any], ...] = ()


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
    pretest: PretestSpec | None = None


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
    pins: tuple[dict[str, Any], ...]
    source_path: Path
    notebook: str | None = None
    cost: CostConfig | None = None


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
        pins=_parse_pins(raw.get("pins")),
        source_path=Path(path).resolve(),
        notebook=_as_str(campaign.get("notebook")),
        cost=_parse_cost(raw.get("cost")),
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
        pretest=_parse_pretest(block.get("pretest")),
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
    exclude = item.get("exclude_question_categories") or []
    return AnalysisSpec(
        id=str(item.get("id") or "analysis"),
        title=str(item.get("title") or item.get("id") or "analysis"),
        group_by=tuple(str(g) for g in groups),
        metrics=tuple(str(m) for m in metrics) if metrics else default_metrics,
        plots=tuple(_parse_plot(p) for p in plots),
        source=str(item.get("source") or default_source),
        experiments=tuple(str(e) for e in experiments),
        where=_parse_where(item.get("where")),
        exclude_question_categories=tuple(int(v) for v in exclude),
        include_pins=bool(item.get("include_pins")),
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


def _parse_pretest(raw: Any) -> PretestSpec | None:
    if not raw or not isinstance(raw, dict):
        return None
    hyps = raw.get("hypotheses") or []
    if isinstance(hyps, str):
        hyps = [hyps]
    n_cells = raw.get("n_cells")
    n_questions = raw.get("n_questions")
    claim = raw.get("scientific_claim")
    return PretestSpec(
        n_cells=int(n_cells) if n_cells is not None else None,
        n_questions=int(n_questions) if n_questions is not None else None,
        scientific_claim=None if claim is None else bool(claim),
        hypotheses=tuple(str(h) for h in hyps),
    )


def _parse_cost(raw: Any) -> CostConfig | None:
    if not raw or not isinstance(raw, dict):
        return None
    volume = raw.get("volume_from") or {}
    mapping = raw.get("map_models") or {}
    scale = raw.get("scale") or {}
    parked = raw.get("parked") or []
    return CostConfig(
        pricing=str(raw.get("pricing") or "configs/models/pricing.yaml"),
        scenario=str(raw.get("scenario") or "off_peak"),
        volume_from={str(k): str(v) for k, v in volume.items()} if isinstance(volume, dict) else {},
        map_models={str(k): str(v) for k, v in mapping.items()} if isinstance(mapping, dict) else {},
        scale={str(k): dict(v) for k, v in scale.items() if isinstance(v, dict)} if isinstance(scale, dict) else {},
        parked=tuple(dict(item) for item in parked if isinstance(item, dict)),
    )


def _as_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _parse_where(raw: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not raw or not isinstance(raw, dict):
        return ()
    rows: list[tuple[str, tuple[str, ...]]] = []
    for key, vals in raw.items():
        if isinstance(vals, list):
            items = tuple(str(v) for v in vals)
        else:
            items = (str(vals),)
        rows.append((str(key), items))
    return tuple(rows)


def _parse_pins(raw: Any) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        if row.get("generation") is not None:
            row["generation"] = str(row["generation"])
        rows.append(row)
    return tuple(rows)
