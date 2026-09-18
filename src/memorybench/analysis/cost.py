"""Campaign cost tables: prior-volume estimates vs priced pack actuals.

Recipes live in analysis YAML ``cost:`` / ``pretest:``. Prices come from
``configs/models/pricing.yaml``. No LLM. Not an invoice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.analysis.campaign_plots import write_grouped_bar, write_metrics_grouped_bar
from src.locomo_eval.pricing import PricingTable, estimate_usd, load_pricing
from src.memorybench.analysis.load_campaign import CampaignConfig, ExperimentAnalysisRef

JUDGE_MODEL = "gpt-4o-mini"
COST_METRICS = ("usd_expected", "usd_actual")
_REPO = Path(__file__).resolve().parents[3]


@dataclass
class CostReport:
    """Tables written under analysis/ for one experiment or the whole campaign."""

    scope: str
    by_cell: pd.DataFrame
    by_stage: pd.DataFrame
    parked: pd.DataFrame
    csv_paths: list[Path]
    plot_paths: list[Path]


def render_cost(
    cfg: CampaignConfig,
    out_dir: Path,
    *,
    root: Path,
    experiment_id: str | None = None,
) -> CostReport | None:
    """Price expected (volume_from) vs actual (this pack). Skip if no ``cost:``."""
    if cfg.cost is None:
        return None
    table = _load_campaign_pricing(cfg, root)
    expected = _expected_rows(cfg, root, table, experiment_id)
    actual = _actual_rows(cfg, root, table, experiment_id)
    by_cell = _join_expected_actual(expected, actual)
    parked = _parked_table(cfg, expected, table)
    by_stage = _stage_table(cfg, by_cell, parked, experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: list[Path] = []
    plot_paths: list[Path] = []
    for name, frame in (
        ("cost_by_cell", by_cell),
        ("cost_by_stage", by_stage),
        ("cost_parked", parked),
    ):
        if frame.empty:
            continue
        path = tables_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        csv_paths.append(path)
    if not by_cell.empty and any(m in by_cell.columns for m in COST_METRICS):
        launched = by_cell.copy()
        if "launched" in launched.columns:
            launched = launched[launched["launched"] != False]
        metric_cols = [m for m in COST_METRICS if m in launched.columns]
        group_cols = [c for c in ("memory_method", "cell_model") if c in launched.columns]
        if not launched.empty and metric_cols and group_cols:
            plot_df = launched.groupby(group_cols, as_index=False)[metric_cols].sum()
            plot_df["cell"] = plot_df[group_cols].astype(str).agg(" × ".join, axis=1)
            dest = plots_dir / "cost_by_cell.png"
            written = write_metrics_grouped_bar(
                plot_df,
                group_col="cell",
                metrics=metric_cols,
                path=dest,
                title="Cost expected vs actual (launched cells)",
            )
            if written is not None:
                plot_paths.append(written)
    if not by_stage.empty:
        stage_metrics = [
            m for m in COST_METRICS if m in by_stage.columns and by_stage[m].notna().any()
        ]
        if stage_metrics and "experiment" in by_stage.columns:
            dest = plots_dir / "cost_by_stage.png"
            written = write_grouped_bar(
                by_stage.melt(
                    id_vars=["experiment"],
                    value_vars=stage_metrics,
                    var_name="cost_kind",
                    value_name="usd",
                ),
                x_col="experiment",
                hue_col="cost_kind",
                metric="usd",
                path=dest,
                title="Campaign cost by stage",
            )
            if written is not None:
                plot_paths.append(written)
    return CostReport(
        scope=experiment_id or cfg.id,
        by_cell=by_cell,
        by_stage=by_stage,
        parked=parked,
        csv_paths=csv_paths,
        plot_paths=plot_paths,
    )


def _load_campaign_pricing(cfg: CampaignConfig, root: Path) -> PricingTable:
    rel = Path(cfg.cost.pricing) if cfg.cost else Path("configs/models/pricing.yaml")
    for candidate in (rel, root / rel, _REPO / rel):
        if candidate.is_file():
            return load_pricing(candidate)
    return load_pricing()


def collect_pack_rows(pack: Path) -> pd.DataFrame:
    """One row per role × model from ``aggregate/by_run/*/cost.json``."""
    by_run = pack / "aggregate" / "by_run"
    if not by_run.is_dir():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cell_dir in sorted(p for p in by_run.iterdir() if p.is_dir()):
        meta = _read_json(cell_dir / "run_meta.json")
        cost = _read_json(cell_dir / "cost.json")
        if not cost:
            continue
        memory_method = str(meta.get("memory_type") or meta.get("memory_method") or "")
        reader_model = str(meta.get("reader_model") or "")
        teacher_model = str(meta.get("teacher_model") or "") or None
        cell_model = teacher_model or reader_model
        n_questions = int(meta.get("n_predictions") or 0)
        thinking = _cost_thinking(meta)
        for role in ("reader", "teacher"):
            bucket = cost.get(role) or {}
            by_model = bucket.get("by_model") or {}
            if not by_model and int(bucket.get("n_calls") or 0):
                by_model = {
                    reader_model if role == "reader" else (teacher_model or ""): bucket
                }
            for model, rec in by_model.items():
                rows.append(
                    _token_row(
                        memory_method=memory_method,
                        role=role,
                        model=str(model),
                        cell_model=cell_model,
                        n_calls=int(rec.get("n_calls") or 0),
                        prompt=int(rec.get("prompt_tokens") or 0),
                        completion=int(rec.get("completion_tokens") or 0),
                        n_questions=n_questions,
                        thinking=thinking,
                    )
                )
        judge_tokens = _judge_tokens(cell_dir)
        if judge_tokens:
            rows.append(
                _token_row(
                    memory_method=memory_method,
                    role="judge",
                    model=JUDGE_MODEL,
                    cell_model=cell_model,
                    n_calls=int(judge_tokens["n_judged"]),
                    prompt=int(judge_tokens["prompt_tokens"]),
                    completion=int(judge_tokens["completion_tokens"]),
                    n_questions=n_questions,
                    thinking=thinking,
                )
            )
    return pd.DataFrame(rows)


def _expected_rows(
    cfg: CampaignConfig,
    root: Path,
    table: PricingTable,
    experiment_id: str | None,
) -> pd.DataFrame:
    cost = cfg.cost
    if cost is None:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for exp_id, ref in cfg.experiments.items():
        if experiment_id and exp_id != experiment_id:
            continue
        pack = _volume_pack(cost, exp_id, ref, root)
        if pack is None:
            continue
        raw = collect_pack_rows(pack)
        if raw.empty:
            continue
        raw = _map_models(raw, cost.map_models)
        raw["experiment"] = exp_id
        frames.append(raw)
    if cost.scale and experiment_id:
        scaled = _scale_from_baseline(cfg, root, experiment_id)
        if scaled is not None and not scaled.empty:
            frames = [scaled]
    elif cost.scale and experiment_id is None:
        for exp_id in cost.scale:
            scaled = _scale_from_baseline(cfg, root, exp_id)
            if scaled is None or scaled.empty:
                continue
            frames = [f for f in frames if f.empty or f["experiment"].iloc[0] != exp_id]
            frames.append(scaled)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["usd"] = [
        estimate_usd(
            row.model,
            int(row.prompt_tokens),
            int(row.completion_tokens),
            scenario=cost.scenario,
            table=table,
        )
        for row in out.itertuples(index=False)
    ]
    return out


def _actual_rows(
    cfg: CampaignConfig,
    root: Path,
    table: PricingTable,
    experiment_id: str | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for exp_id, ref in cfg.experiments.items():
        if experiment_id and exp_id != experiment_id:
            continue
        pack = root / ref.pack if not ref.pack.is_absolute() else ref.pack
        raw = collect_pack_rows(pack)
        if raw.empty:
            continue
        raw["experiment"] = exp_id
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    scenario = cfg.cost.scenario if cfg.cost else "off_peak"
    out["usd"] = [
        estimate_usd(
            row.model,
            int(row.prompt_tokens),
            int(row.completion_tokens),
            scenario=scenario,
            table=table,
        )
        for row in out.itertuples(index=False)
    ]
    return out


def _volume_pack(
    cost: Any,
    exp_id: str,
    ref: ExperimentAnalysisRef,
    root: Path,
) -> Path | None:
    if exp_id in cost.scale:
        return None
    mapped = cost.volume_from.get(exp_id)
    if mapped:
        pack = Path(mapped)
    else:
        pack = ref.pack
    dest = pack if pack.is_absolute() else root / pack
    if not (dest / "aggregate" / "by_run").is_dir():
        return None
    return dest


def _scale_from_baseline(
    cfg: CampaignConfig,
    root: Path,
    experiment_id: str,
) -> pd.DataFrame | None:
    spec = (cfg.cost.scale if cfg.cost else {}) .get(experiment_id)
    if not spec:
        return None
    source_id = str(spec.get("from_experiment") or "baseline")
    source_ref = cfg.experiments.get(source_id)
    if source_ref is None or cfg.cost is None:
        return None
    pack = _volume_pack_direct(cfg.cost, source_id, source_ref, root)
    if pack is None:
        return None
    raw = collect_pack_rows(pack)
    if raw.empty:
        return None
    raw = _map_models(raw, cfg.cost.map_models)
    methods = spec.get("memory_methods") or []
    if methods:
        raw = raw[raw["memory_method"].astype(str).isin([str(m) for m in methods])]
    target = cfg.experiments[experiment_id]
    n_full = _max_questions(raw)
    n_smoke = int(target.pretest.n_questions or 0) if target.pretest else 0
    if n_full <= 0 or n_smoke <= 0:
        return None
    frac = n_smoke / n_full
    out = raw.copy()
    for col in ("n_calls", "prompt_tokens", "completion_tokens", "n_questions"):
        out[col] = (out[col].astype(float) * frac).round().astype(int)
    out["experiment"] = experiment_id
    return out


def _volume_pack_direct(cost: Any, exp_id: str, ref: ExperimentAnalysisRef, root: Path) -> Path | None:
    mapped = cost.volume_from.get(exp_id)
    pack = Path(mapped) if mapped else ref.pack
    dest = pack if pack.is_absolute() else root / pack
    if not (dest / "aggregate" / "by_run").is_dir():
        return None
    return dest


def _map_models(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if df.empty or not mapping:
        return df
    out = df.copy()
    out["model"] = out["model"].replace(mapping)
    out["cell_model"] = out["cell_model"].replace(mapping)
    return out


def _join_expected_actual(expected: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "memory_method", "role", "model", "cell_model"]
    if "thinking" in expected.columns and "thinking" in actual.columns:
        keys.append("thinking")
    exp = _rename_usd(expected, "usd_expected")
    act = _rename_usd(actual, "usd_actual")
    if exp.empty and act.empty:
        return pd.DataFrame()
    if exp.empty:
        act = act.copy()
        act["usd_expected"] = None
        return act
    if act.empty:
        exp = exp.copy()
        exp["usd_actual"] = None
        return exp
    merged = pd.merge(exp, act, on=keys, how="outer", suffixes=("_exp", "_act"))
    for col in ("n_calls", "prompt_tokens", "completion_tokens", "n_questions"):
        if f"{col}_act" in merged.columns:
            merged[col] = merged[f"{col}_act"].fillna(merged.get(f"{col}_exp"))
        elif f"{col}_exp" in merged.columns:
            merged[col] = merged[f"{col}_exp"]
    keep = keys + [
        c
        for c in (
            "n_calls",
            "prompt_tokens",
            "completion_tokens",
            "n_questions",
            "usd_expected",
            "usd_actual",
        )
        if c in merged.columns
    ]
    out = merged[keep].copy()
    if "usd_expected" in out.columns and "usd_actual" in out.columns:
        out["usd_delta"] = out["usd_actual"] - out["usd_expected"]
    out["launched"] = True
    return out


def _parked_table(
    cfg: CampaignConfig, expected: pd.DataFrame, table: PricingTable
) -> pd.DataFrame:
    if cfg.cost is None or not cfg.cost.parked or expected.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for item in cfg.cost.parked:
        copy_from = str(item.get("copy_from") or "")
        api_id = str(item.get("api_model_id") or "")
        if not copy_from or not api_id:
            continue
        subset = expected[expected["cell_model"].astype(str) == copy_from].copy()
        if subset.empty:
            continue
        subset["cell_model"] = api_id
        swap = subset["role"].isin(["reader", "teacher"])
        subset.loc[swap, "model"] = api_id
        subset["usd"] = [
            estimate_usd(
                row.model,
                int(row.prompt_tokens),
                int(row.completion_tokens),
                scenario=cfg.cost.scenario,
                table=table,
            )
            for row in subset.itertuples(index=False)
        ]
        subset["launched"] = False
        subset["parked_id"] = str(item.get("id") or api_id)
        subset["display_name"] = str(item.get("display_name") or api_id)
        rows.append(subset)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _stage_table(
    cfg: CampaignConfig,
    by_cell: pd.DataFrame,
    parked: pd.DataFrame,
    experiment_id: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exp_ids = [experiment_id] if experiment_id else list(cfg.experiments)
    for exp_id in exp_ids:
        ref = cfg.experiments.get(exp_id)
        if ref is None:
            continue
        part = by_cell[by_cell["experiment"] == exp_id] if not by_cell.empty else pd.DataFrame()
        expected = _sum_or_none(part["usd_expected"] if not part.empty and "usd_expected" in part.columns else None)
        actual = _sum_or_none(part["usd_actual"] if not part.empty and "usd_actual" in part.columns else None)
        rows.append(
            {
                "experiment": exp_id,
                "n_cells_expected": ref.pretest.n_cells if ref.pretest else None,
                "scientific_claim": ref.pretest.scientific_claim if ref.pretest else None,
                "usd_expected": expected,
                "usd_actual": actual,
            }
        )
    if parked is not None and not parked.empty and experiment_id is None:
        parked_usd = parked["usd"].sum() if "usd" in parked.columns else None
        rows.append(
            {
                "experiment": "parked",
                "n_cells_expected": None,
                "scientific_claim": False,
                "usd_expected": None if parked_usd is None or pd.isna(parked_usd) else round(float(parked_usd), 4),
                "usd_actual": None,
            }
        )
    return pd.DataFrame(rows)


def _token_row(
    *,
    memory_method: str,
    role: str,
    model: str,
    cell_model: str,
    n_calls: int,
    prompt: int,
    completion: int,
    n_questions: int,
    thinking: str | None = None,
) -> dict[str, Any]:
    row = {
        "memory_method": memory_method,
        "role": role,
        "model": model,
        "cell_model": cell_model,
        "n_calls": n_calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "n_questions": n_questions,
    }
    if thinking is not None:
        row["thinking"] = thinking
    return row


def _cost_thinking(meta: dict[str, Any]) -> str | None:
    from scripts.analysis.campaign_tables import _thinking_token

    raw = meta.get("teacher_thinking")
    if raw is None:
        raw = meta.get("reader_thinking")
    return _thinking_token(raw)


def _judge_tokens(cell_dir: Path) -> dict[str, int] | None:
    metrics = _read_json(cell_dir / "autorater" / "autorater_metrics.json")
    if not metrics:
        return None
    block = metrics.get("metrics") or metrics
    total = int(block.get("autorater_total_tokens_sum") or 0)
    n_judged = int(metrics.get("n_judged") or block.get("n") or 0)
    if total <= 0:
        return None
    # Judge completions are short CORRECT/WRONG lines; pin 20 tok/call.
    completion = min(total, n_judged * 20)
    prompt = max(0, total - completion)
    return {"n_judged": n_judged, "prompt_tokens": prompt, "completion_tokens": completion}


def _rename_usd(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "usd" in out.columns:
        out = out.rename(columns={"usd": col})
    return out


def _max_questions(df: pd.DataFrame) -> int:
    if df.empty or "n_questions" not in df.columns:
        return 0
    return int(df["n_questions"].max() or 0)


def _sum_or_none(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    known = pd.to_numeric(series, errors="coerce").dropna()
    if known.empty:
        return None
    return round(float(known.sum()), 4)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}
