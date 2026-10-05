"""Year-family robustness tables (no LLM, no plots).

Mean tables from YAML recipes stay the source. These functions label
year-to-year moves, family gaps, method-rank flips, thinking deltas, and
category holes so a notebook can read threats to validity without groupby.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from scripts.analysis.campaign_tables import GENERATION_AXIS, _family_label

MOVE_EPS = 0.02
HOLE_MAX = 0.40
CEILING_J = 0.90
APPROACHING_J = 0.80
SCORE_METRICS = ("locomo_f1", "judge_score", "token_f1", "exact_match")
RESOURCE_METRICS = (
    "agent_latency_seconds",
    "writer_latency_seconds",
    "agent_reasoning_tokens",
    "writer_reasoning_tokens",
    "total_latency_seconds_p50",
    "total_latency_seconds_p95",
    "search_latency_seconds_p50",
    "search_latency_seconds_p95",
    "usd_actual",
    "usd_expected",
    "prompt_tokens",
    "completion_tokens",
    "locomo_f1_per_second",
    "judge_score_per_second",
    "locomo_f1_per_usd",
    "judge_score_per_usd",
    "locomo_f1_per_k_reasoning",
    "judge_score_per_k_reasoning",
)
LOWER_BETTER = frozenset(
    {
        "agent_latency_seconds",
        "writer_latency_seconds",
        "agent_reasoning_tokens",
        "writer_reasoning_tokens",
        "total_latency_seconds_p50",
        "total_latency_seconds_p95",
        "search_latency_seconds_p50",
        "search_latency_seconds_p95",
        "usd_actual",
        "usd_expected",
        "prompt_tokens",
        "completion_tokens",
    }
)
_SKIP_KEYS = frozenset(SCORE_METRICS) | frozenset(RESOURCE_METRICS) | {
    "n",
    "delta",
    "move",
    "gap",
    "gap_move",
    "rank",
    "label",
    "headroom",
    "band",
}


def classify_delta(delta: float | None, eps: float = MOVE_EPS) -> str:
    """improving / declining / not_moving; missing when the pair is absent."""
    if delta is None or pd.isna(delta):
        return "missing"
    if float(delta) > eps:
        return "improving"
    if float(delta) < -eps:
        return "declining"
    return "not_moving"


def classify_gap_change(
    earlier: float | None, later: float | None, eps: float = MOVE_EPS
) -> str:
    """Whether |OpenAI − DeepSeek| shrank, grew, or held between two years."""
    if earlier is None or later is None or pd.isna(earlier) or pd.isna(later):
        return "missing"
    early = abs(float(earlier))
    late = abs(float(later))
    if late + eps < early:
        return "gap_closing"
    if late > early + eps:
        return "gap_widening"
    return "gap_stable"


def year_deltas(
    table: pd.DataFrame,
    metrics: list[str] | tuple[str, ...] | None = None,
    *,
    eps: float = MOVE_EPS,
) -> pd.DataFrame:
    """Consecutive year deltas within the same keys (including result_source).

    2024 paper pins do not pair with 2025 live rows. Live 2025 vs 2026 is the
    matched frozen-reader comparison.
    """
    if table.empty or "generation" not in table.columns:
        return pd.DataFrame()
    metrics = _metrics_in(table, metrics)
    keys = _keys(table, extra_drop=("generation",))
    if not metrics or not keys:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = table.groupby([k for k in keys], dropna=False, sort=False)
    for key_vals, piece in grouped:
        key_map = _key_map(keys, key_vals)
        years = [y for y in GENERATION_AXIS if y in set(piece["generation"].astype(str))]
        by_year = {
            str(year): piece[piece["generation"].astype(str) == year]
            for year in years
        }
        for metric in metrics:
            values = {
                year: _first_num(by_year[year], metric)
                for year in years
            }
            for earlier, later in zip(years, years[1:]):
                a = values[earlier]
                b = values[later]
                delta = None if a is None or b is None else b - a
                signed = None if delta is None else (-delta if metric in LOWER_BETTER else delta)
                row = dict(key_map)
                row.update(
                    {
                        "metric": metric,
                        "from_generation": earlier,
                        "to_generation": later,
                        "from_value": a,
                        "to_value": b,
                        "delta": delta,
                        "move": classify_delta(signed, eps),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def family_gaps(
    table: pd.DataFrame,
    metrics: list[str] | tuple[str, ...] | None = None,
    *,
    left_family: str = "OpenAI",
    right_family: str = "DeepSeek",
    eps: float = MOVE_EPS,
) -> pd.DataFrame:
    """left − right per year, then whether |gap| closed or widened."""
    if table.empty or "model_family" not in table.columns:
        return pd.DataFrame()
    if "generation" not in table.columns:
        return pd.DataFrame()
    metrics = _metrics_in(table, metrics)
    keys = _keys(table, extra_drop=("model_family",))
    if not metrics:
        return pd.DataFrame()
    group_cols = [k for k in keys if k != "generation"] or []
    rows: list[dict[str, Any]] = []
    if group_cols:
        grouped = table.groupby(group_cols, dropna=False, sort=False)
    else:
        grouped = [((), table)]
    for key_vals, piece in grouped:
        key_map = _key_map(group_cols, key_vals) if group_cols else {}
        years = [y for y in GENERATION_AXIS if y in set(piece["generation"].astype(str))]
        for metric in metrics:
            year_rows: list[dict[str, Any]] = []
            gaps: dict[str, float | None] = {}
            for year in years:
                slice_year = piece[piece["generation"].astype(str) == year]
                left = _family_num(slice_year, left_family, metric)
                right = _family_num(slice_year, right_family, metric)
                gap = None if left is None or right is None else left - right
                gaps[year] = gap
                year_rows.append(
                    {
                        **key_map,
                        "metric": metric,
                        "generation": year,
                        "left_family": left_family,
                        "right_family": right_family,
                        "left_value": left,
                        "right_value": right,
                        "gap": gap,
                    }
                )
            moves = {
                later: classify_gap_change(gaps[earlier], gaps[later], eps)
                for earlier, later in zip(years, years[1:])
            }
            from_gen = {later: earlier for earlier, later in zip(years, years[1:])}
            for row in year_rows:
                year = str(row["generation"])
                row["from_generation"] = from_gen.get(year)
                row["gap_move"] = moves.get(year, "missing")
                rows.append(row)
    return pd.DataFrame(rows)


def method_ranks(
    table: pd.DataFrame,
    metric: str = "locomo_f1",
) -> pd.DataFrame:
    """Rank memory methods within generation × family × thinking × source."""
    if table.empty or metric not in table.columns or "memory_method" not in table.columns:
        return pd.DataFrame()
    keys = [
        c
        for c in (
            "generation",
            "model_family",
            "thinking",
            "result_source",
        )
        if c in table.columns
    ]
    if not keys:
        return pd.DataFrame()
    work = table.dropna(subset=[metric]).copy()
    if work.empty:
        return pd.DataFrame()
    work["rank"] = work.groupby(keys, dropna=False)[metric].rank(
        ascending=False, method="min"
    )
    keep = keys + ["memory_method", metric, "rank"]
    if "n" in work.columns:
        keep.append("n")
    out = work[keep].sort_values(keys + ["rank"], kind="mergesort").reset_index(drop=True)
    return out


def rank_flips(ranks: pd.DataFrame) -> pd.DataFrame:
    """Pairs of (family or year) whose method order disagrees — a validity flag."""
    if ranks.empty or "memory_method" not in ranks.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if "generation" in ranks.columns and "model_family" in ranks.columns:
        rows.extend(_order_mismatches(ranks, "model_family", "generation"))
        rows.extend(_order_mismatches(ranks, "generation", "model_family"))
    return pd.DataFrame(rows)


def thinking_deltas(
    table: pd.DataFrame,
    metrics: list[str] | tuple[str, ...] | None = None,
    *,
    eps: float = MOVE_EPS,
) -> pd.DataFrame:
    """thinking on minus off within the other keys."""
    if table.empty or "thinking" not in table.columns:
        return pd.DataFrame()
    metrics = _metrics_in(table, metrics)
    keys = _keys(table, extra_drop=("thinking",))
    if not metrics or not keys:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = table.groupby(keys, dropna=False, sort=False)
    for key_vals, piece in grouped:
        key_map = _key_map(keys, key_vals)
        on_rows = piece[piece["thinking"].astype(str).str.lower() == "on"]
        off_rows = piece[piece["thinking"].astype(str).str.lower() == "off"]
        for metric in metrics:
            on_val = _first_num(on_rows, metric)
            off_val = _first_num(off_rows, metric)
            delta = None if on_val is None or off_val is None else on_val - off_val
            signed = None if delta is None else (-delta if metric in LOWER_BETTER else delta)
            move = classify_delta(signed, eps)
            if move == "improving":
                label = "thinking_helps"
            elif move == "declining":
                label = "thinking_hurts"
            elif move == "not_moving":
                label = "thinking_flat"
            else:
                label = "missing"
            row = dict(key_map)
            row.update(
                {
                    "metric": metric,
                    "off_value": off_val,
                    "on_value": on_val,
                    "delta": delta,
                    "move": move,
                    "label": label,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def efficiency(table: pd.DataFrame) -> pd.DataFrame:
    """Score per second, per $1, and per 1k reasoning tokens (ROI numerators)."""
    if table.empty:
        return table
    out = table.copy()
    scores = [m for m in ("locomo_f1", "judge_score") if m in out.columns]
    _ratio(out, scores, "agent_latency_seconds", "per_second")
    _ratio(out, scores, "writer_latency_seconds", "per_writer_second")
    _ratio(out, scores, "usd_actual", "per_usd")
    if "agent_reasoning_tokens" in out.columns:
        denom = pd.to_numeric(out["agent_reasoning_tokens"], errors="coerce") / 1000.0
        for score in scores:
            out[f"{score}_per_k_reasoning"] = pd.to_numeric(out[score], errors="coerce") / denom.replace(
                0, pd.NA
            )
    return out


def saturation(
    table: pd.DataFrame,
    score: str = "judge_score",
    *,
    ceiling: float = CEILING_J,
    approaching: float = APPROACHING_J,
    eps: float = MOVE_EPS,
) -> pd.DataFrame:
    """Benchmark headroom plus costly-vs-flat year moves (saturation / ROI)."""
    if table.empty or score not in table.columns:
        return pd.DataFrame()
    keys = _keys(table, extra_drop=("generation",))
    if "generation" not in table.columns or not keys:
        rows = []
        for _, piece in table.iterrows():
            val = _as_float(piece.get(score))
            row = {k: piece.get(k) for k in table.columns}
            row["metric"] = score
            row["headroom"] = None if val is None else 1.0 - val
            row["band"] = _ceiling_band(val, ceiling, approaching)
            rows.append(row)
        return pd.DataFrame(rows)
    rows: list[dict[str, Any]] = []
    grouped = table.groupby(keys, dropna=False, sort=False)
    for key_vals, piece in grouped:
        key_map = _key_map(keys, key_vals)
        years = [y for y in GENERATION_AXIS if y in set(piece["generation"].astype(str))]
        values = {
            year: _first_num(piece[piece["generation"].astype(str) == year], score)
            for year in years
        }
        for year in years:
            val = values[year]
            row = dict(key_map)
            row.update(
                {
                    "metric": score,
                    "generation": year,
                    "value": val,
                    "headroom": None if val is None else round(1.0 - val, 4),
                    "band": _ceiling_band(val, ceiling, approaching),
                }
            )
            rows.append(row)
        for earlier, later in zip(years, years[1:]):
            a = values[earlier]
            b = values[later]
            delta = None if a is None or b is None else b - a
            score_move = classify_delta(delta, eps)
            later_slice = piece[piece["generation"].astype(str) == later]
            earlier_slice = piece[piece["generation"].astype(str) == earlier]
            resource_up = _resource_rose(earlier_slice, later_slice, eps)
            label = _saturation_move_label(
                score_move,
                resource_up,
                _ceiling_band(b, ceiling, approaching),
            )
            for row in rows:
                if (
                    row.get("generation") == later
                    and all(row.get(k) == key_map.get(k) for k in key_map)
                ):
                    row["from_generation"] = earlier
                    row["delta"] = delta
                    row["move"] = score_move
                    row["label"] = label
                    break
    return pd.DataFrame(rows)


def attach_cell_cost(quality: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    """Join priced cell USD onto a year-family mean table (ROI denominator)."""
    if quality.empty or cost is None or cost.empty or "usd_actual" not in cost.columns:
        return quality
    work = cost.copy()
    if "experiment" in work.columns:
        work["generation"] = work["experiment"].map(_experiment_year)
    if "cell_model" in work.columns:
        work["model_family"] = work["cell_model"].map(_family_label)
    keys = [
        c
        for c in ("generation", "model_family", "memory_method", "thinking")
        if c in work.columns and c in quality.columns
    ]
    if not keys:
        return quality
    extra = [c for c in ("usd_actual", "prompt_tokens", "completion_tokens") if c in work.columns]
    summed = work.groupby(keys, dropna=False, as_index=False)[extra].sum()
    return quality.merge(summed, on=keys, how="left")


def _ratio(out: pd.DataFrame, scores: list[str], denom_col: str, suffix: str) -> None:
    if denom_col not in out.columns:
        return
    denom = pd.to_numeric(out[denom_col], errors="coerce")
    for score in scores:
        out[f"{score}_{suffix}"] = pd.to_numeric(out[score], errors="coerce") / denom.replace(
            0, pd.NA
        )


def _ceiling_band(value: float | None, ceiling: float, approaching: float) -> str:
    if value is None:
        return "missing"
    if value >= ceiling:
        return "near_ceiling"
    if value >= approaching:
        return "approaching_ceiling"
    return "open_headroom"


def _resource_rose(earlier: pd.DataFrame, later: pd.DataFrame, eps: float) -> bool:
    for col in LOWER_BETTER:
        if col not in earlier.columns or col not in later.columns:
            continue
        a = _first_num(earlier, col)
        b = _first_num(later, col)
        if a is None or b is None:
            continue
        if b - a > eps:
            return True
    return False


def _saturation_move_label(score_move: str, resource_up: bool, later_band: str) -> str:
    if later_band == "near_ceiling" and score_move == "not_moving":
        return "saturated_flat"
    if score_move == "improving" and resource_up:
        return "costly_gain"
    if score_move == "improving":
        return "efficient_gain"
    if score_move in {"not_moving", "declining"} and resource_up:
        return "diminishing_returns"
    if score_move == "not_moving":
        return "flat"
    return score_move


def _experiment_year(exp_id: Any) -> str:
    text = str(exp_id or "")
    if "2026" in text:
        return "2026"
    if "2024" in text:
        return "2024"
    return "2025"


def category_holes(
    table: pd.DataFrame,
    metric: str = "locomo_f1",
    *,
    hole_max: float = HOLE_MAX,
    eps: float = MOVE_EPS,
) -> pd.DataFrame:
    """Per-category scores plus hole / persistent_hole / closing_hole labels."""
    if table.empty or "question_category" not in table.columns:
        return pd.DataFrame()
    if metric not in table.columns or "generation" not in table.columns:
        return pd.DataFrame()
    keys = _keys(table, extra_drop=("generation",))
    rows: list[dict[str, Any]] = []
    grouped = table.groupby(keys, dropna=False, sort=False)
    for key_vals, piece in grouped:
        key_map = _key_map(keys, key_vals)
        years = [y for y in GENERATION_AXIS if y in set(piece["generation"].astype(str))]
        values = {
            year: _first_num(
                piece[piece["generation"].astype(str) == year], metric
            )
            for year in years
        }
        hole_years = [
            year
            for year, val in values.items()
            if val is not None and val < hole_max
        ]
        present = [year for year, val in values.items() if val is not None]
        if hole_years and present and hole_years == present:
            hole_label = "persistent_hole"
        elif hole_years and present and present[-1] not in hole_years:
            hole_label = "closing_hole"
        elif hole_years and present and present[-1] in hole_years and present[0] not in hole_years:
            hole_label = "new_hole"
        elif hole_years:
            hole_label = "hole"
        else:
            hole_label = "not_a_hole"
        first = values[present[0]] if present else None
        last = values[present[-1]] if present else None
        delta = None if first is None or last is None else last - first
        row = dict(key_map)
        row.update(
            {
                "metric": metric,
                "first_generation": present[0] if present else None,
                "last_generation": present[-1] if present else None,
                "first_value": first,
                "last_value": last,
                "delta": delta,
                "move": classify_delta(delta, eps),
                "hole_years": ",".join(hole_years),
                "label": hole_label,
            }
        )
        for year, val in values.items():
            row[f"value_{year}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def pin_gaps(
    table: pd.DataFrame,
    metric: str = "judge_score",
    *,
    live_generation: str = "2025",
    pin_generation: str = "2024",
) -> pd.DataFrame:
    """2024 paper/local_clone vs later live OpenAI on the same memory method.

    Not a matched frozen-reader delta — different reader stack. Labelled so the
    notebook does not treat it as 2025→2026 live robustness.
    """
    if table.empty or metric not in table.columns:
        return pd.DataFrame()
    needed = {"generation", "model_family"}
    method_col = "paper_method" if "paper_method" in table.columns else "memory_method"
    source_col = "compare_source" if "compare_source" in table.columns else "result_source"
    needed = needed | {method_col, source_col}
    if not needed.issubset(table.columns):
        return pd.DataFrame()
    pins = table[
        (table["generation"].astype(str) == pin_generation)
        & (table[source_col].astype(str).isin(("paper", "local_clone", "local clone")))
    ]
    live_tokens = ("live", "live model", "gpt-4o-mini + Codex")
    live = table[
        (table["generation"].astype(str) == live_generation)
        & (table[source_col].astype(str).isin(live_tokens))
        & (table["model_family"].astype(str) == "OpenAI")
    ]
    if "thinking" in live.columns:
        off = live[live["thinking"].astype(str).str.lower() == "off"]
        if not off.empty:
            live = off
    rows: list[dict[str, Any]] = []
    for _, pin in pins.iterrows():
        method = str(pin.get(method_col))
        source = str(pin.get(source_col))
        pin_val = _as_float(pin.get(metric))
        live_rows = live[live[method_col].astype(str) == method]
        live_val = _first_num(live_rows, metric)
        delta = None if pin_val is None or live_val is None else live_val - pin_val
        move = classify_delta(delta)
        rows.append(
            {
                "memory_method": method,
                "pin_source": source,
                "pin_generation": pin_generation,
                "live_generation": live_generation,
                "metric": metric,
                "pin_value": pin_val,
                "live_value": live_val,
                "delta": delta,
                "move": move,
                "label": "pin_vs_live_not_matched_stack",
            }
        )
    return pd.DataFrame(rows)


def j_f1_gap(table: pd.DataFrame) -> pd.DataFrame:
    """Judge J minus LoCoMo F1. Large positive = paraphrase / verbosity.

    Same predicted string on both metrics. Not a scorer bug.
    """
    if (
        table.empty
        or "judge_score" not in table.columns
        or "locomo_f1" not in table.columns
    ):
        return pd.DataFrame()
    out = table.copy()
    out["j_minus_f1"] = pd.to_numeric(
        out["judge_score"], errors="coerce"
    ) - pd.to_numeric(out["locomo_f1"], errors="coerce")
    return out


def takeaway_contrast(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metrics: list[str] | tuple[str, ...],
    *,
    takeaway_id: str,
    title: str,
    claim: str,
    finding: str,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    """One row per metric: left minus right, plus the frozen finding text."""
    wanted = list(metrics) or [
        m
        for m in ("judge_score", "locomo_f1", "token_f1", "exact_match")
        if m in left.columns or m in right.columns
    ]
    if not wanted:
        wanted = ["_"]
    rows: list[dict[str, Any]] = []
    n_left = 0 if left.empty or "n" not in left.columns else int(
        pd.to_numeric(left["n"], errors="coerce").fillna(0).sum()
    )
    n_right = 0 if right.empty or "n" not in right.columns else int(
        pd.to_numeric(right["n"], errors="coerce").fillna(0).sum()
    )
    for metric in wanted:
        name = metric if metric != "_" else ""
        lv = _mean_num(left, name) if name else None
        rv = _mean_num(right, name) if name else None
        delta = None if lv is None or rv is None else lv - rv
        rows.append(
            {
                "takeaway_id": takeaway_id,
                "title": title,
                "claim": claim,
                "finding": finding,
                "left_label": left_label,
                "right_label": right_label,
                "n_left": n_left or None,
                "n_right": n_right or None,
                "metric": name or None,
                "left_value": lv,
                "right_value": rv,
                "delta": delta,
            }
        )
    return pd.DataFrame(rows)


def filter_where(
    df: pd.DataFrame, where: tuple[tuple[str, tuple[str, ...]], ...]
) -> pd.DataFrame:
    out = df
    for col, values in where:
        if col not in out.columns:
            continue
        out = out[out[col].astype(str).isin(values)]
    return out


def _mean_num(piece: pd.DataFrame, metric: str) -> float | None:
    if not metric or piece.empty or metric not in piece.columns:
        return None
    series = pd.to_numeric(piece[metric], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def series_gaps(
    table: pd.DataFrame,
    *,
    metric: str,
    series_col: str,
    left: str,
    right: str,
) -> pd.DataFrame:
    """Category (or other key) gap: left minus right, largest absolute first.

    Used after a mean table that already splits one metric by two stacks
    (4o-mini vs Codex, or 2024 model vs harness).
    """
    if (
        table.empty
        or not series_col
        or series_col not in table.columns
        or metric not in table.columns
        or not left
        or not right
    ):
        return pd.DataFrame()
    keys = [c for c in table.columns if c not in {series_col, metric, "n"}]
    left_df = table.loc[table[series_col].astype(str) == left, [*keys, metric]].rename(
        columns={metric: "left_value"}
    )
    right_df = table.loc[table[series_col].astype(str) == right, [*keys, metric]].rename(
        columns={metric: "right_value"}
    )
    if left_df.empty or right_df.empty:
        return pd.DataFrame()
    if keys:
        merged = left_df.merge(right_df, on=keys, how="inner")
    else:
        merged = pd.DataFrame(
            {
                "left_value": [float(left_df["left_value"].iloc[0])],
                "right_value": [float(right_df["right_value"].iloc[0])],
            }
        )
    merged["left_value"] = pd.to_numeric(merged["left_value"], errors="coerce")
    merged["right_value"] = pd.to_numeric(merged["right_value"], errors="coerce")
    merged["delta"] = merged["left_value"] - merged["right_value"]
    merged["left_label"] = left
    merged["right_label"] = right
    merged["metric"] = metric
    merged["_abs"] = merged["delta"].abs()
    merged = merged.sort_values("_abs", ascending=False, kind="mergesort").drop(columns=["_abs"])
    return merged.reset_index(drop=True)


def render_insight(kind: str, table: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Dispatch one YAML insight kind onto an already-aggregated mean table."""
    metrics = kwargs.get("metrics")
    eps = float(kwargs.get("move_eps") or MOVE_EPS)
    if kind == "year_deltas":
        return year_deltas(table, metrics, eps=eps)
    if kind == "family_gaps":
        return family_gaps(
            table,
            metrics,
            left_family=str(kwargs.get("left_family") or "OpenAI"),
            right_family=str(kwargs.get("right_family") or "DeepSeek"),
            eps=eps,
        )
    if kind == "method_ranks":
        return method_ranks(table, str(kwargs.get("metric") or "locomo_f1"))
    if kind == "rank_flips":
        return rank_flips(table)
    if kind == "thinking_deltas":
        return thinking_deltas(table, metrics, eps=eps)
    if kind == "category_holes":
        return category_holes(
            table,
            str(kwargs.get("metric") or "locomo_f1"),
            hole_max=float(kwargs.get("hole_max") or HOLE_MAX),
            eps=eps,
        )
    if kind == "pin_gaps":
        return pin_gaps(
            table,
            str(kwargs.get("metric") or "judge_score"),
            live_generation=str(kwargs.get("live_generation") or "2025"),
        )
    if kind == "efficiency":
        return efficiency(table)
    if kind == "saturation":
        return saturation(
            table,
            str(kwargs.get("metric") or "judge_score"),
            ceiling=float(kwargs.get("hole_max") or CEILING_J),
            eps=eps,
        )
    if kind == "j_f1_gap":
        return j_f1_gap(table)
    if kind == "series_gaps":
        return series_gaps(
            table,
            metric=str(kwargs.get("metric") or "judge_score"),
            series_col=str(kwargs.get("series_col") or ""),
            left=str(kwargs.get("left_series") or ""),
            right=str(kwargs.get("right_series") or ""),
        )
    raise ValueError(f"unknown insight kind {kind}")


def _metrics_in(
    table: pd.DataFrame, metrics: list[str] | tuple[str, ...] | None
) -> list[str]:
    wanted = list(metrics) if metrics else [
        m for m in SCORE_METRICS + RESOURCE_METRICS if m in table.columns
    ]
    return [m for m in wanted if m in table.columns]


def _keys(table: pd.DataFrame, extra_drop: tuple[str, ...] = ()) -> list[str]:
    drop = _SKIP_KEYS | set(extra_drop)
    return [c for c in table.columns if c not in drop and not str(c).startswith("value_")]


def _key_map(keys: list[str], key_vals: Any) -> dict[str, Any]:
    if not keys:
        return {}
    if len(keys) == 1:
        vals = (key_vals,)
    else:
        vals = tuple(key_vals)
    return {k: v for k, v in zip(keys, vals)}


def _first_num(piece: pd.DataFrame, metric: str) -> float | None:
    if piece.empty or metric not in piece.columns:
        return None
    series = pd.to_numeric(piece[metric], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _family_num(piece: pd.DataFrame, family: str, metric: str) -> float | None:
    if "model_family" not in piece.columns:
        return None
    return _first_num(piece[piece["model_family"].astype(str) == family], metric)


def _as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _order_mismatches(
    ranks: pd.DataFrame, split_col: str, hold_col: str
) -> list[dict[str, Any]]:
    """When method rank order disagrees across split_col, holding hold_col fixed."""
    extra = [
        c
        for c in ("thinking", "result_source")
        if c in ranks.columns and c not in {split_col, hold_col}
    ]
    group_cols = [hold_col] + extra
    rows: list[dict[str, Any]] = []
    grouped = ranks.groupby(group_cols, dropna=False, sort=False)
    for key_vals, piece in grouped:
        key_map = _key_map(group_cols, key_vals)
        orders: dict[str, tuple[str, ...]] = {}
        for split, part in piece.groupby(split_col, dropna=False, sort=False):
            ordered = tuple(
                str(v)
                for v in part.sort_values("rank", kind="mergesort")["memory_method"].tolist()
            )
            orders[str(split)] = ordered
        labels = list(orders)
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                if orders[left] != orders[right]:
                    row = dict(key_map)
                    row.update(
                        {
                            "label": "rank_flip",
                            "split": split_col,
                            "left": left,
                            "right": right,
                            "left_order": " > ".join(orders[left]),
                            "right_order": " > ".join(orders[right]),
                            "memory_method": "",
                            "rank": pd.NA,
                        }
                    )
                    rows.append(row)
    return rows
