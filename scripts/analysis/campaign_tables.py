"""Reusable campaign/experiment tables (no LLM).

YAML recipes choose ``group_by`` / ``metrics``; this module computes the table.
Category ids use official LoCoMo JSON names (1=multi-hop), not paper §4.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.metrics.locomo_qa import CATEGORY_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATION_LOOKUP: dict[str, str] | None = None
GENERATION_AXIS = ("2024", "2025", "2026")
_RESULT_SOURCE_ORDER = {"paper": 0, "local_clone": 1, "live": 2}


def annotate_model_family(df: pd.DataFrame, family_from: str = "reader") -> pd.DataFrame:
    """Add ``model_family`` (OpenAI / Anthropic / DeepSeek) for family plots."""
    out = df.copy()
    if family_from == "writer":
        src = out["writer_model"] if "writer_model" in out.columns else None
    else:
        src = None
        if "reader_provider" in out.columns:
            src = out["reader_provider"]
        elif "reader_family" in out.columns:
            src = out["reader_family"]
        elif "reader_model" in out.columns:
            src = out["reader_model"]
    if src is None:
        out["model_family"] = "unknown"
        return out
    out["model_family"] = src.map(_family_label)
    return out


def annotate_generation(df: pd.DataFrame, family_from: str = "reader") -> pd.DataFrame:
    """Add ``generation`` (2024 / 2025 / 2026) from catalog or parquet."""
    out = df.copy()
    if family_from == "writer" and "writer_model" in out.columns:
        out["generation"] = out["writer_model"].map(_generation_year)
        return out
    if "reader_generation" in out.columns:
        out["generation"] = out["reader_generation"].map(_year_token)
        return out
    if "reader_model" in out.columns:
        out["generation"] = out["reader_model"].map(_generation_year)
        return out
    out["generation"] = "unknown"
    return out


def sort_year_family_table(table: pd.DataFrame) -> pd.DataFrame:
    """Stable 2024 → 2025 → 2026, then family / method / source."""
    if table.empty or "generation" not in table.columns:
        return table
    out = table.copy()
    sort_cols: list[str] = []
    if "generation" in out.columns:
        order = {year: i for i, year in enumerate(GENERATION_AXIS)}
        out["_gen_ord"] = out["generation"].map(lambda v: order.get(str(v), 9))
        sort_cols.append("_gen_ord")
    if "model_family" in out.columns:
        sort_cols.append("model_family")
    if "memory_method" in out.columns:
        sort_cols.append("memory_method")
    if "thinking" in out.columns:
        think_order = {"off": 0, "on": 1}
        out["_think_ord"] = out["thinking"].map(
            lambda v: think_order.get(str(v).strip().lower(), 9)
        )
        sort_cols.append("_think_ord")
    if "result_source" in out.columns:
        out["_src_ord"] = out["result_source"].map(
            lambda v: _RESULT_SOURCE_ORDER.get(str(v), 9)
        )
        sort_cols.append("_src_ord")
    if not sort_cols:
        return out
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out.drop(columns=[c for c in ("_gen_ord", "_src_ord", "_think_ord") if c in out.columns])


def _year_token(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return text or "unknown"


def _generation_year(value: Any) -> str:
    key = str(value or "").strip().lower()
    lookup = _generation_lookup()
    if key in lookup:
        return lookup[key]
    return _year_token(value)


def _generation_lookup() -> dict[str, str]:
    global _GENERATION_LOOKUP
    if _GENERATION_LOOKUP is not None:
        return _GENERATION_LOOKUP
    path = _REPO_ROOT / "configs" / "models" / "generation_catalog.yaml"
    mapping: dict[str, str] = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for model_id, block in (raw.get("models") or {}).items():
            year = _year_token((block or {}).get("generation"))
            mapping[str(model_id).lower()] = year
            api = str((block or {}).get("api_model_id") or "").strip().lower()
            if api and api != "to_confirm":
                mapping[api] = year
    _GENERATION_LOOKUP = mapping
    return mapping


def _family_label(value: Any) -> str:
    s = str(value or "").lower()
    if "anthropic" in s or s.startswith("claude"):
        return "Anthropic"
    if "deepseek" in s:
        return "DeepSeek"
    if "openai" in s or s.startswith("gpt"):
        return "OpenAI"
    return str(value or "unknown")


def category_axis_label(value: Any) -> str:
    """Official LoCoMo JSON ids: 1=multi-hop, 4=single-hop (not paper §4.1 order)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return str(value)
    name = CATEGORY_NAMES.get(n)
    if not name:
        return str(n)
    return f"{n} {name.replace('_', '-')}"


def mean_table(df: pd.DataFrame, group_by: list[str], metrics: list[str]) -> pd.DataFrame:
    cols = [c for c in group_by if c in df.columns]
    mets = [m for m in metrics if m in df.columns]
    if not cols or not mets or df.empty:
        return pd.DataFrame()
    grouped = df.groupby(cols, dropna=False)[mets].mean().reset_index()
    counts = df.groupby(cols, dropna=False).size().reset_index(name="n")
    out = grouped.merge(counts, on=cols, how="left")
    if "question_category" in out.columns:
        out = _label_question_categories(out)
    return out


def _label_question_categories(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["_cat_ord"] = pd.to_numeric(out["question_category"], errors="coerce").fillna(99)
    out = out.sort_values("_cat_ord", kind="mergesort").reset_index(drop=True)
    out["question_category"] = [
        category_axis_label(v) if v != 99 else v for v in out["_cat_ord"]
    ]
    return out.drop(columns=["_cat_ord"])
