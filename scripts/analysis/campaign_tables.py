"""Reusable campaign/experiment tables (no LLM).

YAML recipes choose ``group_by`` / ``metrics``; this module computes the table.
Category ids use official LoCoMo JSON names (1=multi-hop), not paper §4.1.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.metrics.locomo_qa import CATEGORY_NAMES


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
