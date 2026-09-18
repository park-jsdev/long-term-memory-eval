"""Reusable campaign/experiment tables (no LLM).

YAML recipes choose ``group_by`` / ``metrics``; this module computes the table.
Category ids use official LoCoMo JSON names (1=multi-hop), not paper §4.1.
"""

from __future__ import annotations

import json
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


def annotate_thinking(df: pd.DataFrame, pack: Path | None = None) -> pd.DataFrame:
    """Add ``thinking`` (on/off). Parquet first; else catalog ``run_meta`` / ``run.json``."""
    out = df.copy()
    if "thinking" in out.columns:
        mapped = out["thinking"].map(_thinking_token)
        if mapped.notna().any():
            out["thinking"] = mapped
            return out
    lookup = _thinking_lookup(pack, out.get("run_id")) if pack is not None else {}
    if lookup and "run_id" in out.columns:
        out["thinking"] = out["run_id"].map(lookup)
        return out
    for src in ("teacher_thinking", "reader_thinking"):
        if src in out.columns:
            mapped = out[src].map(_thinking_token)
            if mapped.notna().any():
                out["thinking"] = mapped
                return out
    return out


def sort_year_family_table(table: pd.DataFrame) -> pd.DataFrame:
    """Stable 2024 → 2025 → 2026, then family / method / thinking / source."""
    if table.empty:
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


def _thinking_token(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"on", "true", "1", "yes"}:
        return "on"
    if text in {"off", "false", "0", "no", "none", "null", ""}:
        if text in {"off", "false", "0", "no"}:
            return "off"
        return None
    return None


def _thinking_lookup(pack: Path, run_ids: Any) -> dict[str, str]:
    found: dict[str, str] = {}
    catalog = pack / "aggregate" / "by_run"
    if catalog.is_dir():
        for meta_path in catalog.glob("*/run_meta.json"):
            _absorb_thinking_file(found, meta_path)
    parent = pack.parent
    ids = []
    if run_ids is not None:
        ids = [str(v) for v in pd.Series(run_ids).dropna().unique().tolist()]
    for run_id in ids:
        if run_id in found:
            continue
        meta_path = parent / run_id / "run_meta.json"
        if meta_path.is_file():
            _absorb_thinking_file(found, meta_path)
    return found


def _absorb_thinking_file(found: dict[str, str], meta_path: Path) -> None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    run_json: dict[str, Any] = {}
    sibling = meta_path.parent / "run.json"
    if sibling.is_file():
        try:
            loaded = json.loads(sibling.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                run_json = loaded
        except (OSError, json.JSONDecodeError):
            pass
    label = _thinking_from_meta(meta if isinstance(meta, dict) else {}, run_json)
    if not label:
        return
    found[meta_path.parent.name] = label
    run_id = str(meta.get("run_id") or "")
    if run_id:
        found[run_id] = label


def _thinking_from_meta(meta: dict[str, Any], run_json: dict[str, Any]) -> str | None:
    teacher = meta.get("teacher_thinking")
    if teacher is None:
        teacher = run_json.get("teacher_thinking")
    reader_block = run_json.get("reader") if isinstance(run_json.get("reader"), dict) else {}
    reader = reader_block.get("thinking")
    if reader is None:
        reader = run_json.get("reader_thinking")
    if reader is None:
        reader = meta.get("reader_thinking")
    if teacher is not None:
        return _thinking_token(teacher)
    return _thinking_token(reader)


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


# Methods that do not retrieve. Search is 0, not "missing".
NO_RETRIEVE_METHODS = frozenset({"full_context"})


def _read_search_latency_map(path: Path) -> dict[str, float]:
    """Map ``question_id`` → search seconds from predictions JSONL or CSV."""
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    if path.suffix.lower() == ".csv":
        header = pd.read_csv(path, nrows=0)
        col = (
            "search_latency_s"
            if "search_latency_s" in header.columns
            else "search_latency_seconds"
            if "search_latency_seconds" in header.columns
            else None
        )
        if col is None or "question_id" not in header.columns:
            return out
        table = pd.read_csv(path, usecols=["question_id", col])
        for qid, raw in zip(table["question_id"], table[col]):
            val = pd.to_numeric(raw, errors="coerce")
            if pd.notna(val) and qid is not None and not (isinstance(qid, float) and pd.isna(qid)):
                out[str(qid)] = float(val)
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if qid is None:
                continue
            raw = row.get("search_latency_s")
            if raw is None:
                raw = row.get("search_latency_seconds")
            if raw is None:
                continue
            try:
                out[str(qid)] = float(raw)
            except (TypeError, ValueError):
                continue
    return out


def overlay_search_latency_from_run(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """Fill missing ``search_latency_seconds`` from one run's predictions file."""
    if df.empty:
        return df
    mapping = {}
    for rel in (
        "predictions.jsonl",
        "reader/predictions.jsonl",
        "memory/retrieve_ranks.jsonl",
        "predictions.csv",
    ):
        mapping = _read_search_latency_map(run_dir / rel)
        if mapping:
            break
    if not mapping or "question_id" not in df.columns:
        return df
    out = df.copy()
    if "search_latency_seconds" not in out.columns:
        out["search_latency_seconds"] = pd.NA
    current = pd.to_numeric(out["search_latency_seconds"], errors="coerce")
    mapped = out["question_id"].map(lambda q: mapping.get(str(q)))
    out["search_latency_seconds"] = current.fillna(pd.to_numeric(mapped, errors="coerce"))
    return out


def join_search_latency(df: pd.DataFrame, pack: Path) -> pd.DataFrame:
    """Backfill search seconds from hashed-run JSONL when catalog parquet omitted it."""
    if df.empty or "run_id" not in df.columns:
        return df
    out = df.copy()
    if "search_latency_seconds" not in out.columns:
        if "search_latency_s" in out.columns:
            out["search_latency_seconds"] = pd.to_numeric(
                out["search_latency_s"], errors="coerce"
            )
        else:
            out["search_latency_seconds"] = pd.NA
    current = pd.to_numeric(out["search_latency_seconds"], errors="coerce")
    experiments_root = pack.parent
    by_run = pack / "aggregate" / "by_run"
    filled = current.copy()
    for run_id, idx in out.groupby("run_id", dropna=False).groups.items():
        if current.loc[idx].notna().any():
            continue
        mapping: dict[str, float] = {}
        for path in (
            by_run / str(run_id) / "predictions.jsonl",
            by_run / str(run_id) / "predictions.csv",
            by_run / str(run_id) / "memory" / "retrieve_ranks.jsonl",
            experiments_root / str(run_id) / "predictions.jsonl",
            experiments_root / str(run_id) / "reader" / "predictions.jsonl",
            experiments_root / str(run_id) / "memory" / "retrieve_ranks.jsonl",
            experiments_root / str(run_id) / "predictions.csv",
        ):
            mapping = _read_search_latency_map(path)
            if mapping:
                break
        if not mapping or "question_id" not in out.columns:
            continue
        mapped = out.loc[idx, "question_id"].map(lambda q: mapping.get(str(q)))
        filled.loc[idx] = pd.to_numeric(mapped, errors="coerce")
    out["search_latency_seconds"] = filled
    return out


def annotate_mem0_latency(df: pd.DataFrame) -> pd.DataFrame:
    """Mem0 search/total seconds. Missing RAG search stays NA (not 0)."""
    out = df.copy()
    if "search_latency_seconds" not in out.columns and "search_latency_s" in out.columns:
        out["search_latency_seconds"] = pd.to_numeric(
            out["search_latency_s"], errors="coerce"
        )
    if "search_latency_seconds" not in out.columns:
        out["search_latency_seconds"] = pd.NA
    search = pd.to_numeric(out["search_latency_seconds"], errors="coerce")
    method_col = (
        "memory_method"
        if "memory_method" in out.columns
        else "memory_type"
        if "memory_type" in out.columns
        else None
    )
    if method_col is not None:
        no_retrieve = out[method_col].astype(str).isin(NO_RETRIEVE_METHODS)
        search = search.where(~(no_retrieve & search.isna()), 0.0)
    out["search_latency_seconds"] = search
    if "agent_latency_seconds" not in out.columns:
        return out
    generate = pd.to_numeric(out["agent_latency_seconds"], errors="coerce")
    # Search NA must not become a fake 0 on the search plot. Total still uses
    # generate + 0 when search was never recorded (old catalog parquet).
    out["total_latency_seconds"] = generate + search.fillna(0.0)
    return out


def mean_table(df: pd.DataFrame, group_by: list[str], metrics: list[str]) -> pd.DataFrame:
    cols = [c for c in group_by if c in df.columns]
    specs = []
    for name in metrics:
        src, how = _metric_reduce(name)
        if src in df.columns:
            specs.append((name, src, how))
    if not cols or not specs or df.empty:
        return pd.DataFrame()
    grouped = df.groupby(cols, dropna=False)
    out = grouped.size().reset_index(name="n")
    for name, src, how in specs:
        if how == "mean":
            series = grouped[src].mean()
        elif how == "p50":
            series = grouped[src].quantile(0.50)
        else:
            series = grouped[src].quantile(0.95)
        piece = series.reset_index(name=name)
        out = out.merge(piece, on=cols, how="left")
    if "question_category" in out.columns:
        out = _label_question_categories(out)
    return out


def _metric_reduce(name: str) -> tuple[str, str]:
    """Map ``foo_p50`` / ``foo_p95`` onto column ``foo``; otherwise mean ``name``."""
    if name.endswith("_p50"):
        return name[:-4], "p50"
    if name.endswith("_p95"):
        return name[:-4], "p95"
    return name, "mean"


def _label_question_categories(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["_cat_ord"] = pd.to_numeric(out["question_category"], errors="coerce").fillna(99)
    out = out.sort_values("_cat_ord", kind="mergesort").reset_index(drop=True)
    out["question_category"] = [
        category_axis_label(v) if v != 99 else v for v in out["_cat_ord"]
    ]
    return out.drop(columns=["_cat_ord"])
