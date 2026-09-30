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
READER_STACK_AXIS = (
    "2024 mini model-only",
    "2024 mini + Codex",
    "2025 GPT-5 model-only",
    "2026 Terra model-only",
)
COMPARE_SOURCE_CODEX = "gpt-4o-mini + Codex"
COMPARE_SOURCE_LIVE = "live model"
COMPARE_SOURCE_AXIS = (
    "paper",
    "local clone",
    COMPARE_SOURCE_LIVE,
    COMPARE_SOURCE_CODEX,
    "2025 GPT-5 model-only",
    "2026 Terra model-only",
)
LIVE_SOURCE_AXIS = ("paper", COMPARE_SOURCE_LIVE, COMPARE_SOURCE_CODEX)
PAPER_METHOD_AXIS = (
    "full_context",
    "rag",
    "openai_memory",
    "mem0",
    "mem0g",
    "session_summaries",
)
MEMORY_LANE_AXIS = (
    "full_context",
    "teacher_session_summaries",
    "teacher_graph",
)
SYSTEM_HARNESS_AXIS = ("chat_completions", "codex")
MINI_SIDE_AXIS = ("4o-mini", "4o-mini + Codex")
GAP_STACK_AXIS = ("2024 model", "2024 model + harness", "2026 model")
_RESULT_SOURCE_ORDER = {
    "paper": 0,
    "local_clone": 1,
    "local clone": 1,
    "live": 2,
    COMPARE_SOURCE_LIVE: 2,
    COMPARE_SOURCE_CODEX: 3,
}


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
    if "reader_stack" in out.columns:
        order = {label: i for i, label in enumerate(READER_STACK_AXIS)}
        out["_stack_ord"] = out["reader_stack"].map(lambda v: order.get(str(v), 9))
        sort_cols.append("_stack_ord")
    if "paper_method" in out.columns:
        order = {label: i for i, label in enumerate(PAPER_METHOD_AXIS)}
        out["_paper_ord"] = out["paper_method"].map(lambda v: order.get(str(v), 9))
        sort_cols.append("_paper_ord")
    if "mini_side" in out.columns:
        order = {label: i for i, label in enumerate(MINI_SIDE_AXIS)}
        out["_mini_ord"] = out["mini_side"].map(lambda v: order.get(str(v), 9))
        sort_cols.append("_mini_ord")
    if "gap_stack" in out.columns:
        order = {label: i for i, label in enumerate(GAP_STACK_AXIS)}
        out["_gap_ord"] = out["gap_stack"].map(lambda v: order.get(str(v), 9))
        sort_cols.append("_gap_ord")
    if "memory_lane" in out.columns:
        order = {label: i for i, label in enumerate(MEMORY_LANE_AXIS)}
        out["_lane_ord"] = out["memory_lane"].map(lambda v: order.get(str(v), 9))
        sort_cols.append("_lane_ord")
    if "system_harness" in out.columns:
        order = {label: i for i, label in enumerate(SYSTEM_HARNESS_AXIS)}
        out["_harness_ord"] = out["system_harness"].map(lambda v: order.get(str(v), 9))
        sort_cols.append("_harness_ord")
    if "compare_source" in out.columns:
        order = {label: i for i, label in enumerate(COMPARE_SOURCE_AXIS)}
        out["_cmp_ord"] = out["compare_source"].map(lambda v: order.get(str(v), 9))
        sort_cols.append("_cmp_ord")
    if "live_source" in out.columns:
        order = {label: i for i, label in enumerate(LIVE_SOURCE_AXIS)}
        out["_live_ord"] = out["live_source"].map(lambda v: order.get(str(v), 9))
        sort_cols.append("_live_ord")
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
    return out.drop(
        columns=[
            c
            for c in (
                "_gen_ord",
                "_src_ord",
                "_think_ord",
                "_stack_ord",
                "_paper_ord",
                "_cmp_ord",
                "_live_ord",
                "_lane_ord",
                "_harness_ord",
                "_mini_ord",
                "_gap_ord",
            )
            if c in out.columns
        ]
    )


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


# Pack metrics.json includes category 5. Overall campaign tables follow Mem0.
RUN_SCORE_METRICS = ("judge_score", "locomo_f1", "token_f1", "exact_match")


def attach_run_judge_score(
    runs: pd.DataFrame, examples: pd.DataFrame
) -> pd.DataFrame:
    """Copy cat-5-excluded mean J / F1 / EM from examples onto each runs row.

    ``runs.parquet`` stores pack LoCoMo F1 (all categories) and has no J.
    Overall campaign tables follow Mem0: drop adversarial category 5 so
    ``source: runs`` F1 matches ``source: examples`` overalls. Keep pack
    values when a run has no example rows. Do not reuse this drop for
    tool-audit counts.
    """
    if runs.empty or examples.empty:
        return runs
    if "run_id" not in runs.columns or "run_id" not in examples.columns:
        return runs
    work = examples
    if "question_category" in work.columns:
        category = pd.to_numeric(work["question_category"], errors="coerce")
        work = work.loc[~category.isin({5})]
    out = runs.copy()
    for col in RUN_SCORE_METRICS:
        if col not in work.columns:
            continue
        means = pd.to_numeric(work[col], errors="coerce").groupby(work["run_id"]).mean()
        if means.empty or int(means.notna().sum()) == 0:
            continue
        overlay = out["run_id"].map(means)
        if col in out.columns:
            out[col] = overlay.where(overlay.notna(), out[col])
        else:
            out[col] = overlay
    return out


AGENT_AUDIT_METRICS = (
    "n_web_search",
    "n_mcp",
    "used_non_workspace_tools",
    "n_retrieval_calls",
    "memory_recall",
    "n_write_events",
    "hop_to_evidence",
    "notes_retrieved",
    "notes_bytes",
)
_IDENTITY_COLS = (
    "agent",
    "agent_persist",
    "agent_sessions",
    "agent_tools",
    "comparison_status",
    "writer_provider",
)


def attach_run_agent_audit(
    runs: pd.DataFrame, examples: pd.DataFrame
) -> pd.DataFrame:
    """Copy per-run mean tool-audit columns from examples onto runs.

    Counts include LoCoMo category 5. A web_search on an adversarial
    question is still a policy breach. Do not reuse the J cat-5 drop.
    """
    if runs.empty or examples.empty:
        return runs
    if "run_id" not in runs.columns or "run_id" not in examples.columns:
        return runs
    cols = [c for c in AGENT_AUDIT_METRICS if c in examples.columns]
    if not cols:
        return runs
    work = examples[["run_id", *cols]].copy()
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    means = work.groupby("run_id", dropna=False)[cols].mean().reset_index()
    out = runs.copy()
    keep = ["run_id"]
    for col in cols:
        if col in out.columns and pd.to_numeric(out[col], errors="coerce").notna().any():
            continue
        if col in out.columns:
            out = out.drop(columns=[col])
        keep.append(col)
    if keep == ["run_id"]:
        return out
    return out.merge(means[keep], on="run_id", how="left")


FAILURE_HARNESS = "harness_execution_failure"
FAILURE_MODE_LABELS = {
    "none": "correct (with evidence)",
    "parametric_success": "correct (no gold evidence)",
    "reasoning_failure": "wrong after retrieve",
    "retrieval_failure": "gold ids not retrieved",
    "harness_execution_failure": "no workspace read",
    "unjudged": "unjudged (no J)",
}
FAILURE_MODE_ORDER = (
    "none",
    "parametric_success",
    "reasoning_failure",
    "retrieval_failure",
    "harness_execution_failure",
    "unjudged",
)
RECALL_BIN_LABELS = {
    "none": "no gold ids retrieved",
    "partial": "partial gold-id chain",
    "full": "all gold ids retrieved",
    "no_gold_ids": "no gold evidence ids",
}
RECALL_BIN_ORDER = ("none", "partial", "full", "no_gold_ids")
JUDGE_F1_LABELS = {
    "both_correct": "J CORRECT and F1>0",
    "j_only": "J CORRECT and F1=0 (paraphrase / verbose)",
    "f1_only": "J WRONG and F1>0",
    "both_wrong": "J WRONG and F1=0",
    "unjudged": "unjudged (no J)",
}
JUDGE_F1_ORDER = (
    "both_correct",
    "j_only",
    "f1_only",
    "both_wrong",
    "unjudged",
)
RETRIEVAL_CALLS_ORDER = ("0", "1", "2-3", "4-8", "9-20", "21+", "unknown")
HOP_BIN_ORDER = ("1", "2", "3-4", "5-8", "9+", "never", "unknown")
QIDX_BIN_ORDER = ("0-4", "5-9", "10-19", "20-39", "40+", "unknown")
NOTES_BYTES_ORDER = ("empty", "1-256", "257-1k", "1k-4k", "4k+", "unknown")


def repair_run_harness_status(
    runs: pd.DataFrame, examples: pd.DataFrame
) -> pd.DataFrame:
    """Cell status is all-failed, not any-failed. Attach the per-question rate.

    Finished packs may still store ``comparison_status=harness_failed`` from
    the old ``any()`` rule. If any question had a successful workspace read,
    relabel the cell ``incomparable`` (native Codex) and keep the rate.
    """
    if runs.empty or "run_id" not in runs.columns:
        return runs
    out = runs.copy()
    if examples.empty or "run_id" not in examples.columns:
        return out
    failed = _harness_failed_mask(examples)
    stats = (
        pd.DataFrame(
            {
                "run_id": examples["run_id"],
                "n_harness_failed": failed.astype(int),
            }
        )
        .groupby("run_id", dropna=False)
        .agg(n_harness_failed=("n_harness_failed", "sum"), n=("n_harness_failed", "size"))
        .reset_index()
    )
    stats["harness_failed_rate"] = stats["n_harness_failed"] / stats["n"]
    keep = ["run_id", "n_harness_failed", "harness_failed_rate"]
    for col in ("n_harness_failed", "harness_failed_rate"):
        if col in out.columns:
            out = out.drop(columns=[col])
    out = out.merge(stats[keep], on="run_id", how="left")
    if "comparison_status" not in out.columns:
        return out
    status = out["comparison_status"].astype(str).str.strip().str.lower()
    rate = pd.to_numeric(out["harness_failed_rate"], errors="coerce")
    partial = status.eq("harness_failed") & rate.notna() & (rate < 1.0)
    out.loc[partial, "comparison_status"] = "incomparable"
    return out


def copy_run_identity_to_examples(
    examples: pd.DataFrame, runs: pd.DataFrame
) -> pd.DataFrame:
    """Copy persist / adapter from runs onto question rows.

    ``examples.parquet`` often omits ``agent_persist`` even when the cell
    row has it. Workspace recipes group on persist.
    """
    cols = [
        c
        for c in (
            "agent",
            "agent_persist",
            "agent_sessions",
            "agent_tools",
            "comparison_status",
        )
        if c in runs.columns
    ]
    if examples.empty or runs.empty or not cols or "run_id" not in examples.columns:
        return examples
    ident = runs[["run_id", *cols]].drop_duplicates("run_id")
    out = examples.merge(ident, on="run_id", how="left", suffixes=("", "_run"))
    for col in cols:
        run_col = f"{col}_run"
        if run_col not in out.columns:
            continue
        if col not in out.columns:
            out[col] = out[run_col]
        else:
            out[col] = out[col].where(_present(out[col]), out[run_col])
        out = out.drop(columns=[run_col])
    return out


def annotate_workspace_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns for harness J vs F1 and retrieval-chain tables.

    LoCoMo F1 and Mem0 J use the same predicted string. F1 is token overlap
    against short gold; J is a generous paraphrase judge. Packed JSON
    wrappers are already stripped before scoring. ``recall_bin`` is gold
    ``dia_id`` coverage across retrieve events (partial chains included).
    ``failure_mode_judge`` reapplies the retrieve/reason split with J
    instead of ``locomo_f1 > 0``.
    """
    if df.empty:
        return df
    out = df.copy()
    if "generated_answer" in out.columns:
        out["answer_n_words"] = [
            len(str(v).split())
            if v is not None and not (isinstance(v, float) and pd.isna(v))
            else 0
            for v in out["generated_answer"]
        ]
    if "reference_answer" in out.columns:
        out["gold_n_words"] = [
            len(str(v).split())
            if v is not None and not (isinstance(v, float) and pd.isna(v))
            else 0
            for v in out["reference_answer"]
        ]
    if "memory_recall" in out.columns:
        out["recall_bin"] = [
            _recall_bin(v)
            for v in pd.to_numeric(out["memory_recall"], errors="coerce")
        ]
    if "n_retrieval_calls" in out.columns:
        out["retrieval_calls_bin"] = [
            _retrieval_calls_bin(v)
            for v in pd.to_numeric(out["n_retrieval_calls"], errors="coerce")
        ]
    if "judge_score" in out.columns or "locomo_f1" in out.columns:
        judge = (
            pd.to_numeric(out["judge_score"], errors="coerce")
            if "judge_score" in out.columns
            else pd.Series([float("nan")] * len(out), index=out.index)
        )
        f1 = (
            pd.to_numeric(out["locomo_f1"], errors="coerce")
            if "locomo_f1" in out.columns
            else pd.Series([float("nan")] * len(out), index=out.index)
        )
        out["judge_vs_f1"] = [
            _judge_vs_f1(j, s) for j, s in zip(judge.tolist(), f1.tolist())
        ]
        evidence = (
            out["evidence_retrieved"]
            if "evidence_retrieved" in out.columns
            else pd.Series([None] * len(out), index=out.index)
        )
        harness = (
            out["failure_mode"].astype(str).eq(FAILURE_HARNESS)
            if "failure_mode" in out.columns
            else pd.Series(False, index=out.index)
        )
        if "harness_failed" in out.columns:
            harness = harness | out["harness_failed"].fillna(False).astype(bool)
        out["failure_mode_judge"] = [
            _failure_mode_judge(failed, ev, j)
            for failed, ev, j in zip(
                harness.tolist(), evidence.tolist(), judge.tolist()
            )
        ]
    qids = (
        out["question_id"]
        if "question_id" in out.columns
        else pd.Series([None] * len(out), index=out.index)
    )
    out["qidx"] = [_qidx_from_id(v) for v in qids]
    out["qidx_bin"] = [_qidx_bin(v) for v in out["qidx"]]
    if "hop_to_evidence" in out.columns:
        out["hop_bin"] = [
            _hop_bin(v) for v in pd.to_numeric(out["hop_to_evidence"], errors="coerce")
        ]
    if "notes_bytes" in out.columns:
        out["notes_bytes_bin"] = [
            _notes_bytes_bin(v)
            for v in pd.to_numeric(out["notes_bytes"], errors="coerce")
        ]
    return out


def _recall_bin(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "no_gold_ids"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "no_gold_ids"
    if score <= 0:
        return "none"
    if score < 1:
        return "partial"
    return "full"


def _hop_bin(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "never"
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if n <= 0:
        return "never"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 8:
        return "5-8"
    return "9+"


def _qidx_from_id(question_id: Any) -> int | None:
    qid = str(question_id or "")
    marker = "-q-"
    if marker in qid:
        tail = qid.rsplit(marker, 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return None
    return None


def _qidx_bin(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if n < 0:
        return "unknown"
    if n <= 4:
        return "0-4"
    if n <= 9:
        return "5-9"
    if n <= 19:
        return "10-19"
    if n <= 39:
        return "20-39"
    return "40+"


def _notes_bytes_bin(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if n <= 0:
        return "empty"
    if n <= 256:
        return "1-256"
    if n <= 1024:
        return "257-1k"
    if n <= 4096:
        return "1k-4k"
    return "4k+"


def _retrieval_calls_bin(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 8:
        return "4-8"
    if n <= 20:
        return "9-20"
    return "21+"


def _judge_vs_f1(judge: Any, f1: Any) -> str:
    if judge is None or (isinstance(judge, float) and pd.isna(judge)):
        return "unjudged"
    j_ok = float(judge) >= 0.5
    f1_ok = (
        f1 is not None
        and not (isinstance(f1, float) and pd.isna(f1))
        and float(f1) > 0
    )
    if j_ok and f1_ok:
        return "both_correct"
    if j_ok:
        return "j_only"
    if f1_ok:
        return "f1_only"
    return "both_wrong"


def _failure_mode_judge(harness_failed: Any, evidence: Any, judge: Any) -> str:
    if bool(harness_failed):
        return FAILURE_HARNESS
    if judge is None or (isinstance(judge, float) and pd.isna(judge)):
        return "unjudged"
    if evidence is None or (isinstance(evidence, float) and pd.isna(evidence)):
        ev = None
    else:
        ev = bool(evidence)
    correct = float(judge) >= 0.5
    if correct:
        return "parametric_success" if ev is False else "none"
    return "retrieval_failure" if ev is False else "reasoning_failure"


def _harness_failed_mask(examples: pd.DataFrame) -> pd.Series:
    if "failure_mode" in examples.columns:
        return examples["failure_mode"].astype(str).eq(FAILURE_HARNESS)
    if "harness_failed" in examples.columns:
        return examples["harness_failed"].fillna(False).astype(bool)
    return pd.Series(False, index=examples.index)


def attach_agent_identity(
    df: pd.DataFrame, pack: Path | None = None
) -> pd.DataFrame:
    """Fill agent / persist / comparison_status / writer_provider from the pack.

    Older ``runs.parquet`` rows omit these even when the manifest and
    ``run_meta.json`` recorded them. Backfill so audit recipes can group.
    """
    if df.empty or "run_id" not in df.columns:
        return df
    out = df.copy()
    lookup = _agent_identity_lookup(pack) if pack is not None else {}
    if lookup:
        mapped = out["run_id"].map(lookup)
        for col in _IDENTITY_COLS:
            incoming = mapped.map(
                lambda rec, c=col: (rec or {}).get(c) if isinstance(rec, dict) else None
            )
            if col not in out.columns:
                out[col] = incoming
            else:
                out[col] = out[col].where(_present(out[col]), incoming)
    if "agent" not in out.columns or not _present(out["agent"]).any():
        if "agent_harness" in out.columns:
            out["agent"] = out["agent_harness"]
    if "agent" in out.columns:
        out["agent"] = out["agent"].map(_agent_token)
    return out


def annotate_reader_stack(df: pd.DataFrame) -> pd.DataFrame:
    """Label model-only vs Codex harness for the 2024–2026 J comparison.

    Thinking-on year cells, DeepSeek, RAG, and persist-on workspace are
    left unlabeled so YAML ``where: reader_stack`` drops them. Persist-off
    Codex is the retrieval harness, not a memory-method cell.
    """
    if df.empty:
        return df
    out = df.copy()
    n = len(out)
    gens = out["generation"] if "generation" in out.columns else None
    methods = out["memory_method"] if "memory_method" in out.columns else None
    models = out["reader_model"] if "reader_model" in out.columns else None
    families = out["model_family"] if "model_family" in out.columns else None
    persist = out["agent_persist"] if "agent_persist" in out.columns else None
    thinking = out["thinking"] if "thinking" in out.columns else None
    labels: list[str | None] = []
    for i in range(n):
        if thinking is not None and _thinking_token(thinking.iloc[i]) == "on":
            labels.append(None)
            continue
        family = None if families is None else families.iloc[i]
        if (
            family is not None
            and not (isinstance(family, float) and pd.isna(family))
            and str(family) not in ("", "nan", "None", "unknown")
            and str(family) != "OpenAI"
        ):
            labels.append(None)
            continue
        method = "" if methods is None else str(methods.iloc[i] or "")
        gen = "" if gens is None else str(gens.iloc[i] or "")
        model = "" if models is None else str(models.iloc[i] or "").lower()
        if method == "workspace_files":
            flag = None if persist is None else persist.iloc[i]
            if flag is True or str(flag).strip().lower() == "true":
                labels.append(None)
                continue
            labels.append("2024 mini + Codex" if gen in {"2024", ""} or "mini" in model else f"{gen} + Codex")
            continue
        if method != "full_context":
            labels.append(None)
            continue
        if gen == "2026" or "5.6" in model or "terra" in model:
            labels.append("2026 Terra model-only")
        elif gen == "2025" or model.startswith("gpt-5"):
            labels.append("2025 GPT-5 model-only")
        elif gen == "2024" or "mini" in model:
            labels.append("2024 mini model-only")
        else:
            labels.append(None)
    out["reader_stack"] = labels
    return out


def annotate_paper_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Align live cells to Table 2 method ids for paper / live / Codex plots.

    ``compare_source`` keeps paper, local clone, live Chat Completions, Codex,
    and 2025/2026 model-only as separate bars. ``live_source`` folds local
    clone into live model for the methods plot (clone pins dropped when a
    pack live-model row exists for that method). Persist-on workspace is
    not the retrieval harness. Graph / summaries map only for Codex writers.
    """
    if df.empty:
        return df
    out = df.copy()
    n = len(out)
    methods = out["memory_method"] if "memory_method" in out.columns else None
    persist = out["agent_persist"] if "agent_persist" in out.columns else None
    harness = out["writer_harness"] if "writer_harness" in out.columns else None
    providers = out["writer_provider"] if "writer_provider" in out.columns else None
    names = out["experiment_name"] if "experiment_name" in out.columns else None
    sources = out["result_source"] if "result_source" in out.columns else None
    gens = out["generation"] if "generation" in out.columns else None
    models = out["reader_model"] if "reader_model" in out.columns else None
    families = out["model_family"] if "model_family" in out.columns else None
    thinking = out["thinking"] if "thinking" in out.columns else None
    paper_methods: list[str | None] = []
    compare_sources: list[str | None] = []
    for i in range(n):
        method = "" if methods is None else str(methods.iloc[i] or "")
        source = "" if sources is None else str(sources.iloc[i] or "")
        pin = _pin_compare_source(source)
        if pin is not None:
            paper_methods.append(method or None)
            compare_sources.append(pin)
            continue
        name = "" if names is None else str(names.iloc[i] or "")
        mapped, live = _live_paper_compare(
            method=method,
            persist=None if persist is None else persist.iloc[i],
            is_codex_writer=_is_codex_writer(
                None if harness is None else harness.iloc[i]
            )
            or _is_codex_writer(
                None if providers is None else providers.iloc[i]
            )
            or "codex" in name.lower(),
            generation="" if gens is None else gens.iloc[i],
            reader_model="" if models is None else models.iloc[i],
            model_family="" if families is None else families.iloc[i],
            thinking=None if thinking is None else thinking.iloc[i],
        )
        paper_methods.append(mapped)
        compare_sources.append(live)
    out["paper_method"] = paper_methods
    out["compare_source"] = compare_sources
    out["live_source"] = [
        COMPARE_SOURCE_LIVE if src == "local clone" else src for src in compare_sources
    ]
    return out


def _pin_compare_source(source: str) -> str | None:
    key = str(source or "").strip()
    if key == "paper":
        return "paper"
    if key in {"local_clone", "local clone"}:
        return "local clone"
    return None


def _live_paper_compare(
    *,
    method: str,
    persist: Any,
    is_codex_writer: bool,
    generation: Any,
    reader_model: Any,
    model_family: Any,
    thinking: Any,
) -> tuple[str | None, str | None]:
    """Return (paper_method, compare_source) for a live pack cell."""
    if method == "workspace_files":
        if _truthy_flag(persist):
            return method, None
        return "full_context", COMPARE_SOURCE_CODEX
    if method == "agent_codex_mem0_facts":
        return "mem0", COMPARE_SOURCE_CODEX
    if method == "full_context":
        return "full_context", _full_context_live_source(
            generation=generation,
            reader_model=reader_model,
            model_family=model_family,
            thinking=thinking,
        )
    if not is_codex_writer:
        return method or None, None
    if method == "teacher_graph":
        return "mem0g", COMPARE_SOURCE_CODEX
    if method == "teacher_session_summaries":
        return "session_summaries", COMPARE_SOURCE_CODEX
    return method or None, None


def _full_context_live_source(
    *,
    generation: Any,
    reader_model: Any,
    model_family: Any,
    thinking: Any,
) -> str | None:
    if _thinking_token(thinking) == "on":
        return None
    family = "" if model_family is None else str(model_family)
    if family not in ("", "nan", "None", "unknown", "OpenAI"):
        return None
    gen = "" if generation is None else str(generation or "")
    model = "" if reader_model is None else str(reader_model or "").lower()
    if gen == "2026" or "5.6" in model or "terra" in model:
        return "2026 Terra model-only"
    if gen == "2025" or model.startswith("gpt-5"):
        return "2025 GPT-5 model-only"
    return COMPARE_SOURCE_LIVE


def _truthy_flag(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes"}


def _is_codex_writer(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() == "codex"


def annotate_memory_lane(df: pd.DataFrame) -> pd.DataFrame:
    """Align stuffed FC, persist-off workspace, and sandwich writers.

    Persist-off ``workspace_files`` is the Codex analog of stuffed
    ``full_context``. Persist-on workspace and Codex facts are not lanes
    (no Chat Completions twin). ``teacher_graph`` stays ``teacher_graph``
    (not paper Mem0g).
    """
    out = df.copy()
    if out.empty:
        out["memory_lane"] = []
        return out
    n = len(out)
    methods = out["memory_method"] if "memory_method" in out.columns else None
    persist = out["agent_persist"] if "agent_persist" in out.columns else None
    lanes: list[str | None] = []
    for i in range(n):
        method = "" if methods is None else str(methods.iloc[i] or "")
        flag = None if persist is None else persist.iloc[i]
        if method == "workspace_files":
            lanes.append(None if _truthy_flag(flag) else "full_context")
        elif method in MEMORY_LANE_AXIS:
            lanes.append(method)
        else:
            lanes.append(None)
    out["memory_lane"] = lanes
    return out


def annotate_system_harness(df: pd.DataFrame) -> pd.DataFrame:
    """Codex vs Chat Completions on the reader path or the writer path.

    Sandwich cells freeze a Chat Completions reader; ``agent`` is none and
    the writer provider is the harness. Workspace cells use Codex as the
    reader. Stuffed ``full_context`` is model-only Chat Completions.
    """
    out = annotate_writer_harness(df)
    n = len(out)
    methods = out["memory_method"] if "memory_method" in out.columns else None
    agents = out["agent"] if "agent" in out.columns else None
    harness = out["writer_harness"] if "writer_harness" in out.columns else None
    providers = out["writer_provider"] if "writer_provider" in out.columns else None
    writers = out["writer_model"] if "writer_model" in out.columns else None
    labels: list[str | None] = []
    for i in range(n):
        method = "" if methods is None else str(methods.iloc[i] or "")
        agent = "none" if agents is None else _agent_token(agents.iloc[i])
        writer = None if writers is None else writers.iloc[i]
        has_writer = (
            writer is not None
            and not (isinstance(writer, float) and pd.isna(writer))
            and str(writer) not in ("", "None", "nan", "null")
        )
        if agent == "codex" or _is_codex_writer(
            None if harness is None else harness.iloc[i]
        ) or _is_codex_provider(None if providers is None else providers.iloc[i]):
            labels.append("codex")
        elif has_writer:
            labels.append("chat_completions")
        elif method == "full_context":
            labels.append("chat_completions")
        else:
            labels.append(None)
    out["system_harness"] = labels
    return out


def annotate_mini_side(df: pd.DataFrame) -> pd.DataFrame:
    """2024 GPT-4o-mini Chat Completions vs the same model inside Codex.

    2026 Chat Completions stays unlabeled so a year chart does not call
    Terra "4o-mini".
    """
    out = df.copy()
    if out.empty:
        out["mini_side"] = []
        return out
    if "system_harness" not in out.columns:
        out = annotate_system_harness(out)
    n = len(out)
    harness = out["system_harness"]
    gens = out["generation"] if "generation" in out.columns else None
    labels: list[str | None] = []
    for i in range(n):
        gen = "" if gens is None else str(gens.iloc[i] or "")
        if gen not in {"", "2024", "nan", "None"}:
            labels.append(None)
            continue
        side = str(harness.iloc[i] or "")
        if side == "chat_completions":
            labels.append("4o-mini")
        elif side == "codex":
            labels.append("4o-mini + Codex")
        else:
            labels.append(None)
    out["mini_side"] = labels
    return out


def annotate_gap_stack(df: pd.DataFrame) -> pd.DataFrame:
    """Full-context stacks for the year vs harness category chart.

    ``2024 model`` is stuffed Chat Completions. ``2024 model + harness`` is
    persist-off Codex workspace. ``2026 model`` is Terra thinking-off
    full_context. Sandwich writers stay unlabeled.
    """
    out = df.copy()
    if out.empty:
        out["gap_stack"] = []
        return out
    if "compare_source" not in out.columns:
        out = annotate_paper_compare(out)
    n = len(out)
    sources = out["compare_source"]
    lanes = out["memory_lane"] if "memory_lane" in out.columns else None
    methods = out["memory_method"] if "memory_method" in out.columns else None
    labels: list[str | None] = []
    for i in range(n):
        source = "" if sources is None else str(sources.iloc[i] or "")
        lane = "" if lanes is None else str(lanes.iloc[i] or "")
        method = "" if methods is None else str(methods.iloc[i] or "")
        full = lane == "full_context" or method == "full_context"
        if source == "2026 Terra model-only":
            labels.append("2026 model")
        elif source == COMPARE_SOURCE_LIVE and full:
            labels.append("2024 model")
        elif source == COMPARE_SOURCE_CODEX and full:
            labels.append("2024 model + harness")
        else:
            labels.append(None)
    out["gap_stack"] = labels
    return out


def annotate_writer_harness(df: pd.DataFrame) -> pd.DataFrame:
    """``codex`` vs ``chat_completions`` for sandwich writer-family plots.

    Codex sandwich cells use a frozen Chat Completions reader, so ``agent``
    is none. The writer provider (or experiment name) is the harness.
    """
    out = df.copy()
    if "writer_harness" in out.columns and _present(out["writer_harness"]).any():
        return out
    n = len(out)
    providers = out["writer_provider"] if "writer_provider" in out.columns else None
    names = out["experiment_name"] if "experiment_name" in out.columns else None
    writers = out["writer_model"] if "writer_model" in out.columns else None
    labels: list[str | None] = []
    for i in range(n):
        provider = None if providers is None else providers.iloc[i]
        writer = None if writers is None else writers.iloc[i]
        name = "" if names is None else str(names.iloc[i] or "")
        has_writer = writer is not None and not (isinstance(writer, float) and pd.isna(writer)) and str(writer) != ""
        if _is_codex_provider(provider):
            labels.append("codex")
        elif has_writer and "codex" in name.lower():
            labels.append("codex")
        elif has_writer:
            labels.append("chat_completions")
        else:
            labels.append(None)
    out["writer_harness"] = labels
    return out


def _present(series: pd.Series) -> pd.Series:
    return series.notna() & ~series.astype(str).str.lower().isin(("", "none", "nan", "null"))


def _agent_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "none"
    text = str(value).strip().lower()
    if text in ("", "none", "null", "nan"):
        return "none"
    return text


def _is_codex_provider(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() == "codex"


def _agent_identity_lookup(pack: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    manifest = pack / "manifest" / "runs.jsonl"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            run_id = row.get("run_id")
            if not run_id:
                continue
            writer = row.get("writer") if isinstance(row.get("writer"), dict) else {}
            contract = row.get("comparison_contract") if isinstance(row.get("comparison_contract"), dict) else {}
            out[str(run_id)] = {
                "agent": row.get("agent"),
                "agent_persist": row.get("agent_persist"),
                "agent_sessions": row.get("agent_sessions"),
                "agent_tools": row.get("agent_tools"),
                "writer_provider": writer.get("provider"),
                "comparison_status": contract.get("status"),
            }
    by_run = pack / "aggregate" / "by_run"
    if by_run.is_dir():
        for meta_path in by_run.glob("*/run_meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            run_id = str(meta.get("run_id") or meta_path.parent.name)
            rec = dict(out.get(run_id) or {})
            if meta.get("agent") is not None:
                rec["agent"] = meta.get("agent")
            if "agent_persist" in meta:
                rec["agent_persist"] = meta.get("agent_persist")
            if meta.get("agent_sessions") is not None:
                rec["agent_sessions"] = meta.get("agent_sessions")
            if meta.get("agent_tools") is not None:
                rec["agent_tools"] = meta.get("agent_tools")
            contract = meta.get("comparison_contract") if isinstance(meta.get("comparison_contract"), dict) else {}
            if contract.get("status"):
                rec["comparison_status"] = contract.get("status")
            out[run_id] = rec
    return out


def mean_table(df: pd.DataFrame, group_by: list[str], metrics: list[str]) -> pd.DataFrame:
    work = df
    if (not work.empty) and any(col not in work.columns for col in group_by):
        work = annotate_writer_harness(work)
        work = annotate_memory_lane(work)
        work = annotate_system_harness(work)
        work = annotate_paper_compare(work)
        work = annotate_mini_side(work)
        work = annotate_gap_stack(work)
        if (
            "paper_method" in group_by
            and "paper_method" not in work.columns
            and "memory_method" in work.columns
        ):
            work = work.copy()
            work["paper_method"] = work["memory_method"]
    cols = [c for c in group_by if c in work.columns]
    specs = []
    want_n = False
    for name in metrics:
        if name == "n":
            want_n = True
            continue
        src, how = _metric_reduce(name)
        if src in work.columns:
            specs.append((name, src, how))
    if not cols or work.empty:
        return pd.DataFrame()
    if not specs and not want_n:
        return pd.DataFrame()
    grouped = work.groupby(cols, dropna=False)
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
    if "failure_mode" in out.columns:
        out = _label_coded(out, "failure_mode", FAILURE_MODE_LABELS, FAILURE_MODE_ORDER)
    if "failure_mode_judge" in out.columns:
        out = _label_coded(
            out, "failure_mode_judge", FAILURE_MODE_LABELS, FAILURE_MODE_ORDER
        )
    if "recall_bin" in out.columns:
        out = _label_coded(out, "recall_bin", RECALL_BIN_LABELS, RECALL_BIN_ORDER)
    if "judge_vs_f1" in out.columns:
        out = _label_coded(out, "judge_vs_f1", JUDGE_F1_LABELS, JUDGE_F1_ORDER)
    if "retrieval_calls_bin" in out.columns:
        out = _label_coded(out, "retrieval_calls_bin", {}, RETRIEVAL_CALLS_ORDER)
    if "hop_bin" in out.columns:
        out = _label_coded(out, "hop_bin", {}, HOP_BIN_ORDER)
    if "qidx_bin" in out.columns:
        out = _label_coded(out, "qidx_bin", {}, QIDX_BIN_ORDER)
    if "notes_bytes_bin" in out.columns:
        out = _label_coded(out, "notes_bytes_bin", {}, NOTES_BYTES_ORDER)
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


def _label_coded(
    table: pd.DataFrame,
    col: str,
    labels: dict[str, str],
    order: tuple[str, ...],
) -> pd.DataFrame:
    out = table.copy()
    raw = out[col].astype(str)
    rank = {key: i for i, key in enumerate(order)}
    out["_ord"] = raw.map(lambda v: rank.get(v, 99))
    if labels:
        out[col] = [labels.get(v, v) for v in raw]
    out = out.sort_values("_ord", kind="mergesort").reset_index(drop=True)
    return out.drop(columns=["_ord"])


def _label_failure_modes(table: pd.DataFrame) -> pd.DataFrame:
    return _label_coded(table, "failure_mode", FAILURE_MODE_LABELS, FAILURE_MODE_ORDER)
