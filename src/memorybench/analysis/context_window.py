"""Context-window utilization vs memory coverage (offline, no LLM).

Full-context in Mem0 / this repo is the **conversation** behind one LoCoMo
question, not the whole dataset and not a filled model window. Paper Table 2
lists 26,031 memory tokens for that baseline.

Who consumes this: ``scripts.analysis.context_window`` and notebook 16.
Does not run as a memorybench job.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.analysis.campaign_tables import (
    annotate_generation,
    annotate_model_family,
    annotate_thinking,
)
from src.locomo_eval.mem0_baselines import (
    ADVERSARIAL_CATEGORY,
    literature_overall_j,
    paper_full_context_memory_tokens,
)
from src.locomo_eval.pricing import estimate_usd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WINDOWS_PATH = ROOT / "configs" / "models" / "context_windows.yaml"
DEFAULT_OUT = ROOT / "experiments" / "_campaign" / "context_window"

# Conversation-associated methods still cover every session; RAG does not.
COVERAGE_KIND = {
    "full_context": "all_turns",
    "raw_chunks": "all_turns",
    "session_summaries": "all_sessions_compressed",
    "teacher_session_summaries": "all_sessions_compressed",
    "rag": "topk_chunks",
    "mem0": "retrieved_facts",
    "mem0g": "retrieved_facts_plus_graph",
    "openai_memory": "extracted_facts_all",
    "teacher_graph": "graph_triples",
    "pooled_teacher_graph": "graph_triples",
    "fused_teacher_graph": "graph_triples",
}

DEFAULT_PACKS = (
    "locomo-2025-readers-openai-deepseek",
    "locomo-2026-readers-openai-deepseek",
    "locomo-mem0-reader-2025-writers-openai-deepseek",
    "locomo-mem0-reader-2026-writers-openai-deepseek",
)
DEFAULT_AUDIT_RUN = "full_context_qa"
DEFAULT_PREPROCESS = "locomo_preprocess"
DEFAULT_LOCAL_RUNS = ("full_context_qa", "rag_k2_256_qa", "mem0_qa")

INJECTION = {
    "builder": "FullContextMemoryBuilder",
    "code": "src/locomo_eval/rag/builders.py",
    "grammar": "{timestamp} | {speaker}: {text}",
    "prompt": "prompts/readers/qa_mem0_v1.txt",
    "scope": (
        "the LoCoMo conversation tied to the question — not the dataset, "
        "not a filled context window"
    ),
    "retrieve": False,
    "truncation": False,
    "paper_memory_tokens": None,  # filled at load from Table 2
}


@dataclass(frozen=True)
class WindowSpec:
    """Published window for one API id. Analysis-only; not a runtime cap."""

    model_id: str
    context_window: int
    max_input: int | None
    source: str


@dataclass
class ContextWindowReport:
    out_dir: Path
    injection: dict[str, Any]
    conversation_table: pd.DataFrame
    cell_table: pd.DataFrame
    year_table: pd.DataFrame
    csv_paths: list[Path] = field(default_factory=list)
    plot_paths: list[Path] = field(default_factory=list)
    summary_path: Path | None = None
    skipped_packs: list[str] = field(default_factory=list)


def load_context_windows(path: str | Path | None = None) -> dict[str, WindowSpec]:
    dest = Path(path) if path is not None else DEFAULT_WINDOWS_PATH
    raw = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
    models: dict[str, WindowSpec] = {}
    for key, block in (raw.get("models") or {}).items():
        if not isinstance(block, dict):
            continue
        models[str(key)] = WindowSpec(
            model_id=str(key),
            context_window=int(block["context_window"]),
            max_input=int(block["max_input"]) if block.get("max_input") else None,
            source=str(block.get("source") or ""),
        )
    for alias, target in (raw.get("aliases") or {}).items():
        if target in models:
            models[str(alias)] = models[str(target)]
    return models


def resolve_window(
    model_id: str | None, windows: dict[str, WindowSpec] | None = None
) -> WindowSpec | None:
    """Exact catalog key, then alias, then prefix before a date suffix."""
    if not model_id:
        return None
    table = windows if windows is not None else load_context_windows()
    key = str(model_id).strip()
    if key in table:
        return table[key]
    lower = key.lower()
    for candidate, spec in table.items():
        if candidate.lower() == lower:
            return spec
    if "-" in key:
        prefix = key.rsplit("-", 1)[0]
        if prefix in table:
            return table[prefix]
    return None


def utilization(prompt_tokens: float | None, window: int | None) -> float | None:
    """Billed input / published context window. None if either side is missing."""
    if prompt_tokens is None or window is None or window <= 0:
        return None
    return float(prompt_tokens) / float(window)


def judged_mask(frame: pd.DataFrame) -> pd.Series:
    """Mem0 J drops LoCoMo category 5 (adversarial)."""
    if "question_category" not in frame.columns:
        return pd.Series(True, index=frame.index)
    cat = pd.to_numeric(frame["question_category"], errors="coerce")
    return cat != ADVERSARIAL_CATEGORY


def coverage_kind(memory_method: str | None) -> str:
    if not memory_method:
        return "unknown"
    return COVERAGE_KIND.get(str(memory_method), "unknown")


def render_context_window_report(
    *,
    root: Path | None = None,
    out_dir: Path | None = None,
    packs: list[str] | tuple[str, ...] | None = None,
    audit_run: str | None = DEFAULT_AUDIT_RUN,
    preprocess: str | None = DEFAULT_PREPROCESS,
    local_runs: list[str] | tuple[str, ...] | None = DEFAULT_LOCAL_RUNS,
    windows_path: Path | None = None,
) -> ContextWindowReport:
    """Write tables/plots/SUMMARY from finished packs + audit dumps."""
    from scripts.analysis.context_window_plots import write_context_window_plots

    root = Path(root) if root is not None else ROOT
    dest = Path(out_dir) if out_dir is not None else root / "experiments" / "_campaign" / "context_window"
    windows = load_context_windows(windows_path)
    injection = dict(INJECTION)
    injection["paper_memory_tokens"] = paper_full_context_memory_tokens()

    skipped: list[str] = []
    cell_parts: list[pd.DataFrame] = []
    pack_names = list(packs) if packs is not None else list(DEFAULT_PACKS)
    for name in pack_names:
        frame, miss = _load_pack_examples(root / "experiments" / name)
        if miss:
            skipped.append(miss)
            continue
        cell_parts.append(_cell_metrics(frame, windows, pack=name))

    for name in list(local_runs or ()):
        frame, miss = _load_local_run(root / "experiments" / name)
        if miss:
            skipped.append(miss)
            continue
        cell_parts.append(_cell_metrics(frame, windows, pack=name, result_source="local_clone"))

    cell_table = _concat_frames(cell_parts)
    if not cell_table.empty:
        cell_table = _attach_coverage(cell_table)
        cell_table = _attach_paper_row(cell_table, windows)

    conversation = _conversation_table(
        root,
        audit_run=audit_run,
        preprocess=preprocess,
    )
    year_table = _year_utilization(cell_table)

    dest.mkdir(parents=True, exist_ok=True)
    tables_dir = dest / "tables"
    plots_dir = dest / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: list[Path] = []
    for name, table in (
        ("cells", cell_table),
        ("conversations", conversation),
        ("year_utilization", year_table),
    ):
        if table.empty:
            continue
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        csv_paths.append(path)

    summary_path = dest / "SUMMARY.md"
    summary_path.write_text(
        _summary_markdown(injection, cell_table, conversation, year_table, skipped),
        encoding="utf-8",
    )
    plot_paths = write_context_window_plots(
        plots_dir,
        cell_table=cell_table,
        conversation_table=conversation,
        year_table=year_table,
    )
    return ContextWindowReport(
        out_dir=dest,
        injection=injection,
        conversation_table=conversation,
        cell_table=cell_table,
        year_table=year_table,
        csv_paths=csv_paths,
        plot_paths=plot_paths,
        summary_path=summary_path,
        skipped_packs=skipped,
    )


def notebook_show_context_window(report: ContextWindowReport) -> None:
    """Display SUMMARY + tables + plots in Jupyter (prints if IPython missing)."""
    try:
        from IPython.display import Image, Markdown, display
    except ImportError:
        print(report.summary_path)
        print(report.cell_table.to_string(index=False) if not report.cell_table.empty else "no cells")
        for path in report.plot_paths:
            print("plot", path)
        return
    if report.skipped_packs:
        display(Markdown("**Skipped:** " + ", ".join(report.skipped_packs)))
    if report.summary_path and report.summary_path.is_file():
        display(Markdown(report.summary_path.read_text(encoding="utf-8")))
    if not report.conversation_table.empty:
        display(Markdown("### Per-conversation transcript size"))
        display(report.conversation_table)
    if not report.cell_table.empty:
        display(Markdown("### Cell window vs input"))
        show_cols = [
            c
            for c in (
                "pack",
                "memory_method",
                "reader_model",
                "thinking",
                "agent_input_tokens_mean",
                "context_window",
                "window_utilization",
                "coverage_vs_full_context",
                "judge_score",
                "locomo_f1",
                "judge_score_per_1k_input",
                "usd_cell",
            )
            if c in report.cell_table.columns
        ]
        display(report.cell_table[show_cols])
    for path in report.plot_paths:
        display(Image(filename=str(path)))


def _load_pack_examples(pack: Path) -> tuple[pd.DataFrame, str | None]:
    path = pack / "aggregate" / "examples.parquet"
    if not path.is_file():
        path = pack / "examples.parquet"
    if not path.is_file():
        return pd.DataFrame(), pack.name
    df = pd.read_parquet(path)
    if df.empty:
        return df, pack.name
    family_from = "writer" if _is_writer_pack(df) else "reader"
    df = annotate_model_family(df, family_from)
    df = annotate_generation(df, "reader")
    df = annotate_thinking(df, pack)
    df["pack"] = pack.name
    df["result_source"] = "live"
    return df, None


def _is_writer_pack(df: pd.DataFrame) -> bool:
    if "writer_model" not in df.columns:
        return False
    return df["writer_model"].notna().any()


def _load_local_run(run_dir: Path) -> tuple[pd.DataFrame, str | None]:
    preds = run_dir / "predictions.jsonl"
    if not preds.is_file():
        return pd.DataFrame(), run_dir.name
    rows: list[dict[str, Any]] = []
    with preds.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            usage = rec.get("usage") or {}
            rows.append(
                {
                    "run_id": rec.get("run_id") or run_dir.name,
                    "pack": run_dir.name,
                    "memory_method": rec.get("memory_type"),
                    "reader_model": rec.get("reader_model"),
                    "conversation_id": rec.get("sample_id"),
                    "question_id": rec.get("question_id"),
                    "question_category": rec.get("category"),
                    "agent_input_tokens": usage.get("prompt_tokens"),
                    "agent_output_tokens": usage.get("completion_tokens"),
                    "agent_latency_seconds": rec.get("latency_s"),
                    "locomo_f1": None,
                    "judge_score": None,
                    "thinking": "off",
                    "result_source": "local_clone",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, run_dir.name
    metrics = _read_json(run_dir / "metrics.json") or {}
    overall = (metrics.get("metrics") or metrics).get("locomo_f1")
    if overall is not None:
        df["locomo_f1"] = float(overall)
    judge = _autorater_j(run_dir)
    if judge is not None:
        mask = judged_mask(df)
        df.loc[mask, "judge_score"] = float(judge)
    df = annotate_model_family(df, "reader")
    df = annotate_generation(df, "reader")
    return df, None


def _autorater_j(run_dir: Path) -> float | None:
    payload = _read_json(run_dir / "autorater" / "autorater_metrics.json") or {}
    metrics = payload.get("metrics") or {}
    raw = metrics.get("llm_judge")
    if raw is None:
        return None
    value = float(raw)
    return value / 100.0 if value > 1.0 else value


def _cell_metrics(
    examples: pd.DataFrame,
    windows: dict[str, WindowSpec],
    *,
    pack: str,
    result_source: str | None = None,
) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame()
    keys = [
        c
        for c in (
            "pack",
            "run_id",
            "memory_method",
            "reader_model",
            "reader_display_name",
            "writer_model",
            "thinking",
            "model_family",
            "generation",
            "result_source",
        )
        if c in examples.columns
    ]
    rows: list[dict[str, Any]] = []
    grouped = examples.groupby(keys, dropna=False, sort=False)
    for key_vals, piece in grouped:
        key_map = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        row = dict(key_map)
        row["pack"] = pack
        if result_source:
            row["result_source"] = result_source
        tokens = pd.to_numeric(piece.get("agent_input_tokens"), errors="coerce")
        row["n"] = int(len(piece))
        row["agent_input_tokens_mean"] = _stat(tokens, "mean")
        row["agent_input_tokens_min"] = _stat(tokens, "min")
        row["agent_input_tokens_p50"] = _stat(tokens, "median")
        row["agent_input_tokens_max"] = _stat(tokens, "max")
        row["agent_input_tokens_sum"] = _stat(tokens, "sum")
        out_tok = pd.to_numeric(piece.get("agent_output_tokens"), errors="coerce")
        row["agent_output_tokens_mean"] = _stat(out_tok, "mean")
        row["agent_output_tokens_sum"] = _stat(out_tok, "sum")
        lat = pd.to_numeric(piece.get("agent_latency_seconds"), errors="coerce")
        row["agent_latency_seconds_mean"] = _stat(lat, "mean")
        f1 = pd.to_numeric(piece.get("locomo_f1"), errors="coerce")
        row["locomo_f1"] = _stat(f1, "mean")
        judged = piece.loc[judged_mask(piece)]
        row["n_judged"] = int(len(judged))
        j = pd.to_numeric(judged.get("judge_score"), errors="coerce") if not judged.empty else None
        row["judge_score"] = _stat(j, "mean") if j is not None else None
        f1_j = pd.to_numeric(judged.get("locomo_f1"), errors="coerce") if not judged.empty else None
        row["locomo_f1_judged"] = _stat(f1_j, "mean") if f1_j is not None else None
        spec = resolve_window(row.get("reader_model"), windows)
        row["context_window"] = spec.context_window if spec else None
        row["max_input"] = spec.max_input if spec else None
        row["window_source"] = spec.source if spec else None
        row["window_utilization"] = utilization(
            row["agent_input_tokens_mean"], row["context_window"]
        )
        row["window_headroom"] = (
            None
            if row["window_utilization"] is None
            else 1.0 - float(row["window_utilization"])
        )
        row["coverage_kind"] = coverage_kind(row.get("memory_method"))
        mean_in = row["agent_input_tokens_mean"]
        if mean_in and mean_in > 0 and row["judge_score"] is not None:
            row["judge_score_per_1k_input"] = float(row["judge_score"]) / (
                float(mean_in) / 1000.0
            )
        else:
            row["judge_score_per_1k_input"] = None
        if mean_in and mean_in > 0 and row["locomo_f1"] is not None:
            row["locomo_f1_per_1k_input"] = float(row["locomo_f1"]) / (
                float(mean_in) / 1000.0
            )
        else:
            row["locomo_f1_per_1k_input"] = None
        if row["agent_latency_seconds_mean"] and row["locomo_f1"] is not None:
            row["locomo_f1_per_second"] = float(row["locomo_f1"]) / float(
                row["agent_latency_seconds_mean"]
            )
        else:
            row["locomo_f1_per_second"] = None
        prompt_sum = row["agent_input_tokens_sum"]
        completion_sum = row["agent_output_tokens_sum"] or 0
        usd = None
        if prompt_sum is not None:
            usd = estimate_usd(
                row.get("reader_model"),
                int(prompt_sum),
                int(completion_sum or 0),
            )
        row["usd_cell"] = usd
        if usd and usd > 0 and row["judge_score"] is not None:
            row["judge_score_per_usd"] = float(row["judge_score"]) / float(usd)
        else:
            row["judge_score_per_usd"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_coverage(cells: pd.DataFrame) -> pd.DataFrame:
    """Coverage vs same-reader full_context mean input (else paper 26,031)."""
    out = cells.copy()
    paper = float(paper_full_context_memory_tokens())
    fc = out[out["memory_method"].astype(str) == "full_context"]
    lookup: dict[str, float] = {}
    if not fc.empty and "reader_model" in fc.columns:
        for model, piece in fc.groupby("reader_model", dropna=False):
            mean = _stat(pd.to_numeric(piece["agent_input_tokens_mean"], errors="coerce"), "mean")
            if mean:
                lookup[str(model)] = float(mean)
    ratios: list[float | None] = []
    for _, row in out.iterrows():
        denom = lookup.get(str(row.get("reader_model") or ""), paper)
        num = row.get("agent_input_tokens_mean")
        if num is None or pd.isna(num) or not denom:
            ratios.append(None)
        else:
            ratios.append(float(num) / float(denom))
    out["coverage_vs_full_context"] = ratios
    return out


def _attach_paper_row(cells: pd.DataFrame, windows: dict[str, WindowSpec]) -> pd.DataFrame:
    tokens = paper_full_context_memory_tokens()
    j = literature_overall_j("Full-context")
    spec = resolve_window("gpt-4o", windows)
    window = spec.context_window if spec else 128000
    row = {
        "pack": "mem0_paper_table2",
        "memory_method": "full_context",
        "reader_model": "paper-gpt-4o-class",
        "model_family": "OpenAI",
        "generation": "2024",
        "thinking": "off",
        "result_source": "paper",
        "n": None,
        "agent_input_tokens_mean": float(tokens),
        "context_window": window,
        "window_utilization": utilization(float(tokens), window),
        "window_headroom": 1.0 - float(utilization(float(tokens), window) or 0),
        "coverage_kind": "all_turns",
        "coverage_vs_full_context": 1.0,
        "judge_score": (float(j) / 100.0) if j is not None and j > 1 else j,
        "locomo_f1": None,
    }
    if row["judge_score"] is not None and tokens:
        row["judge_score_per_1k_input"] = float(row["judge_score"]) / (tokens / 1000.0)
    extra = pd.DataFrame([row])
    return _concat_frames([cells, extra])


def _year_utilization(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()
    fc = cells[cells["memory_method"].astype(str) == "full_context"].copy()
    if fc.empty or "generation" not in fc.columns:
        return pd.DataFrame()
    keys = [c for c in ("generation", "model_family", "thinking", "result_source") if c in fc.columns]
    metrics = [
        c
        for c in (
            "window_utilization",
            "agent_input_tokens_mean",
            "judge_score",
            "judge_score_per_1k_input",
            "locomo_f1",
            "usd_cell",
        )
        if c in fc.columns
    ]
    if not keys or not metrics:
        return pd.DataFrame()
    return (
        fc.groupby(keys, dropna=False, as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(keys)
    )


def _conversation_table(
    root: Path, *, audit_run: str | None, preprocess: str | None
) -> pd.DataFrame:
    if not audit_run:
        return pd.DataFrame()
    run = root / "experiments" / audit_run
    index_path = run / "memory" / "index.jsonl"
    if not index_path.is_file():
        return pd.DataFrame()
    mem_rows = [
        rec
        for rec in _jsonl(index_path)
        if rec.get("key_kind") in (None, "sample")
    ]
    sessions = _session_counts(root / "experiments" / preprocess) if preprocess else {}
    billed = _billed_by_conversation(run)
    rows: list[dict[str, Any]] = []
    for rec in mem_rows:
        sid = str(rec.get("sample_id") or "")
        chars = rec.get("n_chars")
        turns = rec.get("n_source_ids")
        prompt = billed.get(sid)
        rows.append(
            {
                "conversation_id": sid,
                "n_sessions": sessions.get(sid),
                "n_turns": turns,
                "n_chars": chars,
                "agent_input_tokens_mean": prompt,
                "chars_per_token": (
                    float(chars) / float(prompt) if chars and prompt else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _session_counts(preprocess_dir: Path) -> dict[str, int]:
    by_sample = preprocess_dir / "preprocess" / "by_sample"
    if not by_sample.is_dir():
        return {}
    out: dict[str, int] = {}
    for folder in by_sample.iterdir():
        sess = folder / "sessions.jsonl"
        if not sess.is_file():
            continue
        out[folder.name] = sum(1 for line in sess.open(encoding="utf-8") if line.strip())
    return out


def _billed_by_conversation(run_dir: Path) -> dict[str, float]:
    preds = run_dir / "predictions.jsonl"
    if not preds.is_file():
        return {}
    buckets: dict[str, list[float]] = {}
    with preds.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            usage = rec.get("usage") or {}
            tok = usage.get("prompt_tokens")
            sid = rec.get("sample_id")
            if tok is None or not sid:
                continue
            buckets.setdefault(str(sid), []).append(float(tok))
    return {sid: sum(vals) / len(vals) for sid, vals in buckets.items()}


def _summary_markdown(
    injection: dict[str, Any],
    cells: pd.DataFrame,
    conversations: pd.DataFrame,
    year_table: pd.DataFrame,
    skipped: list[str],
) -> str:
    lines = [
        "# Context window vs full-context coverage",
        "",
        "## What `full_context` means",
        "",
        f"- Builder: `{injection['builder']}` (`{injection['code']}`).",
        f"- Grammar: `{injection['grammar']}` (Mem0 `clean_chat_history`).",
        f"- Prompt fill: `{injection['prompt']}` placeholders `{{memory}}` + `{{question}}`.",
        f"- Scope: {injection['scope']}.",
        "- Retrieval: none (search latency 0). Truncation: none (`max_chars` is null).",
        f"- Mem0 paper Table 2 Full-context `memory_tokens`: **{injection['paper_memory_tokens']}**.",
        "",
        "Eval is an outer loop over questions. Each question already knows its",
        "conversation. Retrieval (RAG / Mem0) stays **inside** that conversation.",
        "",
        "## Dataset vs memory granularity",
        "",
        "- LoCoMo → conversations → sessions → turns. One question belongs to one conversation.",
        "- `full_context`: raw transcript of that conversation.",
        "- Session summaries: still every session, one compressed block each.",
        "- RAG: top-k token chunks (k in {1,2}), not every session.",
        "- Mem0 / Mem0g: extracted facts (+ graph), retrieved inside the conversation.",
        "",
    ]
    if skipped:
        lines.append("Skipped packs: " + ", ".join(skipped))
        lines.append("")
    if not conversations.empty and "agent_input_tokens_mean" in conversations.columns:
        tok = pd.to_numeric(conversations["agent_input_tokens_mean"], errors="coerce")
        lines.append("## Local transcript sizes (`full_context_qa`)")
        lines.append("")
        lines.append(
            f"- Conversations: **{len(conversations)}**; billed input "
            f"min {tok.min():.0f} / mean {tok.mean():.0f} / max {tok.max():.0f}."
        )
        if "n_sessions" in conversations.columns:
            sess = pd.to_numeric(conversations["n_sessions"], errors="coerce")
            lines.append(
                f"- Sessions per conversation: {int(sess.min())}–{int(sess.max())}."
            )
        lines.append("")
    if not cells.empty and "window_utilization" in cells.columns:
        fc = cells[
            (cells["memory_method"].astype(str) == "full_context")
            & (cells.get("result_source", "live") != "paper")
        ]
        if not fc.empty:
            util = pd.to_numeric(fc["window_utilization"], errors="coerce")
            lines.append("## Window vs input (live + local_clone full_context)")
            lines.append("")
            lines.append(
                f"- Utilization min {100 * util.min():.1f}% / "
                f"median {100 * util.median():.1f}% / max {100 * util.max():.1f}%."
            )
            lines.append(
                "- Larger 2025–2026 windows **lower** fill rate; LoCoMo transcripts "
                "stay ~28k tokens. Idle window is not extra evidence."
            )
            lines.append("")
    lines.extend(
        [
            "## Why scores follow coverage, not fill rate",
            "",
            "- RAG input is ~0.9k tokens (~3% of FC): multi-hop/temporal evidence is often missing.",
            "- Teacher session summaries keep every session at ~5–8k tokens (~20% of FC).",
            "- Thinking-on helps FC more than RAG because the transcript is already in the prompt.",
            "- Do not read a 1M window as a LoCoMo advantage once 33k already fits.",
            "",
        ]
    )
    if not year_table.empty:
        lines.append("Year-utilization table: `tables/year_utilization.csv`.")
        lines.append("")
    return "\n".join(lines)


def _stat(series: pd.Series | None, how: str) -> float | None:
    if series is None:
        return None
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    if how == "mean":
        return float(clean.mean())
    if how == "median":
        return float(clean.median())
    if how == "min":
        return float(clean.min())
    if how == "max":
        return float(clean.max())
    if how == "sum":
        return float(clean.sum())
    return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    cols: list[str] = []
    seen: set[str] = set()
    for frame in nonempty:
        for col in frame.columns:
            if col not in seen:
                cols.append(col)
                seen.add(col)
    aligned = [frame.reindex(columns=cols) for frame in nonempty]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat(aligned, ignore_index=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
