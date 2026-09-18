"""Write notebook-friendly Parquet next to the human JSONL audit pack."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.locomo_eval.metrics import score_row

EXAMPLE_COLUMNS = (
    "run_id",
    "experiment_name",
    "experiment_type",
    "benchmark",
    "memory_method",
    "seed",
    "reader_provider",
    "reader_model",
    "reader_display_name",
    "reader_family",
    "reader_generation",
    "writer_model",
    "judge_provider",
    "judge_model",
    "prompt_path",
    "conversation_id",
    "question_id",
    "question_category",
    "question",
    "reference_answer",
    "generated_answer",
    "retrieved_memories",
    "exact_match",
    "token_f1",
    "locomo_f1",
    "judge_score",
    "judge_reasoning",
    "agent_input_tokens",
    "agent_output_tokens",
    "agent_reasoning_tokens",
    "agent_latency_seconds",
    "search_latency_seconds",
    "total_latency_seconds",
    "thinking",
    "judge_input_tokens",
    "judge_output_tokens",
    "judge_latency_seconds",
    "retry_count",
)


def write_examples_parquet(
    dest: Path,
    *,
    spec,
    prediction_rows: list[dict[str, Any]],
    verdicts_by_qid: dict[str, dict[str, Any]] | None = None,
) -> Path:
    verdicts_by_qid = verdicts_by_qid or {}
    records = [
        _example_record(spec, row, verdicts_by_qid.get(str(row.get("question_id") or "")))
        for row in prediction_rows
    ]
    return _write_parquet(dest, records, list(EXAMPLE_COLUMNS))


def write_summary_parquet(dest: Path, summary_row: dict[str, Any]) -> Path:
    return _write_parquet(dest, [summary_row], list(summary_row.keys()))


def load_prediction_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_autorater_verdicts(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "autorater" / "autorater_verdicts.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or "")
            if qid:
                out[qid] = row
    return out


def _example_record(
    spec, row: dict[str, Any], verdict: dict[str, Any] | None
) -> dict[str, Any]:
    scores = score_row(
        str(row.get("predicted_answer") or ""),
        str(row.get("reference_answer") or ""),
        int(row.get("category") or 0),
    )
    usage = row.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    verdict = verdict or {}
    judge_usage = verdict.get("usage") or {}
    if not isinstance(judge_usage, dict):
        judge_usage = {}
    llm_score = verdict.get("llm_score")
    if llm_score is None and "label" in verdict:
        llm_score = 1 if str(verdict.get("label")).upper() == "CORRECT" else 0
    return {
        "run_id": spec.run_id,
        "experiment_name": spec.experiment_name,
        "experiment_type": spec.experiment_type,
        "benchmark": spec.benchmark,
        "memory_method": spec.memory_method,
        "seed": int(spec.seed),
        "reader_provider": spec.reader.provider,
        "reader_model": spec.reader.api_model_id,
        "reader_display_name": spec.reader.display_name,
        "reader_family": spec.reader.family,
        "reader_generation": _gen_str(spec.reader.generation),
        "writer_model": spec.writer.api_model_id if spec.writer else None,
        "judge_provider": spec.judge_provider,
        "judge_model": spec.judge_model,
        "prompt_path": spec.prompt_path,
        "conversation_id": str(row.get("sample_id") or ""),
        "question_id": str(row.get("question_id") or ""),
        "question_category": int(row.get("category") or 0),
        "question": str(row.get("question") or ""),
        "reference_answer": str(row.get("reference_answer") or ""),
        "generated_answer": str(row.get("predicted_answer") or ""),
        "retrieved_memories": str(row.get("memory_text") or ""),
        "exact_match": float(scores["exact_match"]),
        "token_f1": float(scores["token_f1"]),
        "locomo_f1": float(scores["locomo_f1"]),
        "judge_score": _as_float(llm_score),
        "judge_reasoning": _as_str(verdict.get("reasoning") or verdict.get("raw_text")),
        "agent_input_tokens": _as_int(usage.get("prompt_tokens")),
        "agent_output_tokens": _as_int(usage.get("completion_tokens")),
        "agent_reasoning_tokens": _as_int(
            usage.get("reasoning_tokens")
            if usage.get("reasoning_tokens") is not None
            else row.get("reasoning_tokens")
        ),
        "agent_latency_seconds": _as_float(row.get("latency_s")),
        "search_latency_seconds": _as_float(row.get("search_latency_s")),
        "total_latency_seconds": _total_latency_seconds(
            row.get("search_latency_s"), row.get("latency_s")
        ),
        "thinking": spec.thinking_label(),
        "judge_input_tokens": _as_int(judge_usage.get("prompt_tokens")),
        "judge_output_tokens": _as_int(judge_usage.get("completion_tokens")),
        "judge_latency_seconds": _as_float(verdict.get("latency_s")),
        "retry_count": _as_int(row.get("retry_count")) or 0,
    }


def qa_summary_row(
    spec,
    *,
    run_dir: Path,
    n_examples: int,
    metrics: dict[str, Any],
    meta: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    m = (metrics or {}).get("metrics") or metrics or {}
    cost = _read_json(run_dir / "cost.json")
    teacher_cost = cost.get("teacher") or {}
    reader_cost = cost.get("reader") or {}
    quality = _read_json(run_dir / "memory" / "teachers" / "quality.json")
    teacher_latency = _mean_teacher_latency(quality)
    return {
        "run_id": spec.run_id,
        "experiment_name": spec.experiment_name,
        "experiment_type": spec.experiment_type,
        "benchmark": spec.benchmark,
        "memory_method": spec.memory_method,
        "seed": int(spec.seed),
        "reader_provider": spec.reader.provider,
        "reader_model": spec.reader.api_model_id,
        "reader_display_name": spec.reader.display_name,
        "judge_provider": spec.judge_provider,
        "judge_model": spec.judge_model,
        "writer_model": spec.writer.api_model_id if spec.writer else None,
        "thinking": spec.thinking_label(),
        "num_examples": int(n_examples),
        "primary_score": _as_float(m.get("locomo_f1")),
        "exact_match": _as_float(m.get("exact_match")),
        "token_f1": _as_float(m.get("token_f1")),
        "locomo_f1": _as_float(m.get("locomo_f1")),
        "agent_reasoning_tokens": _as_int(reader_cost.get("reasoning_tokens")),
        "teacher_reasoning_tokens": _as_int(teacher_cost.get("reasoning_tokens")),
        "teacher_n_calls": _as_int(teacher_cost.get("n_calls")),
        "teacher_latency_seconds": teacher_latency,
        "git_commit": meta.get("code_git_hash"),
        "config_hash": spec.run_id.rsplit("-", 1)[-1],
        "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
    }


def _write_parquet(dest: Path, records: list[dict[str, Any]], columns: list[str]) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to write analysis Parquet. "
            "pip install pyarrow pandas"
        ) from exc
    if not records:
        table = pa.table({col: [] for col in columns})
    else:
        aligned = [{col: rec.get(col) for col in columns} for rec in records]
        table = pa.Table.from_pylist(aligned)
    pq.write_table(table, dest)
    return dest


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _total_latency_seconds(search: Any, generate: Any) -> float | None:
    """Mem0 Table 2 total = search + answer generate. Missing search counts as 0."""
    gen = _as_float(generate)
    if gen is None:
        return None
    found = _as_float(search)
    return round(gen + (0.0 if found is None else found), 6)


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _gen_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_teacher_latency(quality: dict[str, Any]) -> float | None:
    """Mean teacher-call latency from quality.json (write path, not the frozen reader)."""
    by_teacher = quality.get("by_teacher") or {}
    if not isinstance(by_teacher, dict) or not by_teacher:
        return None
    weighted = 0.0
    n = 0
    for rec in by_teacher.values():
        latency = rec.get("mean_latency_s")
        calls = rec.get("n_calls") or 0
        if latency is None or int(calls) <= 0:
            continue
        weighted += float(latency) * int(calls)
        n += int(calls)
    if n <= 0:
        return None
    return weighted / n
