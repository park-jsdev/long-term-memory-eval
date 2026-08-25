"""CLI: run the LoCoMo QA pipeline for **one** memory config.

This module does not compare conditions and is not a multi-run orchestrator.
Typical sandwich experiment:

1. Call this once with config A (e.g. ``configs/raw_chunks.yaml``) → ``experiments/<run_id_A>/``
2. Call this once with config B (e.g. ``configs/session_summaries.yaml``) → ``experiments/<run_id_B>/``
3. Compare the two audit packs offline:
   ``python scripts/compare_full_runs.py --runs experiments/<run_id_A> experiments/<run_id_B> ...``
   (memory axis) or ``scripts/compare_cross_model.py`` (reader/teacher model axis)

Each ``--config`` YAML is standalone. ``pipeline.memory`` is a builder id in
memory.py (``raw_chunks`` / ``session_summaries``), not a path to another YAML.

    python -m src.locomo_eval.run --config configs/raw_chunks.yaml --run-id cmp_raw_chunks
    python -m src.locomo_eval.run --config configs/session_summaries.yaml --run-id cmp_session_summaries
    python scripts/compare_full_runs.py --runs experiments/cmp_raw_chunks experiments/cmp_session_summaries --out experiments/compare_raw_chunks_session_summaries
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root on sys.path so `src.*` imports work when run as a module or script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.locomo_eval.dataset import iter_questions, load_conversations
from src.locomo_eval.env import load_env
from src.locomo_eval.memory import (
    TeacherSessionMemoryBuilder,
    get_memory_builder,
    is_question_independent,
    resolve_memory_name,
)
from src.locomo_eval.memory_log import write_memory_run_log
from src.locomo_eval.metrics import summarize_predictions
from src.locomo_eval.models import resolve_model
from src.locomo_eval.prompts import load_prompt_template
from src.locomo_eval.readers import get_reader
from src.locomo_eval.report import write_jsonl, write_run_report
from src.locomo_eval.schemas import Memory, Prediction
from src.locomo_eval.teachers import get_teacher


def _git_hash() -> str | None:
    """Pin HEAD in run_meta so you can check out this commit and reproduce the run.

    Returns None if git isn't available; missing git should not fail the experiment.
    TODO: Remove cross-version features after development.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _file_sha256(path: Path) -> str | None:
    """Fingerprint the input JSON so you can tell if a later run used different data."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_teacher(cfg: dict, overrides: argparse.Namespace, memory_name: str, reader_name: str):
    """Construct a Teacher only for teacher_session_summaries. Other builders ignore teacher YAML.

    LlmResponseHash is not passed (future optimization; same as the answer reader).
    """
    if resolve_memory_name(memory_name) != TeacherSessionMemoryBuilder.name:
        return None
    tcfg = cfg.get("teacher") or {}
    provider = getattr(overrides, "teacher", None) or tcfg.get("provider")
    model = getattr(overrides, "teacher_model", None) or tcfg.get("model")
    if not provider:
        provider = "mock" if str(reader_name).lower() == "mock" else "openai"
    if not model:
        raise SystemExit(
            "teacher_session_summaries requires teacher.model in YAML or --teacher-model"
        )
    prompt_path = tcfg.get("prompt_path")
    return get_teacher(
        provider,
        model=model,
        temperature=float(tcfg.get("temperature", 0.0)),
        max_tokens=int(tcfg.get("max_tokens", 512)),
        llm_response_hash=None,
        max_retries=int(tcfg.get("max_retries", 8)),
        min_request_interval_s=float(tcfg.get("min_request_interval_s", 0.5)),
        max_wait_s=float(tcfg.get("max_wait_s", 3600.0)),
        prompt_path=prompt_path,
    )


def _reset_run_output(run_dir: Path) -> None:
    """Clear generated artifacts so one run id always means one fresh run.

    Evaluation must never reuse answer responses. Removing the dependent
    autorater pack also prevents reports based on old predictions surviving a
    regenerated source run.
    """
    for dirname in ("plots", "memory", "autorater"):
        path = run_dir / dirname
        if path.is_dir():
            shutil.rmtree(path)
    for filename in (
        "predictions.jsonl",
        "predictions.csv",
        "metrics.json",
        "metrics_by_category.csv",
        "run_meta.json",
    ):
        path = run_dir / filename
        if path.is_file():
            path.unlink()


def run_locomo_pipeline_with_memory_config(cfg: dict, overrides: argparse.Namespace) -> Path:
    """Run load → memory builder → answer LLM → string metrics for one YAML.

    Returns the ``experiments/<run_id>/`` audit directory. Call this once per
    memory config; do not pass two builders here. A vs B is two invocations
    plus ``scripts/compare_full_runs.py`` (or ``compare_cross_model.py``).

    Which memory builder runs is ``cfg["pipeline"]["memory"]``, unless
    ``--memory`` overrides it. Not tied to ``configs/baseline.yaml``.
    """
    data_path = Path(overrides.data or cfg["data"]["raw_path"])
    # Builder id (raw_chunks / session_summaries), not a path to another YAML.
    memory_name = overrides.memory or cfg["pipeline"]["memory"]
    reader_name = overrides.reader or cfg["reader"]["provider"]
    model = overrides.model or cfg["reader"]["model"]
    prompt_path = Path(overrides.prompt or cfg["pipeline"]["prompt_path"])
    max_questions = overrides.max_questions
    if max_questions is None:
        max_questions = cfg["pipeline"].get("max_questions")

    run_id = overrides.run_id or cfg["run"].get("run_id") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _reset_run_output(run_dir)
    pred_path = run_dir / "predictions.jsonl"

    prompt_version, prompt_template = load_prompt_template(prompt_path)
    max_chars = cfg["pipeline"].get("memory_max_chars")
    if max_chars is not None:
        max_chars = int(max_chars)
    teacher = _build_teacher(cfg, overrides, memory_name, reader_name)
    builder = get_memory_builder(memory_name, max_chars=max_chars, teacher=teacher)

    rate_cfg = cfg.get("reader") or {}
    reader_model = "mock" if reader_name.lower() == "mock" else model
    reader = get_reader(
        reader_name,
        model=reader_model,
        temperature=float(rate_cfg.get("temperature", 0.0)),
        max_tokens=int(rate_cfg.get("max_tokens", 64)),
        llm_response_hash=None,  # evaluation factories prohibit response stores
        max_retries=int(rate_cfg.get("max_retries", 8)),
        min_request_interval_s=float(rate_cfg.get("min_request_interval_s", 0.0)),
        max_wait_s=float(rate_cfg.get("max_wait_s", 3600.0)),
    )

    conversations = load_conversations(data_path)
    if overrides.sample_id:
        conversations = [c for c in conversations if c.sample_id == overrides.sample_id]
        if not conversations:
            raise SystemExit(f"No sample_id match: {overrides.sample_id}")

    # These builders do not depend on the question; build once per conversation.
    memory_by_sample = {}
    if is_question_independent(memory_name):
        for conv in conversations:
            q = conv.questions[0] if conv.questions else None
            if q is None:
                continue
            memory_by_sample[conv.sample_id] = builder.build(conv, q)

    pairs = list(iter_questions(conversations))
    if max_questions is not None:
        pairs = pairs[: int(max_questions)]

    teacher_model = getattr(builder, "teacher_model", None)
    teacher_provider = getattr(builder, "teacher_provider", None)
    print(
        f"Run {run_id}: {len(pairs)} questions | memory={builder.name} | "
        f"reader={reader_name}/{reader.model_name}"
        + (f" | teacher={teacher_provider}/{teacher_model}" if teacher_model else "")
    )

    prediction_rows: list[dict] = []
    n_api = 0
    used_memories: dict[str, Memory] = {}
    example_question: str | None = None
    try:
        for i, (conv, q) in enumerate(pairs, start=1):
            if conv.sample_id in memory_by_sample:
                memory = memory_by_sample[conv.sample_id]
            else:
                memory = builder.build(conv, q)
            used_memories[conv.sample_id] = memory
            if example_question is None:
                example_question = q.question

            answer, meta = reader.answer(memory.text, q.question, prompt_template)
            if meta.get("cached"):
                raise RuntimeError(
                    "Cached answer returned in cache-free evaluation pipeline."
                )
            n_api += 1

            pred = Prediction(
                sample_id=q.sample_id,
                question_id=q.question_id,
                question=q.question,
                reference_answer=q.answer,
                predicted_answer=answer,
                category=q.category,
                memory_type=memory.memory_type,
                memory_text=memory.text,
                reader_model=reader.model_name,
                prompt_version=prompt_version,
                evidence=list(q.evidence),
                run_id=run_id,
                cached=False,
                teacher_model=memory.teacher_model,
                teacher_provider=memory.teacher_provider,
            )
            row = pred.to_dict()
            row["latency_s"] = meta.get("latency_s")
            row["usage"] = meta.get("usage")
            row["memory_schema_version"] = memory.schema_version
            prediction_rows.append(row)

            # Flush for audit after a crash. The next invocation clears this
            # partial file and regenerates; it never resumes.
            write_jsonl(pred_path, prediction_rows)

            if i % 10 == 0 or i == len(pairs):
                print(
                    f"  [{i}/{len(pairs)}] {q.question_id} api_calls_this_run={n_api}"
                )
    except Exception as exc:
        write_jsonl(pred_path, prediction_rows)
        if used_memories:
            write_memory_run_log(
                run_dir,
                used_memories,
                max_chars_cfg=max_chars,
                prompt_template=prompt_template,
                example_question=example_question,
            )
        print(
            f"\nStopped early ({type(exc).__name__}: {exc})\n"
            f"Saved {len(prediction_rows)} rows to {pred_path}\n"
            f"  new_api={n_api}\n"
            f"Re-run the same --run-id to regenerate from the beginning.\n"
            f"If error is RPD rate limit, wait for the daily window to reset "
            f"or add payment method / raise limits at platform.openai.com."
        )
        raise

    if used_memories:
        mem_log_dir = write_memory_run_log(
            run_dir,
            used_memories,
            max_chars_cfg=max_chars,
            prompt_template=prompt_template,
            example_question=example_question,
        )
        print(f"Memory audit: {mem_log_dir}")

    summary = summarize_predictions(prediction_rows)
    summary["memory_type"] = builder.name
    summary["reader_model"] = reader.model_name
    summary["reader_family"] = resolve_model(reader.model_name).family
    summary["prompt_version"] = prompt_version
    if teacher_model:
        summary["teacher_model"] = teacher_model
        summary["teacher_provider"] = teacher_provider
        summary["teacher_family"] = resolve_model(teacher_model).family

    # Pins for later audit: data file, code commit, prompt, model, memory type.
    meta_out = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "locomo_pin": cfg["data"].get("locomo_commit"),
        "memory_type": builder.name,
        "memory_max_chars": max_chars,
        "reader_provider": reader_name,
        "reader_model": reader.model_name,
        "reader_family": resolve_model(reader.model_name).family,
        "teacher_provider": teacher_provider,
        "teacher_model": teacher_model,
        "teacher_family": resolve_model(teacher_model).family if teacher_model else None,
        "temperature": rate_cfg.get("temperature", 0.0),
        "max_tokens": rate_cfg.get("max_tokens", 64),
        "max_retries": rate_cfg.get("max_retries", 8),
        "min_request_interval_s": rate_cfg.get("min_request_interval_s", 0.0),
        "prompt_path": str(prompt_path),
        "prompt_version": prompt_version,
        "max_questions": max_questions,
        "n_predictions": len(prediction_rows),
        "regenerated_from_scratch": True,
        "n_new_api_calls": n_api,
        "code_git_hash": _git_hash(),
        "package_version": "0.1.4",
        "memory_schema_version": "memory_io.v1",
        "memory_audit_dir": "memory",
    }

    paths = write_run_report(run_dir, prediction_rows, summary, meta_out)
    print("Wrote audit package:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print(
        f"API stats: new_api={n_api} (cache/resume disabled)"
    )
    print("Metrics:", summary["metrics"])
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the LoCoMo QA pipeline for one memory YAML "
            "(not a comparison; run twice then scripts/compare_full_runs.py)"
        )
    )
    p.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="YAML path (default: session_summaries). Also configs/raw_chunks.yaml, configs/session_summaries.yaml",
    )
    p.add_argument("--data", default=None, help="Override path to locomo10.json")
    p.add_argument(
        "--memory",
        default=None,
        help="Override pipeline.memory builder id (raw_chunks, session_summaries, teacher_session_summaries)",
    )
    p.add_argument("--reader", default=None, help="openai | mock")
    p.add_argument("--model", default=None, help="Answer-model id, e.g. gpt-4.1-mini or gpt-5.6-luna")
    p.add_argument("--teacher", default=None, help="Teacher provider: openai | mock (teacher_session_summaries only)")
    p.add_argument(
        "--teacher-model",
        default=None,
        help="Teacher model id (teacher_session_summaries), e.g. gpt-4.1-mini or gpt-5.6-luna",
    )
    p.add_argument("--prompt", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--sample-id", default=None, help="Restrict to one conversation")
    return p


def main(argv: list[str] | None = None) -> None:
    """Argparse wrapper: load one YAML, run one pipeline, exit. No comparison."""
    loaded = load_env()
    if loaded is not None:
        print(f"Loaded env from {loaded}")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_locomo_pipeline_with_memory_config(cfg, args)


if __name__ == "__main__":
    main()
