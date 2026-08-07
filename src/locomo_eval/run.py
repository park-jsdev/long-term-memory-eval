"""End-to-end Phase 1 baseline: load → memory → reader → report."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root on sys.path so `src.*` imports work when run as a module or script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.locomo_eval.cache import ResponseCache
from src.locomo_eval.dataset import iter_questions, load_conversations
from src.locomo_eval.env import load_env
from src.locomo_eval.memory import get_memory_builder
from src.locomo_eval.metrics import summarize_predictions
from src.locomo_eval.prompts import load_prompt_template
from src.locomo_eval.readers import get_reader
from src.locomo_eval.report import write_run_report
from src.locomo_eval.schemas import Prediction


def _git_hash() -> str | None:
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
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_baseline(cfg: dict, overrides: argparse.Namespace) -> Path:
    data_path = Path(overrides.data or cfg["data"]["raw_path"])
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

    prompt_version, prompt_template = load_prompt_template(prompt_path)
    builder = get_memory_builder(memory_name)
    cache = ResponseCache(cfg["run"].get("cache_dir", "experiments/cache"))
    reader_model = "mock" if reader_name.lower() == "mock" else model
    reader = get_reader(
        reader_name,
        model=reader_model,
        temperature=float(cfg["reader"].get("temperature", 0.0)),
        max_tokens=int(cfg["reader"].get("max_tokens", 64)),
        cache=cache,
    )

    conversations = load_conversations(data_path)
    if overrides.sample_id:
        conversations = [c for c in conversations if c.sample_id == overrides.sample_id]
        if not conversations:
            raise SystemExit(f"No sample_id match: {overrides.sample_id}")

    # Precompute memory once per conversation for session_summary (no query dependence).
    memory_by_sample = {}
    if memory_name == "session_summary":
        for conv in conversations:
            # question unused for this builder; pass first/placeholder if needed
            q = conv.questions[0] if conv.questions else None
            if q is None:
                continue
            memory_by_sample[conv.sample_id] = builder.build(conv, q)

    pairs = list(iter_questions(conversations))
    if max_questions is not None:
        pairs = pairs[: int(max_questions)]

    print(
        f"Run {run_id}: {len(pairs)} questions | memory={memory_name} | "
        f"reader={reader_name}/{reader.model_name}"
    )

    prediction_rows: list[dict] = []
    for i, (conv, q) in enumerate(pairs, start=1):
        if conv.sample_id in memory_by_sample:
            memory = memory_by_sample[conv.sample_id]
        else:
            memory = builder.build(conv, q)

        answer, meta = reader.answer(memory.text, q.question, prompt_template)
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
            cached=bool(meta.get("cached")),
        )
        row = pred.to_dict()
        row["latency_s"] = meta.get("latency_s")
        row["usage"] = meta.get("usage")
        prediction_rows.append(row)

        if i % 10 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}] {q.question_id} cached={meta.get('cached')}")

    summary = summarize_predictions(prediction_rows)
    summary["memory_type"] = memory_name
    summary["reader_model"] = reader.model_name
    summary["prompt_version"] = prompt_version

    meta_out = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "locomo_pin": cfg["data"].get("locomo_commit"),
        "memory_type": memory_name,
        "reader_provider": reader_name,
        "reader_model": reader.model_name,
        "temperature": cfg["reader"].get("temperature", 0.0),
        "max_tokens": cfg["reader"].get("max_tokens", 64),
        "prompt_path": str(prompt_path),
        "prompt_version": prompt_version,
        "max_questions": max_questions,
        "n_predictions": len(prediction_rows),
        "code_git_hash": _git_hash(),
        "package_version": "0.1.0",
    }

    paths = write_run_report(run_dir, prediction_rows, summary, meta_out)
    print("Wrote audit package:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("Metrics:", summary["metrics"])
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1 LoCoMo session-summary baseline")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--data", default=None, help="Override path to locomo10.json")
    p.add_argument("--memory", default=None, help="session_summary")
    p.add_argument("--reader", default=None, help="openai | mock")
    p.add_argument("--model", default=None, help="Model name, e.g. gpt-4.1-mini")
    p.add_argument("--prompt", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--sample-id", default=None, help="Restrict to one conversation")
    return p


def main(argv: list[str] | None = None) -> None:
    loaded = load_env()
    if loaded is not None:
        print(f"Loaded env from {loaded}")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_baseline(cfg, args)


if __name__ == "__main__":
    main()
