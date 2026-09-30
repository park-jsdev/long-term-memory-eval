"""CLI: deterministic preprocess write-index over LoCoMo (no LLM, no QA).

Index the full dataset:

    python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess

Then 10 reader LLM calls per sanity memory (same question subset, dump-backed):

    python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess --eval-questions 10

Mock (no API):

    python -m src.locomo_eval.preprocess.run_index --eval-questions 10 --eval-reader mock --run-id smoke_preprocess
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.locomo_eval.dataset import load_raw
from src.locomo_eval.env import load_env
from src.locomo_eval.preprocess import DataIngestor, PreprocessingPipeline
from src.locomo_eval.preprocess.dump import (
    SCHEMA_VERSION,
    write_index_meta,
    write_sample_dump,
)
from src.locomo_eval.preprocess.session_documents import build_session_documents


DEFAULT_EVAL_CONFIGS = (
    "configs/writers/raw_chunks.yaml",
    "configs/writers/session_summaries.yaml",
)


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


def run_preprocess_index(cfg: dict, overrides: argparse.Namespace) -> Path:
    data_path = Path(overrides.data or cfg["data"]["raw_path"])
    if not data_path.is_file():
        raise SystemExit(f"Missing {data_path}. Run python scripts/fetch_locomo.py")
    run_id = overrides.run_id or cfg["run"].get("run_id") or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    index_root = out_root / str(run_id) / "preprocess"
    if index_root.exists():
        shutil.rmtree(index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    samples = load_raw(data_path)
    if getattr(overrides, "sample_id", None):
        samples = [s for s in samples if str(s.get("sample_id")) == overrides.sample_id]
        if not samples:
            raise SystemExit(f"No sample_id match: {overrides.sample_id}")
    if getattr(overrides, "max_samples", None) is not None:
        samples = samples[: int(overrides.max_samples)]

    ingestor = DataIngestor()
    pipeline = PreprocessingPipeline()
    sample_rows: list[dict] = []
    print(f"Preprocess index {run_id}: {len(samples)} samples | llm_calls=0")
    for sample in samples:
        conv = ingestor.ingest_sample(sample)
        processed = pipeline.process(conv)
        documents = build_session_documents(sample)
        write_sample_dump(
            index_root, processed=processed, documents=documents
        )
        n_turns = sum(len(b.turns) for b in processed.session_blocks)
        sample_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": processed.sample_id,
                "n_session_blocks": len(processed.session_blocks),
                "n_documents": len(documents),
                "n_turns": n_turns,
                "question_ids": list(processed.question_ids),
                "sessions_path": f"by_sample/{processed.sample_id}/sessions.jsonl",
                "documents_path": f"by_sample/{processed.sample_id}/documents.jsonl",
            }
        )
        print(
            f"  wrote {processed.sample_id}: sessions={len(processed.session_blocks)} "
            f"turns={n_turns} documents={len(documents)}"
        )

    run_meta = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "locomo_commit": cfg.get("data", {}).get("locomo_commit"),
        "code_git_hash": _git_hash(),
        "max_samples": getattr(overrides, "max_samples", None),
        "n_samples": len(sample_rows),
        "gold_answer_in_index": False,
        "llm_calls": 0,
    }
    write_index_meta(index_root, samples=sample_rows, run_meta=run_meta)
    print(f"Wrote {index_root} (n_samples={len(sample_rows)})")
    return index_root


def _eval_after_index(
    cfg: dict, overrides: argparse.Namespace, index_run_id: str
) -> None:
    """Same frozen reader; dump-backed raw_chunks / session_summaries; N questions."""
    from src.locomo_eval.run import run_locomo_pipeline_with_memory_config

    n_q = int(overrides.eval_questions)
    eval_reader = getattr(overrides, "eval_reader", None) or "openai"
    configs = list(getattr(overrides, "eval_configs", None) or DEFAULT_EVAL_CONFIGS)
    data_path = Path(overrides.data or cfg["data"]["raw_path"])
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    print(
        f"Eval after index: {n_q} questions × {len(configs)} memories | "
        f"reader={eval_reader} | preprocess_index={index_run_id}"
    )
    for rel in configs:
        eval_cfg = load_config(ROOT / rel if not Path(rel).is_file() else rel)
        memory_name = eval_cfg["pipeline"]["memory"]
        eval_run_id = f"{index_run_id}_{memory_name}_n{n_q}"
        ns = argparse.Namespace(
            data=str(data_path),
            memory=None,
            reader=eval_reader,
            model=None,
            writer=None,
            writer_model=None,
            prompt=None,
            output_dir=str(out_root),
            run_id=eval_run_id,
            max_questions=n_q,
            sample_id=None,
            mem0_index_run_id=None,
            preprocess_index_run_id=index_run_id,
            retrieve_top_k=None,
            temperature=None,
            max_tokens=None,
            message_layout=None,
            question_sample=getattr(overrides, "question_sample", None) or "round_robin",
        )
        run_dir = run_locomo_pipeline_with_memory_config(eval_cfg, ns)
        print(f"  eval {memory_name}: {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Deterministic preprocess write-index (no LLM). "
            "Optional --eval-questions runs dump-backed sanity memories."
        )
    )
    p.add_argument("--config", default="configs/writers/preprocess.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--sample-id", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--eval-questions",
        type=int,
        default=None,
        help="After indexing, run this many reader LLM calls per sanity memory YAML "
        "(round-robin across conversations by default).",
    )
    p.add_argument(
        "--question-sample",
        default="round_robin",
        choices=("round_robin", "prefix"),
        help="How --eval-questions picks items (default: round_robin)",
    )
    p.add_argument(
        "--eval-reader",
        default=None,
        help="openai | mock (default openai when --eval-questions is set)",
    )
    p.add_argument(
        "--eval-configs",
        nargs="*",
        default=None,
        help="Memory YAMLs to eval (default: raw_chunks + session_summaries)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    loaded = load_env()
    if loaded is not None:
        print(f"Loaded env from {loaded}")
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    index_root = run_preprocess_index(cfg, args)
    if args.eval_questions is not None:
        run_id = args.run_id or cfg["run"].get("run_id") or index_root.parent.name
        _eval_after_index(cfg, args, str(run_id))


if __name__ == "__main__":
    main()
