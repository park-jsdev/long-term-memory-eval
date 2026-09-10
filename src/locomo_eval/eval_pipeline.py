"""Evaluation pipeline: index (if needed) → QA → string metrics → autorater.

One experimental method per invocation. Compare packs afterwards with
``scripts/compare_full_runs.py`` (sandwich / representation) or
``scripts.analysis.aggregate_seeds`` (multi-judge CIs).

    python -m src.locomo_eval.eval_pipeline --method rag --reader mock --embedder mock --max-samples 1 --max-questions 5 --run-id smoke_eval_rag
    python -m src.locomo_eval.eval_pipeline --method full_context --max-questions 3 --run-id smoke_full_context
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.locomo_eval.env import load_env
from src.locomo_eval.memory import resolve_memory_name
from src.locomo_eval.run import run_locomo_pipeline_with_memory_config

METHOD_CONFIGS: dict[str, str] = {
    "rag": "configs/rag.yaml",
    "full_context": "configs/full_context.yaml",
    "openai_memory": "configs/openai_memory.yaml",
    "mem0": "configs/mem0.yaml",
    "mem0g": "configs/mem0g.yaml",
    "raw_chunks": "configs/raw_chunks.yaml",
    "session_summaries": "configs/session_summaries.yaml",
    "teacher_session_summaries": "configs/teacher_session_summaries.yaml",
    "mem0_baseline": "configs/mem0_baseline.yaml",
}

INDEX_METHODS = {
    "rag": ("rag_index", "python -m src.locomo_eval.rag.run_index"),
    "mem0": ("mem0_index", "python -m src.locomo_eval.mem0.run_index"),
    "mem0g": ("mem0_index", "python -m src.locomo_eval.mem0.run_index"),
    "openai_memory": (
        "openai_memory_index",
        "python -m src.locomo_eval.openai_memory.run_index",
    ),
}


def yaml_index_run_id(resolved: str, cfg: dict) -> str | None:
    if resolved == "rag":
        return (cfg.get("rag") or {}).get("index_run_id")
    if resolved == "openai_memory":
        return (cfg.get("openai_memory") or {}).get("index_run_id")
    if resolved in ("mem0", "mem0g"):
        return (cfg.get("mem0") or {}).get("index_run_id")
    return None


def resolve_eval_index_run_id(
    resolved: str, cfg: dict, args: argparse.Namespace
) -> str:
    """Dump id for this eval invocation.

    A subset smoke (``--max-samples`` / ``--sample-id``) must not overwrite the
    YAML full-dump id (e.g. ``rag_locomo10``) unless the caller passed
    ``--index-run-id`` explicitly.
    """
    if getattr(args, "index_run_id", None):
        return str(args.index_run_id)
    subset = (
        getattr(args, "max_samples", None) is not None
        or getattr(args, "sample_id", None)
    )
    if subset:
        return f"{getattr(args, 'run_id', None) or 'eval'}_index"
    yaml_id = yaml_index_run_id(resolved, cfg)
    return str(yaml_id or f"{getattr(args, 'run_id', None) or 'eval'}_index")


def default_config_for_method(method: str) -> str:
    resolved = resolve_memory_name(method) if method not in METHOD_CONFIGS else method
    if method == "mem0_baseline":
        return METHOD_CONFIGS["mem0_baseline"]
    if resolved in METHOD_CONFIGS:
        return METHOD_CONFIGS[resolved]
    if method in METHOD_CONFIGS:
        return METHOD_CONFIGS[method]
    raise SystemExit(
        f"Unknown method '{method}'. Known: {sorted(METHOD_CONFIGS)}"
    )


def _index_exists(out_root: Path, index_run_id: str, dump_name: str) -> bool:
    meta = out_root / index_run_id / dump_name / "run_meta.json"
    return meta.is_file()


def _run_index_if_needed(
    method: str,
    cfg: dict,
    args: argparse.Namespace,
    out_root: Path,
) -> str | None:
    resolved = resolve_memory_name(method) if method != "mem0_baseline" else method
    if resolved not in INDEX_METHODS:
        return None
    dump_name, _hint = INDEX_METHODS[resolved]
    index_run = resolve_eval_index_run_id(resolved, cfg, args)
    if args.skip_index:
        return str(index_run)
    if _index_exists(out_root, str(index_run), dump_name) and not args.force_index:
        print(f"Reusing existing {dump_name} at experiments/{index_run}/{dump_name}")
        return str(index_run)

    index_ns = argparse.Namespace(
        data=args.data,
        output_dir=str(out_root),
        run_id=str(index_run),
        sample_id=args.sample_id,
        max_samples=args.max_samples,
        extractor=args.extractor,
        embedder=args.embedder,
        enable_graph=True if resolved == "mem0g" else (False if resolved == "mem0" else None),
        chunk_size=args.chunk_size,
        max_sessions=None,
    )
    print(f"Indexing {resolved} → experiments/{index_run}/{dump_name}")
    if resolved == "rag":
        from src.locomo_eval.rag.run_index import run_rag_index

        run_rag_index(cfg, index_ns)
    elif resolved == "openai_memory":
        from src.locomo_eval.openai_memory.run_index import run_openai_memory_index

        run_openai_memory_index(cfg, index_ns)
    else:
        from src.locomo_eval.mem0.run_index import run_mem0_index

        run_mem0_index(cfg, index_ns)
    return str(index_run)


def _qa_overrides(args: argparse.Namespace, index_run_id: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        data=args.data,
        memory=args.memory,
        reader=args.reader,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        message_layout=args.message_layout,
        teacher=args.teacher,
        teacher_model=args.teacher_model,
        prompt=args.prompt,
        output_dir=args.output_dir,
        run_id=args.run_id,
        max_questions=args.max_questions,
        question_sample=args.question_sample,
        sample_id=args.sample_id,
        max_samples=args.max_samples,
        mem0_index_run_id=index_run_id if args.method in ("mem0", "mem0g") else None,
        rag_index_run_id=index_run_id if args.method == "rag" else None,
        rag_k=args.k,
        openai_memory_index_run_id=(
            index_run_id if args.method == "openai_memory" else None
        ),
        preprocess_index_run_id=None,
        retrieve_top_k=None,
        config=getattr(args, "config", None),
        pool=getattr(args, "pool", None),
        fusion=getattr(args, "fusion", None),
        thinking=getattr(args, "thinking", None),
    )


def _run_autorater_seeds(
    run_dir: Path,
    *,
    autorater: str,
    n_judge_runs: int,
    max_questions: int | None,
    label: str | None,
) -> dict[str, Any]:
    from scripts.analysis.aggregate_seeds import aggregate_autorater_seeds
    from scripts.analysis.run_benchmark import run_benchmark
    from src.locomo_eval.autorater import get_autorater
    from src.config import load_config as _load

    acfg = {}
    cfg_path = ROOT / "configs" / "autorater.yaml"
    if cfg_path.is_file():
        acfg = (_load(cfg_path).get("autorater") or {})
    provider = autorater or acfg.get("provider") or "openai"
    rater_factory = lambda: get_autorater(
        provider,
        model=acfg.get("model") or "gpt-4o-mini",
        temperature=float(acfg.get("temperature", 0.0)),
        max_tokens=acfg.get("max_tokens"),
        max_retries=int(acfg.get("max_retries", 8)),
        min_request_interval_s=float(acfg.get("min_request_interval_s", 0.5)),
        max_wait_s=float(acfg.get("max_wait_s", 3600.0)),
        prompt_path=acfg.get("prompt_path"),
    )
    pred = run_dir / "predictions.jsonl"
    if n_judge_runs <= 1:
        result = run_benchmark(
            pred,
            out_dir=run_dir / "autorater",
            autorater=rater_factory(),
            max_questions=max_questions,
            our_label=label,
            prompt_version="autorater_mem0_v1",
        )
        return {"n_judge_runs": 1, "autorater": result["out_dir"]}

    seed_dirs: list[Path] = []
    for i in range(int(n_judge_runs)):
        seed_dir = run_dir / "autorater_seeds" / f"seed_{i:02d}"
        print(f"Judge seed {i + 1}/{n_judge_runs} → {seed_dir}")
        run_benchmark(
            pred,
            out_dir=seed_dir,
            autorater=rater_factory(),
            max_questions=max_questions,
            our_label=label,
            prompt_version="autorater_mem0_v1",
        )
        seed_dirs.append(seed_dir)
    agg = aggregate_autorater_seeds(seed_dirs, out_dir=run_dir / "autorater")
    return {"n_judge_runs": n_judge_runs, "autorater": str(run_dir / "autorater"), **agg}


def run_eval_pipeline(args: argparse.Namespace) -> Path:
    method = str(args.method).strip().lower().replace("-", "_")
    if method in ("fullcontext", "full_context"):
        method = "full_context"
    if method == "openai":
        method = "openai_memory"
    args.method = method
    cfg_path = args.config or default_config_for_method(method)
    args.config = cfg_path
    cfg = load_config(cfg_path if Path(cfg_path).is_file() else ROOT / cfg_path)
    if args.memory is None and method not in ("mem0_baseline",):
        try:
            args.memory = resolve_memory_name(method)
        except ValueError:
            pass
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.run_id = run_id
    out_root = Path(args.output_dir or cfg["run"]["output_dir"])
    args.output_dir = str(out_root)

    index_run_id = _run_index_if_needed(method, cfg, args, out_root)
    if args.skip_qa:
        print(f"Skip QA (--skip-qa). Index run id: {index_run_id}")
        return out_root / (index_run_id or run_id)

    qa_args = _qa_overrides(args, index_run_id)
    run_dir = run_locomo_pipeline_with_memory_config(cfg, qa_args)

    if args.offline_evaluate:
        from src.locomo_eval.offline_evaluate import main as offline_main

        offline_main(
            [
                "--predictions",
                str(run_dir / "predictions.jsonl"),
                "--output-dir",
                str(run_dir),
            ]
        )

    if args.autorater:
        _run_autorater_seeds(
            run_dir,
            autorater=args.autorater,
            n_judge_runs=int(args.n_judge_runs),
            max_questions=args.max_questions,
            label=args.label or method,
        )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Index (if needed) + QA + optional autorater for one memory method. "
            "Not a comparison; run twice then scripts/compare_full_runs.py."
        )
    )
    p.add_argument(
        "--method",
        required=True,
        help="rag | full_context | openai_memory | mem0 | mem0g | raw_chunks | "
        "session_summaries | teacher_session_summaries | mem0_baseline | custom YAML via --config",
    )
    p.add_argument("--config", default=None, help="Override YAML (default: configs/<method>.yaml)")
    p.add_argument("--data", default=None)
    p.add_argument("--memory", default=None, help="Override pipeline.memory")
    p.add_argument("--reader", default=None, help="openai | deepseek | anthropic | mock")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--message-layout", default=None)
    p.add_argument("--teacher", default=None)
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--question-sample", default=None, choices=("round_robin", "prefix"))
    p.add_argument("--sample-id", default=None)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap conversations for index and QA (file order). "
        "Round-robin --max-questions then stays inside those samples. "
        "Subset smokes write experiments/<run_id>_index, not the YAML full-dump id.",
    )
    p.add_argument("--index-run-id", default=None, help="Reuse/write index under this run id")
    p.add_argument("--skip-index", action="store_true")
    p.add_argument("--force-index", action="store_true")
    p.add_argument("--skip-qa", action="store_true")
    p.add_argument("--extractor", default=None, help="Index extractor: openai | mock")
    p.add_argument("--embedder", default=None, help="Index embedder: openai | mock")
    p.add_argument("--chunk-size", type=int, default=None, help="RAG token window")
    p.add_argument("--k", type=int, default=None, help="RAG top-k (paper: 1 or 2)")
    p.add_argument(
        "--autorater",
        default=None,
        help="openai | mock. Omit to skip the judge step.",
    )
    p.add_argument(
        "--n-judge-runs",
        type=int,
        default=1,
        help="Independent autorater runs on the same predictions (paper: 10)",
    )
    p.add_argument(
        "--offline-evaluate",
        action="store_true",
        help="Re-run string metrics after QA (already written by run.py; optional extra)",
    )
    p.add_argument("--label", default=None, help="vs-literature row name")
    return p


def main(argv: list[str] | None = None) -> None:
    loaded = load_env()
    if loaded is not None:
        print(f"Loaded env from {loaded}")
    args = build_parser().parse_args(argv)
    run_dir = run_eval_pipeline(args)
    print(f"Eval pipeline wrote {run_dir}")


if __name__ == "__main__":
    main()
