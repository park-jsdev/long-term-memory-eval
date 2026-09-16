"""CLI: offline Mem0 / Mem0g write-index over LoCoMo (no QA).

    python -m src.locomo_eval.mem0.run_index --config configs/writers/mem0.yaml --run-id mem0_locomo10
    python -m src.locomo_eval.mem0.run_index --config configs/writers/mem0g.yaml --run-id mem0g_locomo10
    python -m src.locomo_eval.mem0.run_index --config configs/writers/mem0.yaml --extractor mock --embedder mock --max-samples 1 --run-id smoke_mem0_index
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
from src.locomo_eval.env import load_env
from src.locomo_eval.mem0.dump import write_index_meta, write_sample_dump
from src.locomo_eval.mem0.embeddings import get_embedder
from src.locomo_eval.mem0.extract import get_fact_extractor
from src.locomo_eval.mem0.graph_memory import Mem0GraphMemory, openai_graph_callables
from src.locomo_eval.mem0.indexer import Mem0Indexer
from src.locomo_eval.mem0.schemas import SCHEMA_VERSION
from src.locomo_eval.mem0.update import get_memory_updater
from src.locomo_eval.models import resolve_model
from src.locomo_eval.preprocess import DataIngestor, PreprocessingPipeline


def _git_hash() -> str | None:
    """Pin HEAD in run_meta so you can check out this commit and reproduce the index."""
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
    """Fingerprint locomo10.json so later eval can tell if the index used different data."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mem0_section(cfg: dict) -> dict:
    return cfg.get("mem0") or {}


def build_indexer(cfg: dict, overrides: argparse.Namespace) -> tuple[Mem0Indexer, dict]:
    mcfg = _mem0_section(cfg)
    extract_cfg = mcfg.get("extract") or {}
    embed_cfg = mcfg.get("embed") or {}
    extractor_name = overrides.extractor or extract_cfg.get("provider") or "openai"
    embedder_name = overrides.embedder or embed_cfg.get("provider") or "openai"
    extract_model = extract_cfg.get("model") or "gpt-4o-mini"
    enable_graph = bool(mcfg.get("enable_graph", False))
    if getattr(overrides, "enable_graph", None) is not None:
        enable_graph = bool(overrides.enable_graph)

    extractor = get_fact_extractor(
        extractor_name,
        model=extract_model,
        temperature=float(extract_cfg.get("temperature", 0.0)),
        max_tokens=int(extract_cfg.get("max_tokens", 512)),
        max_retries=int(extract_cfg.get("max_retries", 8)),
        min_request_interval_s=float(extract_cfg.get("min_request_interval_s", 0.5)),
        max_wait_s=float(extract_cfg.get("max_wait_s", 3600.0)),
        prompt_path=extract_cfg.get("prompt_path"),
    )
    updater = get_memory_updater(
        extractor_name,
        model=extract_model,
        temperature=float(extract_cfg.get("temperature", 0.0)),
        max_tokens=int(extract_cfg.get("max_tokens", 512)),
        max_retries=int(extract_cfg.get("max_retries", 8)),
        min_request_interval_s=float(extract_cfg.get("min_request_interval_s", 0.5)),
        max_wait_s=float(extract_cfg.get("max_wait_s", 3600.0)),
        prompt_path=(mcfg.get("update") or {}).get("prompt_path"),
    )
    embedder = get_embedder(
        embedder_name,
        model=embed_cfg.get("model") or "text-embedding-3-small",
    )
    graph = None
    if enable_graph:
        entity_fn = relation_fn = conflict_fn = None
        if extractor_name == "openai":
            entity_fn, relation_fn, conflict_fn = openai_graph_callables(
                model=extract_model,
                temperature=float(extract_cfg.get("temperature", 0.0)),
                max_tokens=int(extract_cfg.get("max_tokens", 512)),
                max_retries=int(extract_cfg.get("max_retries", 8)),
                min_request_interval_s=float(
                    extract_cfg.get("min_request_interval_s", 0.5)
                ),
                max_wait_s=float(extract_cfg.get("max_wait_s", 3600.0)),
            )
        graph = Mem0GraphMemory(
            embedder,
            threshold=float(mcfg.get("graph_threshold", 0.7)),
            entity_extractor=entity_fn,
            relation_extractor=relation_fn,
            conflict_resolver=conflict_fn,
        )
    indexer = Mem0Indexer(
        extractor=extractor,
        updater=updater,
        embedder=embedder,
        graph=graph,
        batch_size=int(mcfg.get("batch_size", 2)),
        similar_s=int(mcfg.get("similar_s", 10)),
        max_sessions=overrides.max_sessions,
    )
    meta = {
        "enable_graph": enable_graph,
        "extractor_provider": extractor.provider,
        "extractor_model": extractor.model_name,
        "updater_provider": updater.provider,
        "updater_model": updater.model_name,
        "embedder_provider": embedder.provider,
        "embedder_model": embedder.model_name,
        "extractor_family": resolve_model(extractor.model_name).family,
    }
    return indexer, meta


def run_mem0_index(cfg: dict, overrides: argparse.Namespace) -> Path:
    data_path = Path(overrides.data or cfg["data"]["raw_path"])
    run_id = overrides.run_id or cfg["run"].get("run_id") or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    index_root = out_root / run_id / "mem0_index"
    if index_root.exists():
        shutil.rmtree(index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    enable_graph = bool((_mem0_section(cfg)).get("enable_graph", False))
    if getattr(overrides, "enable_graph", None) is not None:
        enable_graph = bool(overrides.enable_graph)

    indexer, component_meta = build_indexer(cfg, overrides)
    conversations = DataIngestor().load(data_path)
    if overrides.sample_id:
        conversations = [c for c in conversations if c.sample_id == overrides.sample_id]
        if not conversations:
            raise SystemExit(f"No sample_id match: {overrides.sample_id}")
    if overrides.max_samples is not None:
        conversations = conversations[: int(overrides.max_samples)]

    pipeline = PreprocessingPipeline()
    sample_rows: list[dict] = []
    print(
        f"Mem0 index {run_id}: {len(conversations)} samples | "
        f"graph={enable_graph} | extractor={component_meta['extractor_provider']}/"
        f"{component_meta['extractor_model']}"
    )
    for conv in conversations:
        processed = pipeline.process(conv)
        # Graph is per-sample: replace so samples do not share nodes.
        if enable_graph:
            prev = indexer.graph
            indexer.graph = Mem0GraphMemory(
                indexer.embedder,
                threshold=getattr(prev, "threshold", 0.7) if prev else 0.7,
                entity_extractor=getattr(prev, "entity_extractor", None) if prev else None,
                relation_extractor=getattr(prev, "relation_extractor", None)
                if prev
                else None,
                conflict_resolver=getattr(prev, "conflict_resolver", None)
                if prev
                else None,
            )
        store_a, store_b, log = indexer.index_conversation(processed)
        write_sample_dump(
            index_root,
            sample_id=processed.sample_id,
            store_a=store_a,
            store_b=store_b,
            ingest_log=log,
            graph=indexer.graph if enable_graph else None,
        )
        n_edges = 0
        n_nodes = 0
        if enable_graph and indexer.graph is not None:
            dumped = indexer.graph.to_dict()
            n_nodes = len(dumped.get("nodes") or [])
            n_edges = len(dumped.get("edges") or [])
        sample_rows.append(
            {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": processed.sample_id,
                    "n_session_blocks": len(processed.session_blocks),
                "n_facts_a": len(store_a.facts),
                "n_facts_b": len(store_b.facts),
                "n_graph_nodes": n_nodes,
                "n_graph_edges": n_edges,
                "n_ingest_rows": len(log),
                "speaker_a_path": f"by_sample/{processed.sample_id}/speaker_a.json",
                "speaker_b_path": f"by_sample/{processed.sample_id}/speaker_b.json",
                "graph_path": (
                    f"by_sample/{processed.sample_id}/graph.json"
                    if enable_graph
                    else None
                ),
            }
        )
        print(
            f"  wrote {processed.sample_id}: facts_a={len(store_a.facts)} "
            f"facts_b={len(store_b.facts)} graph_edges={n_edges}"
        )

    mcfg = _mem0_section(cfg)
    run_meta = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "locomo_commit": cfg.get("data", {}).get("locomo_commit"),
        "code_git_hash": _git_hash(),
        "enable_graph": enable_graph,
        "batch_size": int(mcfg.get("batch_size", 2)),
        "similar_s": int(mcfg.get("similar_s", 10)),
        "top_k": int(mcfg.get("top_k", 30)),
        "graph_threshold": float(mcfg.get("graph_threshold", 0.7)),
        "max_samples": overrides.max_samples,
        "max_sessions": overrides.max_sessions,
        "n_samples": len(sample_rows),
        "gold_answer_in_index": False,
        **component_meta,
    }
    write_index_meta(index_root, samples=sample_rows, run_meta=run_meta)
    print(f"Wrote {index_root} (n_samples={len(sample_rows)})")
    return index_root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a Mem0 / Mem0g write-index dump (no LoCoMo QA)."
    )
    p.add_argument("--config", default="configs/writers/mem0.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--sample-id", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--extractor", default=None, help="openai | mock")
    p.add_argument("--embedder", default=None, help="openai | mock")
    p.add_argument(
        "--enable-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override mem0.enable_graph",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    loaded = load_env()
    if loaded is not None:
        print(f"Loaded env from {loaded}")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_mem0_index(cfg, args)


if __name__ == "__main__":
    main()
