"""CLI: offline RAG chunk+embed index over LoCoMo (no QA).

    python -m src.locomo_eval.rag.run_index --config configs/rag.yaml --run-id rag_locomo10
    python -m src.locomo_eval.rag.run_index --config configs/rag.yaml --embedder mock --max-samples 1 --run-id smoke_rag_index
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
from src.locomo_eval.dataset import load_conversations
from src.locomo_eval.env import load_env
from src.locomo_eval.mem0.embeddings import get_embedder
from src.locomo_eval.rag.chunk import DEFAULT_ENCODING, chunk_transcript, format_conversation_transcript
from src.locomo_eval.rag.dump import SCHEMA_VERSION, write_index_meta, write_sample_dump


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


def _rag_section(cfg: dict) -> dict:
    return cfg.get("rag") or {}


def run_rag_index(cfg: dict, overrides: argparse.Namespace) -> Path:
    data_path = Path(overrides.data or cfg["data"]["raw_path"])
    run_id = overrides.run_id or cfg["run"].get("run_id") or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    index_root = out_root / run_id / "rag_index"
    if index_root.exists():
        shutil.rmtree(index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    rcfg = _rag_section(cfg)
    embed_cfg = rcfg.get("embed") or {}
    embedder_name = overrides.embedder or embed_cfg.get("provider") or "openai"
    embed_model = embed_cfg.get("model") or "text-embedding-3-small"
    chunk_size = (
        overrides.chunk_size
        if getattr(overrides, "chunk_size", None) is not None
        else int(rcfg.get("chunk_size", 256))
    )
    encoding = str(rcfg.get("encoding") or DEFAULT_ENCODING)
    embedder = get_embedder(embedder_name, model=embed_model)

    conversations = load_conversations(data_path)
    if overrides.sample_id:
        conversations = [c for c in conversations if c.sample_id == overrides.sample_id]
        if not conversations:
            raise SystemExit(f"No sample_id match: {overrides.sample_id}")
    if overrides.max_samples is not None:
        conversations = conversations[: int(overrides.max_samples)]

    sample_rows: list[dict] = []
    print(
        f"RAG index {run_id}: {len(conversations)} samples | "
        f"chunk_size={chunk_size} | embedder={embedder.provider}/{embedder.model_name}"
    )
    for conv in conversations:
        transcript = format_conversation_transcript(conv)
        chunks = chunk_transcript(
            transcript,
            int(chunk_size),
            encoding_name=encoding,
            sample_id=conv.sample_id,
        )
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts) if texts else []
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        write_sample_dump(
            index_root,
            sample_id=conv.sample_id,
            transcript=transcript,
            chunks=chunks,
            chunk_size=int(chunk_size),
            encoding=encoding,
        )
        sample_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": conv.sample_id,
                "n_chunks": len(chunks),
                "n_transcript_chars": len(transcript),
                "n_transcript_tokens": sum(c.n_tokens for c in chunks),
                "chunks_path": f"by_sample/{conv.sample_id}/chunks.json",
            }
        )
        print(f"  wrote {conv.sample_id}: n_chunks={len(chunks)}")

    run_meta = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "locomo_commit": cfg.get("data", {}).get("locomo_commit"),
        "code_git_hash": _git_hash(),
        "chunk_size": int(chunk_size),
        "encoding": encoding,
        "k_default": int(rcfg.get("k", 2)),
        "embedder_provider": embedder.provider,
        "embedder_model": embedder.model_name,
        "max_samples": overrides.max_samples,
        "n_samples": len(sample_rows),
        "gold_answer_in_index": False,
        "method": "rag",
        "paper_note": (
            "Architecture clone of Mem0 evaluation/src/rag.py; "
            "not a Table 2 J claim."
        ),
    }
    write_index_meta(index_root, samples=sample_rows, run_meta=run_meta)
    print(f"Wrote {index_root} (n_samples={len(sample_rows)})")
    return index_root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a RAG chunk+embed dump (no LoCoMo QA)."
    )
    p.add_argument("--config", default="configs/rag.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--sample-id", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--embedder", default=None, help="openai | mock")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Token window (paper: 128–8192). Negative = full transcript as one chunk.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    loaded = load_env()
    if loaded is not None:
        print(f"Loaded env from {loaded}")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_rag_index(cfg, args)


if __name__ == "__main__":
    main()
