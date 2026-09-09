"""CLI: offline OpenAI-memory extract dump over LoCoMo (no QA).

    python -m src.locomo_eval.openai_memory.run_index --config configs/openai_memory.yaml --run-id openai_memory_locomo10
    python -m src.locomo_eval.openai_memory.run_index --extractor mock --max-samples 1 --run-id smoke_openai_memory_index
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
from src.locomo_eval.models import resolve_model
from src.locomo_eval.openai_memory.dump import SCHEMA_VERSION, write_index_meta, write_sample_dump
from src.locomo_eval.openai_memory.extract import get_openai_memory_extractor
from src.locomo_eval.rag.chunk import format_conversation_transcript


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


def run_openai_memory_index(cfg: dict, overrides: argparse.Namespace) -> Path:
    data_path = Path(overrides.data or cfg["data"]["raw_path"])
    run_id = overrides.run_id or cfg["run"].get("run_id") or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    index_root = out_root / run_id / "openai_memory_index"
    if index_root.exists():
        shutil.rmtree(index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    ocfg = cfg.get("openai_memory") or {}
    extract_cfg = ocfg.get("extract") or {}
    extractor_name = overrides.extractor or extract_cfg.get("provider") or "openai"
    extract_model = extract_cfg.get("model") or "gpt-4o-mini"
    extractor = get_openai_memory_extractor(
        extractor_name,
        model=extract_model,
        temperature=float(extract_cfg.get("temperature", 0.0)),
        max_tokens=int(extract_cfg.get("max_tokens", 2048)),
        max_retries=int(extract_cfg.get("max_retries", 8)),
        min_request_interval_s=float(extract_cfg.get("min_request_interval_s", 0.5)),
        max_wait_s=float(extract_cfg.get("max_wait_s", 3600.0)),
        prompt_path=extract_cfg.get("prompt_path"),
    )

    conversations = load_conversations(data_path)
    if overrides.sample_id:
        conversations = [c for c in conversations if c.sample_id == overrides.sample_id]
        if not conversations:
            raise SystemExit(f"No sample_id match: {overrides.sample_id}")
    if overrides.max_samples is not None:
        conversations = conversations[: int(overrides.max_samples)]

    sample_rows: list[dict] = []
    print(
        f"OpenAI-memory index {run_id}: {len(conversations)} samples | "
        f"extractor={extractor.provider}/{extractor.model_name}"
    )
    for conv in conversations:
        transcript = format_conversation_transcript(conv)
        memories, _meta = extractor.extract(conv, transcript)
        write_sample_dump(
            index_root,
            sample_id=conv.sample_id,
            transcript=transcript,
            memories=memories,
        )
        sample_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": conv.sample_id,
                "n_memories": len(memories),
                "memories_path": f"by_sample/{conv.sample_id}/memories.json",
            }
        )
        print(f"  wrote {conv.sample_id}: n_memories={len(memories)}")

    run_meta = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "locomo_commit": cfg.get("data", {}).get("locomo_commit"),
        "code_git_hash": _git_hash(),
        "extractor_provider": extractor.provider,
        "extractor_model": extractor.model_name,
        "extractor_family": resolve_model(extractor.model_name).family,
        "max_samples": overrides.max_samples,
        "n_samples": len(sample_rows),
        "gold_answer_in_index": False,
        "retrieve": "all",
        "method": "openai_memory",
        "paper_note": (
            "Architecture clone of the paper's privileged OpenAI-memory protocol. "
            "Not ChatGPT Memory product; not Table 2 J=52.90."
        ),
    }
    write_index_meta(index_root, samples=sample_rows, run_meta=run_meta)
    print(f"Wrote {index_root} (n_samples={len(sample_rows)})")
    return index_root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build an OpenAI-memory extract dump (privileged retrieve-all, no QA)."
    )
    p.add_argument("--config", default="configs/openai_memory.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--sample-id", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--extractor", default=None, help="openai | mock")
    return p


def main(argv: list[str] | None = None) -> None:
    loaded = load_env()
    if loaded is not None:
        print(f"Loaded env from {loaded}")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_openai_memory_index(cfg, args)


if __name__ == "__main__":
    main()
