"""CLI: run the LoCoMo QA pipeline for **one** memory config.

This module does not compare conditions and is not a multi-run orchestrator.
Typical sandwich experiment:

1. Call this once with config A (e.g. ``configs/writers/raw_chunks.yaml``) → ``experiments/<run_id_A>/``
2. Call this once with config B (e.g. ``configs/writers/session_summaries.yaml``) → ``experiments/<run_id_B>/``
3. Compare the two audit packs offline:
   ``python scripts/compare_full_runs.py --runs experiments/<run_id_A> experiments/<run_id_B> ...``
   (memory axis) or ``scripts/compare_cross_model.py`` (reader/teacher model axis)

Each ``--config`` YAML is standalone. ``pipeline.memory`` is a builder id in
memory.py (``raw_chunks`` / ``session_summaries``), not a path to another YAML.

    python -m src.locomo_eval.run --config configs/writers/raw_chunks.yaml --run-id cmp_raw_chunks
    python -m src.locomo_eval.run --config configs/writers/session_summaries.yaml --run-id cmp_session_summaries
    python scripts/compare_full_runs.py --runs experiments/cmp_raw_chunks experiments/cmp_session_summaries --out experiments/compare_raw_chunks_session_summaries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root on sys.path so `src.*` imports work when run as a module or script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.locomo_eval.dataset import iter_questions, load_conversations, select_question_pairs
from src.locomo_eval.env import load_env
from src.locomo_eval.fusion import (
    FUSION_MAJORITY,
    FUSION_NONE,
    FUSION_POLICIES,
    POOL_EQUAL_WEIGHT,
    POOL_SINGLE,
)
from src.locomo_eval.memory import (
    AgentFactMemoryBuilder,
    ORCHESTRATED_GRAPH_NAMES,
    TeacherSessionMemoryBuilder,
    get_memory_builder,
    is_question_independent,
    resolve_memory_name,
)
from src.locomo_eval.agents import get_agent_runner, render_agent_prompt
from src.locomo_eval.agents.comparison import (
    comparison_status,
    contract_sha256,
    resolve_contract,
    validate_strict_contract,
    workspace_manifest_sha256,
)
from src.locomo_eval.agents.dump import write_agent_module
from src.locomo_eval.agents.metrics import score_agent_row, summarize_agent_rows
from src.locomo_eval.agents.protocol import AgentRequest
from src.locomo_eval.metrics import score_row, summarize_predictions
from src.locomo_eval.experiments.audit_writer import (
    write_claim_audit,
    write_frozen_config,
    write_graph_module,
    write_reader_module,
    write_teacher_module,
)
from src.locomo_eval.experiments.prompt_bundle import repo_rel, write_prompt_bundle
from src.locomo_eval.memory_log import (
    collect_fusion_log,
    collect_ingest_log,
    collect_retrieve_log,
    collect_session_texts,
    collect_teacher_call_log,
    write_memory_run_log,
)
from src.locomo_eval.experiments.audit_layout import audit_layout_meta
from src.locomo_eval.models import resolve_model
from src.locomo_eval.prompts import load_prompt_template, locate_prompt_file
from src.locomo_eval.readers import get_reader
from src.locomo_eval.report import write_jsonl, write_run_report
from src.locomo_eval.schemas import Memory, Prediction
from src.locomo_eval.teacher_callers import DEFAULT_TEACHER_THINKING
from src.locomo_eval.teacher_orchestrator import TeacherOrchestrator
from src.locomo_eval.teachers import get_teacher


_ORCH_DEFAULTS = {
    "teacher_graph": (POOL_SINGLE, FUSION_NONE),
    "pooled_teacher_graph": (POOL_EQUAL_WEIGHT, FUSION_NONE),
    "fused_teacher_graph": (POOL_EQUAL_WEIGHT, FUSION_MAJORITY),
}


def _mem0_builder_kwargs(
    cfg: dict, overrides: argparse.Namespace, memory_name: str, reader_name: str
) -> dict:
    """Index-dir + query embedder for mem0/mem0g. Other builders ignore these kwargs."""
    if resolve_memory_name(memory_name) not in ("mem0", "mem0g"):
        return {}
    mcfg = cfg.get("mem0") or {}
    index_run = getattr(overrides, "mem0_index_run_id", None) or mcfg.get("index_run_id")
    if not index_run:
        raise SystemExit(
            "mem0/mem0g requires mem0.index_run_id in YAML or --mem0-index-run-id "
            "(after python -m src.locomo_eval.mem0.run_index)."
        )
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    embed_cfg = mcfg.get("embed") or {}
    provider = embed_cfg.get("provider") or "openai"
    if str(reader_name).lower() == "mock":
        provider = "mock"
    from src.locomo_eval.mem0.embeddings import get_embedder

    return {
        "mem0_index_dir": out_root / str(index_run) / "mem0_index",
        "mem0_top_k": int(mcfg.get("top_k", 30)),
        "mem0_embedder": get_embedder(
            provider,
            model=embed_cfg.get("model") or "text-embedding-3-small",
        ),
    }


def _rag_builder_kwargs(
    cfg: dict, overrides: argparse.Namespace, memory_name: str, reader_name: str
) -> dict:
    """Index-dir + query embedder for rag. Other builders ignore these kwargs."""
    if resolve_memory_name(memory_name) != "rag":
        return {}
    rcfg = cfg.get("rag") or {}
    index_run = getattr(overrides, "rag_index_run_id", None) or rcfg.get("index_run_id")
    if not index_run:
        raise SystemExit(
            "rag requires rag.index_run_id in YAML or --rag-index-run-id "
            "(after python -m src.locomo_eval.rag.run_index)."
        )
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    embed_cfg = rcfg.get("embed") or {}
    provider = embed_cfg.get("provider") or "openai"
    if str(reader_name).lower() == "mock":
        provider = "mock"
    from src.locomo_eval.mem0.embeddings import get_embedder

    k = getattr(overrides, "rag_k", None)
    if k is None:
        k = rcfg.get("k", 2)
    return {
        "rag_index_dir": out_root / str(index_run) / "rag_index",
        "rag_top_k": int(k),
        "rag_embedder": get_embedder(
            provider,
            model=embed_cfg.get("model") or "text-embedding-3-small",
        ),
    }


def _openai_memory_builder_kwargs(
    cfg: dict, overrides: argparse.Namespace, memory_name: str
) -> dict:
    if resolve_memory_name(memory_name) != "openai_memory":
        return {}
    ocfg = cfg.get("openai_memory") or {}
    index_run = getattr(overrides, "openai_memory_index_run_id", None) or ocfg.get(
        "index_run_id"
    )
    if not index_run:
        raise SystemExit(
            "openai_memory requires openai_memory.index_run_id in YAML or "
            "--openai-memory-index-run-id (after python -m "
            "src.locomo_eval.openai_memory.run_index)."
        )
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    return {
        "openai_memory_index_dir": out_root / str(index_run) / "openai_memory_index",
    }


def _preprocess_builder_kwargs(
    cfg: dict, overrides: argparse.Namespace, memory_name: str
) -> dict:
    """Optional dump dir for raw_chunks / session_summaries. Other builders ignore these."""
    resolved = resolve_memory_name(memory_name)
    if resolved not in ("raw_chunks", "session_summaries"):
        return {}
    pcfg = cfg.get("preprocess") or {}
    index_run = getattr(overrides, "preprocess_index_run_id", None) or pcfg.get(
        "index_run_id"
    )
    if not index_run:
        return {}
    out_root = Path(overrides.output_dir or cfg["run"]["output_dir"])
    index_dir = out_root / str(index_run) / "preprocess"
    top_k = pcfg.get("top_k")
    if getattr(overrides, "retrieve_top_k", None) is not None:
        top_k = overrides.retrieve_top_k
    if top_k is not None:
        top_k = int(top_k)
    return {
        "preprocess_index_dir": index_dir,
        "retrieve_top_k": top_k,
    }


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


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"Invalid boolean {value!r}; use on/off or true/false")


def _teacher_call_kwargs(cfg: dict, overrides: argparse.Namespace | None = None) -> dict:
    tcfg = cfg.get("teacher") or {}
    thinking = None
    if overrides is not None:
        flag = getattr(overrides, "thinking", None)
        if flag == "on":
            thinking = True
        elif flag == "off":
            thinking = False
    if thinking is None:
        thinking = _as_bool(tcfg.get("thinking"), DEFAULT_TEACHER_THINKING)
    budget = tcfg.get("thinking_budget_tokens", 1024)
    max_tokens = int(tcfg.get("max_tokens", 512))
    if overrides is not None:
        teacher_max = getattr(overrides, "teacher_max_tokens", None)
        if teacher_max is not None:
            max_tokens = int(teacher_max)
    return {
        "temperature": float(tcfg.get("temperature", 0.0)),
        "max_tokens": max_tokens,
        "max_retries": int(tcfg.get("max_retries", 8)),
        "min_request_interval_s": float(tcfg.get("min_request_interval_s", 0.5)),
        "max_wait_s": float(tcfg.get("max_wait_s", 3600.0)),
        "prompt_path": tcfg.get("prompt_path"),
        "graph_prompt_path": tcfg.get("graph_prompt_path"),
        "thinking": thinking,
        "thinking_budget_tokens": int(budget),
    }


def _force_mock_teachers(overrides: argparse.Namespace, reader_name: str) -> bool:
    if str(reader_name).lower() == "mock":
        return True
    return str(getattr(overrides, "teacher", None) or "").lower() == "mock"


def _resolve_agent_run(
    cfg: dict, overrides: argparse.Namespace, memory_name: str, reader_name: str
) -> dict[str, Any]:
    """Decide whether this invocation uses a coding-agent harness.

    ``agent.adapter: none`` (or omitted) keeps the one-shot reader. Mock reader
    forces the mock adapter so local/GCP smokes never need the Codex binary.
    """
    acfg = dict(cfg.get("agent") or {})
    adapter = getattr(overrides, "agent", None) or acfg.get("adapter") or acfg.get("id")
    persist_flag = getattr(overrides, "agent_persist", None)
    if persist_flag == "on":
        persist = True
    elif persist_flag == "off":
        persist = False
    else:
        persist = _as_bool(acfg.get("persist_memory"), False)
    tools = getattr(overrides, "agent_tools", None) or acfg.get("tools") or "native"
    answer_mode = getattr(overrides, "answer_mode", None) or (
        (cfg.get("pipeline") or {}).get("answer_mode") or "reader"
    )
    if adapter and str(adapter).strip().lower() not in ("", "none"):
        answer_mode = "agent"
    if resolve_memory_name(memory_name) == "workspace_files":
        answer_mode = "agent"
        adapter = adapter or "mock"
    if str(answer_mode).lower() != "agent":
        return {
            "enabled": False,
            "adapter": None,
            "model": None,
            "persist_memory": persist,
            "tools": str(tools),
        }
    adapter = str(adapter or acfg.get("adapter") or "mock").strip().lower()
    model = (
        getattr(overrides, "model", None)
        or acfg.get("model")
        or (cfg.get("reader") or {}).get("model")
        or "gpt-5"
    )
    if str(reader_name).lower() == "mock" or adapter == "mock":
        adapter = "mock"
        model = "mock"
    return {
        "enabled": True,
        "adapter": adapter,
        "model": model,
        "persist_memory": persist,
        "tools": str(tools),
        "timeout_s": float(acfg.get("timeout_s") or 600),
        "codex_bin": acfg.get("codex_bin"),
    }


def _build_teacher(cfg: dict, overrides: argparse.Namespace, memory_name: str, reader_name: str):
    """Construct a Teacher only for teacher_session_summaries. Other builders ignore teacher YAML."""
    if resolve_memory_name(memory_name) not in (
        TeacherSessionMemoryBuilder.name,
        AgentFactMemoryBuilder.name,
    ):
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
    kwargs = _teacher_call_kwargs(cfg, overrides)
    return get_teacher(provider, model=model, **kwargs)


def _teacher_roster(
    cfg: dict, overrides: argparse.Namespace, memory_name: str, reader_name: str
) -> list[dict]:
    """Resolved teacher slots for this memory method (before get_teacher).

    ``teacher_graph`` + ``--teacher-model`` replaces the one sandwich writer.
    Pooled/fused keep the YAML roster (cheap_k3); experiment matrices must
    not schedule those methods.
    """
    resolved = resolve_memory_name(memory_name)
    force_mock = _force_mock_teachers(overrides, reader_name)
    tcfg = cfg.get("teacher") or {}
    roster = list(cfg.get("teachers") or [])
    if not roster:
        provider = getattr(overrides, "teacher", None) or tcfg.get("provider")
        model = getattr(overrides, "teacher_model", None) or tcfg.get("model")
        if not provider:
            provider = "mock" if force_mock else "openai"
        if not model:
            return []
        return [{"id": provider, "provider": provider, "model": model}]
    if getattr(overrides, "teacher_model", None) and resolved == "teacher_graph":
        roster = [{**roster[0], "model": overrides.teacher_model}]
        if getattr(overrides, "teacher", None):
            roster[0]["provider"] = overrides.teacher
            roster[0]["id"] = overrides.teacher
    return roster


def _teacher_cardinality(
    orch: TeacherOrchestrator | None, teacher
) -> tuple[int, list[str]]:
    """How many write-path teachers this run constructed (0 if none)."""
    if orch is not None:
        ids = [str(t.teacher_id) for t in orch.teachers]
        return len(ids), ids
    if teacher is not None:
        tid = str(getattr(teacher, "teacher_id", None) or teacher.model_name)
        return 1, [tid]
    return 0, []


def _build_orchestrator(
    cfg: dict, overrides: argparse.Namespace, memory_name: str, reader_name: str
) -> TeacherOrchestrator | None:
    """K teachers + pool/fusion for graph memory conditions. None for other builders."""
    resolved = resolve_memory_name(memory_name)
    if resolved not in ORCHESTRATED_GRAPH_NAMES:
        return None
    force_mock = _force_mock_teachers(overrides, reader_name)
    roster = _teacher_roster(cfg, overrides, memory_name, reader_name)
    if not roster:
        raise SystemExit(
            f"{resolved} requires teacher.model in YAML, teachers:, or --teacher-model"
        )

    kwargs = _teacher_call_kwargs(cfg, overrides)
    teachers = []
    for i, slot in enumerate(roster):
        provider = "mock" if force_mock else (slot.get("provider") or "openai")
        model = slot.get("model")
        if not model:
            raise SystemExit(f"{resolved} teacher slot {i} is missing model")
        tid = str(slot.get("id") or model)
        teachers.append(get_teacher(provider, model=model, teacher_id=tid, **kwargs))

    ocfg = cfg.get("orchestrator") or {}
    default_pool, default_fusion = _ORCH_DEFAULTS[resolved]
    pool = getattr(overrides, "pool", None) or ocfg.get("pool") or default_pool
    fusion = getattr(overrides, "fusion", None) or ocfg.get("fusion") or default_fusion
    majority_k = ocfg.get("majority_k")
    if majority_k is not None:
        majority_k = int(majority_k)
    embed_cfg = ocfg.get("embed") or {}
    embed_provider = embed_cfg.get("provider") or "mock"
    if force_mock:
        embed_provider = "mock"
    from src.locomo_eval.mem0.embeddings import get_embedder

    return TeacherOrchestrator(
        teachers,
        pool=str(pool),
        fusion=str(fusion),
        min_votes=majority_k,
        rng_seed=int(ocfg.get("rng_seed", 0)),
        embedder=get_embedder(
            embed_provider,
            model=embed_cfg.get("model") or "text-embedding-3-small",
        ),
        thinking=kwargs.get("thinking"),
    )


def _listed_index_sample_ids(index_dir: Path | None) -> list[str]:
    """Sample ids recorded in an index dump's ``index.jsonl`` (empty if none)."""
    if index_dir is None:
        return []
    path = Path(index_dir) / "index.jsonl"
    if not path.is_file():
        return []
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if sid:
            ids.append(str(sid))
    return ids


def _filter_conversations_to_index(
    conversations: list,
    index_dir: Path | None,
    *,
    subset_requested: bool,
) -> list:
    """Keep QA inside the dump. A partial smoke index must not round-robin into missing samples."""
    dumped = _listed_index_sample_ids(index_dir)
    if not dumped:
        return conversations
    dumped_set = set(dumped)
    missing = [c.sample_id for c in conversations if c.sample_id not in dumped_set]
    kept = [c for c in conversations if c.sample_id in dumped_set]
    if missing and not subset_requested:
        raise SystemExit(
            f"Index dump at {index_dir} has {len(dumped)} sample(s) but QA needs "
            f"{len(conversations)} (missing e.g. {missing[0]}). Re-index without "
            "--max-samples, or pass --max-samples / --sample-id to evaluate only "
            "indexed conversations."
        )
    if not kept:
        raise SystemExit(
            f"No overlap between QA conversations and index dump at {index_dir}."
        )
    if missing:
        print(
            f"QA restricted to {len(kept)} indexed sample(s) "
            f"(dump missing {len(missing)}, e.g. {missing[0]})"
        )
    return kept


def _dump_memory_and_claim_audit(
    run_dir: Path,
    builder: Any,
    *,
    used_memories: dict,
    used_memories_by_question: dict,
    question_sample_ids: dict,
    max_chars: int | None,
    prompt_template: str,
    example_question: str | None,
    prediction_rows: list[dict],
    reader_traces: list[dict],
    metrics: dict | None = None,
    meta: dict | None = None,
) -> None:
    write_memory_run_log(
        run_dir,
        used_memories,
        max_chars_cfg=max_chars,
        prompt_template=prompt_template,
        example_question=example_question,
        memories_by_question=used_memories_by_question or None,
        question_sample_ids=question_sample_ids or None,
    )
    write_teacher_module(
        run_dir,
        calls=collect_teacher_call_log(builder),
        fusion_rows=collect_fusion_log(builder),
        session_texts=collect_session_texts(builder),
    )
    write_graph_module(run_dir, getattr(builder, "graphs_by_sample", {}) or {})
    write_claim_audit(
        run_dir,
        prediction_rows=prediction_rows,
        reader_traces=reader_traces,
        teacher_calls=collect_teacher_call_log(builder),
        fusion_rows=collect_fusion_log(builder),
        ingest_rows=collect_ingest_log(builder),
        retrieve_ranks=collect_retrieve_log(builder),
        memories_by_sample=used_memories,
        memories_by_question=used_memories_by_question or None,
        metrics=metrics,
        meta=meta,
    )


def _reset_run_output(run_dir: Path) -> None:
    """Clear generated artifacts so one run id always means one fresh run.

    Reusing a run id must not keep old answers, metrics, or autorater reports.
    """
    for dirname in ("plots", "memory", "autorater", "reader", "agent", "prompts"):
        path = run_dir / dirname
        if path.is_dir():
            shutil.rmtree(path)
    for filename in (
        "predictions.jsonl",
        "predictions.csv",
        "metrics.json",
        "metrics_by_category.csv",
        "run_meta.json",
        "TRACE.md",
        "cost.json",
        "SUMMARY.md",
        "ATTRIBUTION.md",
        "attribution.jsonl",
        "config.source.yaml",
        "config.resolved.yaml",
        "examples.parquet",
        "summary.parquet",
        "run.json",
        "_SUCCESS",
        "errors.jsonl",
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
    ``--memory`` overrides it. Not tied to ``configs/presets/mem0_baseline.yaml``.
    """
    data_path = Path(overrides.data or cfg["data"]["raw_path"])
    # Builder id (raw_chunks / session_summaries), not a path to another YAML.
    memory_name = overrides.memory or cfg["pipeline"]["memory"]
    reader_name = overrides.reader or cfg["reader"]["provider"]
    model = overrides.model or cfg["reader"]["model"]
    agent_run = _resolve_agent_run(cfg, overrides, memory_name, reader_name)
    if agent_run["enabled"] and resolve_memory_name(memory_name) != "workspace_files":
        raise SystemExit(
            "Agent harness requires pipeline.memory=workspace_files "
            "(conversation files). Use full_context for model-only one-shot."
        )
    prompt_path = locate_prompt_file(overrides.prompt or cfg["pipeline"]["prompt_path"])
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
    # Claim-audit frozen config (YAML + CLI overrides). Prompt bundle then writes
    # TRACE.md / prompts/ and does not replace these files if they already exist.
    write_frozen_config(
        run_dir,
        cfg=cfg,
        overrides=overrides,
        source_config=getattr(overrides, "config", None),
        repo_root=ROOT,
    )
    pred_path = run_dir / "predictions.jsonl"

    prompt_version, prompt_template = load_prompt_template(prompt_path)
    cfg_snapshot = {
        **cfg,
        "pipeline": {
            **(cfg.get("pipeline") or {}),
            "prompt_path": repo_rel(prompt_path),
        },
    }
    write_prompt_bundle(
        run_dir,
        cfg_snapshot,
        config_entry=getattr(overrides, "config", None),
    )
    max_chars = cfg["pipeline"].get("memory_max_chars")
    if max_chars is not None:
        max_chars = int(max_chars)
    teacher = _build_teacher(cfg, overrides, memory_name, reader_name)
    orchestrator = _build_orchestrator(cfg, overrides, memory_name, reader_name)
    mem0_kwargs = _mem0_builder_kwargs(cfg, overrides, memory_name, reader_name)
    preprocess_kwargs = _preprocess_builder_kwargs(cfg, overrides, memory_name)
    rag_kwargs = _rag_builder_kwargs(cfg, overrides, memory_name, reader_name)
    openai_kwargs = _openai_memory_builder_kwargs(cfg, overrides, memory_name)
    agent_workspace_root = None
    if agent_run["enabled"]:
        agent_workspace_root = run_dir / "agent" / "workspaces"
    builder = get_memory_builder(
        memory_name,
        max_chars=max_chars,
        teacher=teacher,
        orchestrator=orchestrator,
        **mem0_kwargs,
        **preprocess_kwargs,
        **rag_kwargs,
        **openai_kwargs,
        agent_workspace_root=agent_workspace_root,
        agent_persist_memory=bool(agent_run.get("persist_memory")),
    )

    rate_cfg = cfg.get("reader") or {}
    temperature_override = getattr(overrides, "temperature", None)
    reader_temperature = (
        float(temperature_override)
        if temperature_override is not None
        else float(rate_cfg.get("temperature", 0.0))
    )
    max_tokens_override = getattr(overrides, "max_tokens", None)
    reader_max_tokens_raw = (
        max_tokens_override
        if max_tokens_override is not None
        else rate_cfg.get("max_tokens", 64)
    )
    reader_max_tokens = (
        int(reader_max_tokens_raw) if reader_max_tokens_raw is not None else None
    )
    message_layout_override = getattr(overrides, "message_layout", None)
    reader_message_layout = str(
        message_layout_override
        or rate_cfg.get("message_layout")
        or "default_system_user"
    )
    reader_model = "mock" if reader_name.lower() == "mock" else model
    if agent_run["enabled"] and agent_run["adapter"] == "mock":
        reader_model = "mock"
    reader_thinking = None
    thinking_flag = getattr(overrides, "reader_thinking", None)
    if thinking_flag == "on":
        reader_thinking = True
    elif thinking_flag == "off":
        reader_thinking = False
    agent_runner = None
    comparison_contract: dict[str, Any] | None = None
    if agent_run["enabled"]:
        agent_kwargs = {}
        if agent_run.get("timeout_s") is not None:
            agent_kwargs["timeout_s"] = agent_run["timeout_s"]
        if agent_run.get("codex_bin"):
            agent_kwargs["codex_bin"] = agent_run["codex_bin"]
        agent_runner = get_agent_runner(
            agent_run["adapter"],
            model=str(agent_run["model"] or reader_model),
            persist_memory=bool(agent_run.get("persist_memory")),
            tools=str(agent_run.get("tools") or "native"),
            **agent_kwargs,
        )
        reader = None
        answer_model_name = agent_runner.model_name
        answer_provider = agent_run["adapter"]
        comparison_cfg = dict((cfg.get("agent") or {}).get("comparison") or {})
        validate_strict_contract(
            comparison_cfg,
            adapter=answer_provider,
            model_snapshot=(comparison_cfg.get("backbone") or {}).get("model_snapshot"),
        )
        comparison_contract = resolve_contract(
            config=cfg,
            adapter=answer_provider,
            model=answer_model_name,
            model_snapshot=(comparison_cfg.get("backbone") or {}).get("model_snapshot"),
            prompt_path=prompt_path,
            memory_type=memory_name,
        )
    else:
        reader = get_reader(
            reader_name,
            model=reader_model,
            temperature=reader_temperature,
            max_tokens=reader_max_tokens,
            max_retries=int(rate_cfg.get("max_retries", 8)),
            min_request_interval_s=float(rate_cfg.get("min_request_interval_s", 0.0)),
            max_wait_s=float(rate_cfg.get("max_wait_s", 3600.0)),
            message_layout=reader_message_layout,
            thinking=reader_thinking,
        )
        answer_model_name = reader.model_name
        answer_provider = reader_name

    conversations = load_conversations(data_path)
    if overrides.sample_id:
        conversations = [c for c in conversations if c.sample_id == overrides.sample_id]
        if not conversations:
            raise SystemExit(f"No sample_id match: {overrides.sample_id}")
    max_samples = getattr(overrides, "max_samples", None)
    if max_samples is not None:
        conversations = conversations[: int(max_samples)]
    index_dir = (
        rag_kwargs.get("rag_index_dir")
        or mem0_kwargs.get("mem0_index_dir")
        or openai_kwargs.get("openai_memory_index_dir")
    )
    conversations = _filter_conversations_to_index(
        conversations,
        index_dir,
        subset_requested=bool(overrides.sample_id or max_samples is not None),
    )

    # These builders do not depend on the question; build once per conversation.
    memory_by_sample = {}
    retrieve_top_k = preprocess_kwargs.get("retrieve_top_k")
    if is_question_independent(memory_name, retrieve_top_k=retrieve_top_k):
        for conv in conversations:
            q = conv.questions[0] if conv.questions else None
            if q is None:
                continue
            memory_by_sample[conv.sample_id] = builder.build(conv, q)

    pairs = list(iter_questions(conversations))
    question_sample = (
        getattr(overrides, "question_sample", None)
        or (cfg.get("pipeline") or {}).get("question_sample")
        or "round_robin"
    )
    if max_questions is not None:
        pairs = select_question_pairs(
            pairs, int(max_questions), mode=str(question_sample)
        )

    teacher_model = getattr(builder, "teacher_model", None)
    teacher_provider = getattr(builder, "teacher_provider", None)
    orch = getattr(builder, "orchestrator", None)
    n_teachers, teacher_ids = _teacher_cardinality(orchestrator, teacher)
    print(
        f"Run {run_id}: {len(pairs)} questions | memory={builder.name} | "
        f"{'agent' if agent_runner else 'reader'}={answer_provider}/{answer_model_name}"
        + (f" | teacher={teacher_provider}/{teacher_model}" if teacher_model else "")
        + (
            f" | pool={orch.pool} fusion={orch.fusion}"
            if orch is not None
            else ""
        )
        + (
            f" | persist={agent_run.get('persist_memory')} tools={agent_run.get('tools')}"
            if agent_runner is not None
            else ""
        )
        + f" | question_sample={question_sample}"
    )

    prediction_rows: list[dict] = []
    reader_traces: list[dict] = []
    agent_traces: list[dict] = []
    agent_trajectories: list[dict] = []
    agent_events: list[dict] = []
    trajectories_by_qid: dict[str, Any] = {}
    n_api = 0
    used_memories: dict[str, Memory] = {}
    used_memories_by_question: dict[str, Memory] = {}
    question_sample_ids: dict[str, str] = {}
    example_question: str | None = None
    question_independent = is_question_independent(
        memory_name, retrieve_top_k=retrieve_top_k
    )
    try:
        for i, (conv, q) in enumerate(pairs, start=1):
            if conv.sample_id in memory_by_sample:
                memory = memory_by_sample[conv.sample_id]
            else:
                memory = builder.build(conv, q)
            used_memories[conv.sample_id] = memory
            if not question_independent:
                used_memories_by_question[q.question_id] = memory
                question_sample_ids[q.question_id] = conv.sample_id
            if example_question is None:
                example_question = q.question

            if agent_runner is not None:
                workspaces = getattr(builder, "workspaces", {}) or {}
                workspace_dir = workspaces.get(conv.sample_id)
                if workspace_dir is None:
                    raise RuntimeError(
                        f"workspace_files missing for sample {conv.sample_id}"
                    )
                filled = render_agent_prompt(prompt_template, q.question)
                result = agent_runner.run(
                    AgentRequest(
                        workspace_dir=Path(workspace_dir),
                        sample_id=conv.sample_id,
                        question_id=q.question_id,
                        question=q.question,
                        prompt=filled,
                        evidence_ids=list(q.evidence),
                        tool_budget=(comparison_contract or {}).get("tool_budget") or {},
                    )
                )
                if comparison_contract is not None:
                    comparison_contract["context"]["workspace_manifest_sha256"] = (
                        workspace_manifest_sha256(Path(workspace_dir))
                    )
                answer = result.predicted_answer
                meta = {
                    **(result.call_meta or {}),
                    "latency_s": result.latency_s,
                    "usage": result.usage,
                    "reasoning": (result.call_meta or {}).get("reasoning") or "",
                }
                trajectories_by_qid[q.question_id] = result.trajectory
                agent_trajectories.append(
                    {
                        "sample_id": conv.sample_id,
                        "question_id": q.question_id,
                        **result.trajectory.to_dict(),
                    }
                )
                for raw in result.events_raw:
                    agent_events.append(
                        {
                            "sample_id": conv.sample_id,
                            "question_id": q.question_id,
                            "event": raw,
                        }
                    )
                agent_traces.append(
                    result.to_trace_row(
                        sample_id=conv.sample_id,
                        question_id=q.question_id,
                        adapter=answer_provider,
                        model=answer_model_name,
                        prompt_version=prompt_version,
                    )
                )
            else:
                answer, meta = reader.answer(memory.text, q.question, prompt_template)
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
                reader_model=answer_model_name,
                prompt_version=prompt_version,
                evidence=list(q.evidence),
                run_id=run_id,
                teacher_model=memory.teacher_model,
                teacher_provider=memory.teacher_provider,
            )
            row = pred.to_dict()
            row["latency_s"] = meta.get("latency_s")
            row["usage"] = meta.get("usage")
            row["memory_schema_version"] = memory.schema_version
            if memory.search_latency_s is not None:
                row["search_latency_s"] = memory.search_latency_s
            if agent_runner is not None:
                row["agent"] = answer_provider
                row["agent_persist"] = agent_run.get("persist_memory")
                row["agent_tools"] = agent_run.get("tools")
            prediction_rows.append(row)
            reader_traces.append(
                {
                    "sample_id": q.sample_id,
                    "question_id": q.question_id,
                    "role": "agent" if agent_runner is not None else "reader",
                    "provider": answer_provider,
                    "model": answer_model_name,
                    "predicted_answer": answer,
                    "reasoning": meta.get("reasoning") or "",
                    "reasoning_tokens": meta.get("reasoning_tokens"),
                    "reasoning_effort": meta.get("reasoning_effort"),
                    "thinking": meta.get("thinking"),
                    "thinking_supported": meta.get("thinking_supported"),
                    "latency_s": meta.get("latency_s"),
                    "usage": meta.get("usage") or {},
                    "prompt_version": prompt_version,
                }
            )

            # Flush for audit after a crash. The next invocation clears this
            # partial file and regenerates.
            write_jsonl(pred_path, prediction_rows)

            if i % 10 == 0 or i == len(pairs):
                print(
                    f"  [{i}/{len(pairs)}] {q.question_id} api_calls_this_run={n_api}"
                )
    except Exception as exc:
        write_jsonl(pred_path, prediction_rows)
        if used_memories:
            _dump_memory_and_claim_audit(
                run_dir,
                builder,
                used_memories=used_memories,
                used_memories_by_question=used_memories_by_question,
                question_sample_ids=question_sample_ids,
                max_chars=max_chars,
                prompt_template=prompt_template,
                example_question=example_question,
                prediction_rows=prediction_rows,
                reader_traces=reader_traces,
            )
            write_reader_module(
                run_dir,
                prediction_rows=prediction_rows,
                traces=reader_traces,
            )
            if agent_runner is not None:
                write_agent_module(
                    run_dir,
                    traces=agent_traces,
                    trajectories=agent_trajectories,
                    events=agent_events,
                    metrics={},
                    comparison_contract=comparison_contract,
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
            memories_by_question=used_memories_by_question or None,
            question_sample_ids=question_sample_ids or None,
        )
        print(f"Memory audit: {mem_log_dir}")
        teachers_dir = write_teacher_module(
            run_dir,
            calls=collect_teacher_call_log(builder),
            fusion_rows=collect_fusion_log(builder),
            session_texts=collect_session_texts(builder),
        )
        if teachers_dir is not None:
            print(f"Teacher traces: {teachers_dir}")
        graph_dir = write_graph_module(
            run_dir, getattr(builder, "graphs_by_sample", {}) or {}
        )
        if graph_dir is not None:
            print(f"Graph snapshot: {graph_dir}")

    summary = summarize_predictions(prediction_rows)
    summary["memory_type"] = builder.name
    summary["reader_model"] = answer_model_name
    summary["reader_family"] = resolve_model(answer_model_name).family
    summary["prompt_version"] = prompt_version
    if teacher_model:
        summary["teacher_model"] = teacher_model
        summary["teacher_provider"] = teacher_provider
        if "+" in str(teacher_model):
            summary["teacher_family"] = "multi"
        else:
            summary["teacher_family"] = resolve_model(teacher_model).family
    teacher_thinking = None
    if orch is not None:
        teacher_thinking = getattr(orch, "thinking", None)
    elif teacher is not None:
        teacher_thinking = getattr(teacher, "thinking", None)
    if orch is not None:
        summary["teacher_pool"] = orch.pool
        summary["teacher_fusion"] = orch.fusion
    summary["n_teachers"] = n_teachers
    summary["teacher_ids"] = teacher_ids
    if teacher_thinking is not None:
        summary["teacher_thinking"] = teacher_thinking

    agent_metric_rows: list[dict] = []
    if trajectories_by_qid:
        for row in prediction_rows:
            traj = trajectories_by_qid.get(str(row.get("question_id") or ""))
            if traj is None:
                continue
            scores = score_row(
                str(row.get("predicted_answer") or ""),
                str(row.get("reference_answer") or ""),
                int(row.get("category") or 0),
            )
            extra = score_agent_row(
                locomo_f1=float(scores["locomo_f1"]),
                exact_match=float(scores["exact_match"]),
                trajectory=traj,
            )
            row.update(extra)
            agent_metric_rows.append(extra)
        summary["agent"] = summarize_agent_rows(agent_metric_rows)
    if comparison_contract is not None:
        comparison_contract["status"] = comparison_status(
            comparison_contract,
            harness_failed=any(
                bool(row.get("harness_failed")) for row in agent_metric_rows
            ),
        )
        comparison_contract["sha256"] = contract_sha256(comparison_contract)

    # Pins for later audit: data file, code commit, prompt, model, memory type.
    meta_out = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "locomo_pin": cfg["data"].get("locomo_commit"),
        "memory_type": builder.name,
        "memory_max_chars": max_chars,
        "preprocess_index_run_id": getattr(overrides, "preprocess_index_run_id", None)
        or (cfg.get("preprocess") or {}).get("index_run_id"),
        "retrieve_top_k": retrieve_top_k,
        "rag_index_run_id": getattr(overrides, "rag_index_run_id", None)
        or (cfg.get("rag") or {}).get("index_run_id"),
        "rag_k": rag_kwargs.get("rag_top_k"),
        "openai_memory_index_run_id": getattr(
            overrides, "openai_memory_index_run_id", None
        )
        or (cfg.get("openai_memory") or {}).get("index_run_id"),
        "reader_provider": answer_provider,
        "reader_model": answer_model_name,
        "reader_family": resolve_model(answer_model_name).family,
        "teacher_provider": teacher_provider,
        "teacher_model": teacher_model,
        "teacher_family": (
            "multi"
            if teacher_model and "+" in str(teacher_model)
            else (resolve_model(teacher_model).family if teacher_model else None)
        ),
        "teacher_pool": orch.pool if orch is not None else None,
        "teacher_fusion": orch.fusion if orch is not None else None,
        "n_teachers": n_teachers,
        "teacher_ids": teacher_ids,
        "teacher_thinking": teacher_thinking,
        "temperature": reader_temperature,
        "reader_thinking": reader_thinking,
        "max_tokens": reader_max_tokens,
        "message_layout": reader_message_layout,
        "max_retries": rate_cfg.get("max_retries", 8),
        "min_request_interval_s": rate_cfg.get("min_request_interval_s", 0.0),
        "prompt_path": repo_rel(prompt_path),
        "prompt_sha256": _file_sha256(prompt_path),
        "prompt_version": prompt_version,
        "max_questions": max_questions,
        "max_samples": max_samples,
        "question_sample": question_sample,
        "n_predictions": len(prediction_rows),
        "n_new_api_calls": n_api,
        "code_git_hash": _git_hash(),
        "package_version": "0.1.5",
        "memory_schema_version": "memory_io.v1",
        "memory_audit_dir": "memory",
        "audit_layout": audit_layout_meta(),
        "answer_mode": "agent" if agent_runner is not None else "reader",
        "agent": agent_run.get("adapter") if agent_runner is not None else None,
        "agent_persist": agent_run.get("persist_memory") if agent_runner is not None else None,
        "agent_tools": agent_run.get("tools") if agent_runner is not None else None,
        "agent_model": answer_model_name if agent_runner is not None else None,
        "agent_adapter": answer_provider if agent_runner is not None else None,
        "comparison_contract": comparison_contract,
        "comparison_contract_sha256": (
            comparison_contract.get("sha256") if comparison_contract else None
        ),
    }

    paths = write_run_report(run_dir, prediction_rows, summary, meta_out)
    write_reader_module(
        run_dir,
        prediction_rows=prediction_rows,
        traces=reader_traces,
        summary=summary,
    )
    if agent_runner is not None:
        agent_paths = write_agent_module(
            run_dir,
            traces=agent_traces,
            trajectories=agent_trajectories,
            events=agent_events,
            metrics=summary.get("agent") or {},
            summary_rows=agent_metric_rows,
            comparison_contract=comparison_contract,
        )
        print("Agent audit:")
        for k, v in agent_paths.items():
            print(f"  {k}: {v}")
    claim_paths = write_claim_audit(
        run_dir,
        prediction_rows=prediction_rows,
        reader_traces=reader_traces,
        teacher_calls=collect_teacher_call_log(builder),
        fusion_rows=collect_fusion_log(builder),
        ingest_rows=collect_ingest_log(builder),
        retrieve_ranks=collect_retrieve_log(builder),
        memories_by_sample=used_memories,
        memories_by_question=used_memories_by_question or None,
        metrics=summary,
        meta=meta_out,
    )
    print("Wrote audit package:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    if claim_paths:
        print("Claim audit:")
        for k, v in claim_paths.items():
            print(f"  {k}: {v}")
    print(
        f"API stats: new_api={n_api}"
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
        default="configs/presets/mem0_baseline.yaml",
        help="YAML path (default: Mem0-parity controls over session_summaries)",
    )
    p.add_argument("--data", default=None, help="Override path to locomo10.json")
    p.add_argument(
        "--memory",
        default=None,
        help="Override pipeline.memory builder id (raw_chunks, session_summaries, teacher_session_summaries, teacher_graph, pooled_teacher_graph, fused_teacher_graph, full_context, rag, openai_memory, mem0, mem0g, workspace_files)",
    )
    p.add_argument("--reader", default=None, help="openai | deepseek | anthropic | mock")
    p.add_argument("--model", default=None, help="Answer-model id, e.g. gpt-4.1-mini or gpt-5.6-luna")
    p.add_argument(
        "--agent",
        default=None,
        help="Harness adapter: mock | codex | claude_code | opencode | pi (omit for one-shot reader)",
    )
    p.add_argument(
        "--agent-persist",
        default=None,
        choices=("on", "off"),
        help="Agent-specific persistent memory (Codex ephemeral vs workspace notes)",
    )
    p.add_argument(
        "--agent-tools",
        default=None,
        choices=("native", "controlled"),
        help="Native harness tools vs a later controlled allowlist",
    )
    p.add_argument("--temperature", type=float, default=None, help="Override reader temperature")
    p.add_argument("--max-tokens", type=int, default=None, help="Override reader completion-token limit")
    p.add_argument(
        "--reader-thinking",
        default=None,
        choices=("on", "off"),
        help="Answer-model thinking/reasoning (reader sweep). Frozen sandwich readers omit this.",
    )
    p.add_argument(
        "--message-layout",
        choices=("default_system_user", "mem0_system_only"),
        default=None,
        help="Override Chat Completions message layout",
    )
    p.add_argument(
        "--teacher",
        default=None,
        help="Teacher provider: openai | anthropic | deepseek | mock",
    )
    p.add_argument(
        "--teacher-model",
        default=None,
        help="Teacher model id (single-teacher conditions), e.g. gpt-4o-mini or claude-haiku-4-5",
    )
    p.add_argument(
        "--pool",
        default=None,
        choices=("single", "equal_weight", "random", "round_robin"),
        help="Override orchestrator.pool for teacher graph conditions",
    )
    p.add_argument(
        "--fusion",
        default=None,
        choices=FUSION_POLICIES,
        help="Override orchestrator.fusion for teacher graph conditions",
    )
    p.add_argument(
        "--thinking",
        default=None,
        choices=("on", "off"),
        help="Teacher thinking/reasoning (default: on). Frozen reader is unchanged.",
    )
    p.add_argument(
        "--teacher-max-tokens",
        type=int,
        default=None,
        help="Override teacher completion-token limit (not the frozen reader).",
    )
    p.add_argument("--prompt", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument(
        "--question-sample",
        default=None,
        choices=("round_robin", "prefix"),
        help="How --max-questions picks items: round_robin (default, one per conversation first) or prefix (file order)",
    )
    p.add_argument("--sample-id", default=None, help="Restrict to one conversation")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap conversations (file order) before --max-questions. "
        "Keeps round-robin inside an indexed subset.",
    )
    p.add_argument(
        "--mem0-index-run-id",
        default=None,
        help="experiments/<id>/mem0_index dump for mem0/mem0g builders",
    )
    p.add_argument(
        "--preprocess-index-run-id",
        default=None,
        help="experiments/<id>/preprocess dump for raw_chunks / session_summaries",
    )
    p.add_argument(
        "--retrieve-top-k",
        type=int,
        default=None,
        help="Optional session-unit cap after naive rank (default: all units)",
    )
    p.add_argument(
        "--rag-index-run-id",
        default=None,
        help="experiments/<id>/rag_index dump for the rag builder",
    )
    p.add_argument(
        "--rag-k",
        type=int,
        default=None,
        help="RAG top-k chunks (paper: 1 or 2)",
    )
    p.add_argument(
        "--openai-memory-index-run-id",
        default=None,
        help="experiments/<id>/openai_memory_index dump for openai_memory",
    )
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
