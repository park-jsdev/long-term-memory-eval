"""Expand an experiment YAML into an ordered list of ``ExperimentRunSpec``.

The same YAML always yields the same order and the same hashed run ids.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

from src.experiment_runner.experiment_run_spec import (
    ExperimentRunSpec,
    ReaderModelRef,
    WriterModelRef,
)
from src.experiment_runner.experiment_types import (
    AGENT,
    MATRIX_AXIS_ORDER,
    SANDWICH,
    require_known_type,
)
from src.experiment_runner.hashed_run_id import hashed_run_id
from src.experiment_runner.locomo_method_yaml import method_yaml_for
from src.experiment_runner.shared_index_paths import index_run_ids_for_memory
from src.locomo_eval.memory import is_removed_multi_model_method
from src.locomo_eval.prompts import (
    QA_MEM0_V1,
    QA_WORKSPACE_NOTES_ONLY_V1,
    QA_WORKSPACE_PERSIST_V1,
    QA_WORKSPACE_V1,
)
from src.locomo_eval.agents.comparison import validate_strict_contract

# Completion-token caps when ``matrix.thinking`` is set. Overrides catalog
# ``teacher:`` overlays so GPT-5 on-cells are not stuck at thinking=false.
# Writer-on is 32768 because graph+high-effort 2025 smoke filled the old
# 8192 cap with reasoning_tokens. Writer-off is 8192 because 2026
# DeepSeek-V4 graph-off truncated JSON at 4096 (parse fallback_mock).
THINKING_OUTPUT_TOKENS = {
    ("reader", False): 256,
    ("reader", True): 8192,
    ("writer", False): 8192,
    ("writer", True): 32768,
}


def expand_run_matrix(cfg: dict[str, Any]) -> list[ExperimentRunSpec]:
    experiment = cfg.get("experiment") or {}
    exp_type = require_known_type(experiment.get("type") or "sweep")
    name = str(experiment.get("name") or "unnamed")
    freeze = dict(experiment.get("freeze") or {})
    matrix = dict(cfg.get("matrix") or {})
    benchmark_cfg = cfg.get("benchmark") or {}
    benchmark = str(benchmark_cfg.get("name") or "locomo")
    prompt_path = str(
        freeze.get("prompt_path")
        or (cfg.get("pipeline") or {}).get("prompt_path")
        or QA_MEM0_V1
    )
    judge = cfg.get("judge") or freeze.get("judge") or {}
    if isinstance(judge, str):
        judge = {"api_model_id": judge, "provider": "openai"}
    subset = cfg.get("subset") or {}
    catalog = cfg.get("_model_catalog") or {}
    shared = cfg.get("shared_indexes") or {}
    agent_comparison = dict(cfg.get("agent_comparison") or {})

    if exp_type == SANDWICH:
        matrix = _apply_sandwich_freeze(matrix, freeze)
    if exp_type == AGENT:
        matrix = _apply_agent_freeze(matrix, freeze)

    axes = _axis_values(matrix, freeze, catalog)
    keys = [key for key in MATRIX_AXIS_ORDER if key in axes]
    extra = sorted(k for k in axes if k not in keys)
    keys.extend(extra)
    value_lists = [axes[k] for k in keys]
    combos = list(product(*value_lists)) if value_lists else [()]

    writer_axis = "writer" in axes
    specs: list[ExperimentRunSpec] = []
    for combo in combos:
        cell = dict(zip(keys, combo))
        reader = cell["reader"]
        memory_method = str(cell["memory_method"])
        if is_removed_multi_model_method(memory_method):
            raise ValueError(
                f"Experiment matrix includes removed multi-model method {memory_method!r}. "
                "Only one writer model is supported."
            )
        writer = cell.get("writer")
        if writer_axis and _memory_uses_writer(memory_method) != (
            writer is not None
        ):
            # full_context × a writer model is not a frozen-reader claim.
            continue
        seed = int(cell.get("seed") or 1)
        thinking_flag = cell.get("thinking")
        if thinking_flag is not None:
            reader, writer = _apply_thinking_axis(reader, writer, bool(thinking_flag))
        agent_id = _normalize_agent(cell.get("agent"))
        agent_persist = cell.get("agent_persist")
        agent_sessions = cell.get("agent_sessions")
        agent_tools = cell.get("agent_tools")
        agent_prompt_mode = _normalize_agent_prompt_mode(
            cell.get("agent_prompt_mode")
        )
        if "agent" in axes and not _agent_combo_allowed(
            agent_id,
            memory_method,
            agent_persist,
            agent_tools,
            agent_sessions,
            agent_prompt_mode,
        ):
            continue
        if agent_id:
            validate_strict_contract(
                agent_comparison,
                adapter=agent_id,
                model_snapshot=reader.model_snapshot,
            )
        cell_prompt = _prompt_for_agent_cell(
            agent_id,
            memory_method,
            prompt_path,
            persist=agent_persist,
            sessions=agent_sessions,
            prompt_mode=agent_prompt_mode,
        ) if "agent" in axes else prompt_path
        indexes = index_run_ids_for_memory(memory_method, shared)
        status = _cell_status(reader, writer)
        run_index = len(specs)
        spec = ExperimentRunSpec(
            experiment_name=name,
            experiment_type=exp_type,
            run_index=run_index,
            run_id="",  # filled after identity is known
            benchmark=benchmark,
            memory_method=memory_method,
            reader=reader,
            seed=seed,
            prompt_path=cell_prompt,
            judge_provider=str(judge.get("provider") or "openai"),
            judge_model=str(judge.get("api_model_id") or judge.get("model") or "gpt-4o-mini"),
            writer=writer,
            max_questions=_as_int(subset.get("max_questions")),
            max_samples=_as_int(subset.get("max_samples")),
            sample_id=_as_str(subset.get("sample_id")),
            method_yaml=method_yaml_for(memory_method),
            mem0_index_run_id=indexes.get("mem0"),
            rag_index_run_id=indexes.get("rag"),
            status=status,
            agent=agent_id,
            agent_persist=(
                False if agent_persist is None else bool(agent_persist)
            )
            if agent_id
            else None,
            agent_sessions=(
                _normalize_sessions(agent_sessions) if agent_id else None
            ),
            agent_tools=(str(agent_tools or "native") if agent_id else None),
            agent_prompt_mode=agent_prompt_mode if agent_id else None,
            agent_comparison=agent_comparison if agent_id else {},
        )
        run_id = hashed_run_id(name, spec.qa_identity())
        specs.append(
            ExperimentRunSpec(
                **{**spec.__dict__, "run_id": run_id, "run_index": run_index}
            )
        )
    return specs


def _apply_sandwich_freeze(
    matrix: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    out = dict(matrix)
    if "reader" in out and _len_axis(out["reader"]) > 1:
        raise ValueError(
            "frozen-reader experiments (type sandwich) freeze the reader; put one reader in "
            "experiment.freeze.reader, not a list under matrix.reader"
        )
    frozen = freeze.get("reader")
    if frozen is None:
        raise ValueError("frozen-reader experiments (type sandwich) require experiment.freeze.reader")
    out["reader"] = [frozen]
    return out


def _apply_agent_freeze(
    matrix: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    """Agent experiments freeze the model inside the harness, as a frozen reader does."""
    out = dict(matrix)
    if "reader" in out and _len_axis(out["reader"]) > 1:
        raise ValueError(
            "agent experiments freeze the harness model; put one reader in "
            "experiment.freeze.reader, not a list under matrix.reader"
        )
    frozen = freeze.get("reader")
    if frozen is None:
        raise ValueError("agent experiments require experiment.freeze.reader")
    out["reader"] = [frozen]
    return out


def _normalize_agent(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in ("", "none", "null", "false", "off"):
        return None
    return key


def _agent_combo_allowed(
    agent_id: str | None,
    memory_method: str,
    persist: Any,
    tools: Any,
    sessions: Any = None,
    prompt_mode: str = "workspace",
) -> bool:
    """Keep workspace retrieval distinct from exact reader-prompt parity."""
    persist_on = persist is True or persist == "on"
    sessions_key = _normalize_sessions(sessions)
    mode = _normalize_agent_prompt_mode(prompt_mode)
    if agent_id is None:
        if memory_method == "workspace_files":
            return False
        if persist_on:
            return False
        if sessions_key == "notes_only":
            return False
        tools_key = None if tools is None else str(tools).strip().lower()
        if tools_key not in (None, "native"):
            return False
        return True
    if mode == "workspace" and memory_method != "workspace_files":
        return False
    if mode == "reader_prompt" and memory_method not in (
        "full_context",
        "session_summaries",
    ):
        return False
    if sessions_key == "notes_only" and not persist_on:
        return False
    return True


def _normalize_sessions(value: Any) -> str:
    if value is None:
        return "full"
    key = str(value).strip().lower()
    if key in ("", "full", "on", "true"):
        return "full"
    if key in ("notes_only", "notes-only", "hidden"):
        return "notes_only"
    return "full"


def _normalize_agent_prompt_mode(value: Any) -> str:
    """Name whether Codex retrieves files or receives the reader's payload."""
    key = str(value or "workspace").strip().lower()
    if key in ("workspace", "workspace_files"):
        return "workspace"
    if key in ("reader_prompt", "reader-prompt", "prompt_parity"):
        return "reader_prompt"
    raise ValueError(
        f"agent_prompt_mode must be workspace or reader_prompt, got {value!r}"
    )


def _prompt_for_agent_cell(
    agent_id: str | None,
    memory_method: str,
    freeze_prompt: str,
    persist: Any = None,
    sessions: Any = None,
    prompt_mode: str = "workspace",
) -> str:
    if agent_id is None:
        return QA_MEM0_V1
    if _normalize_agent_prompt_mode(prompt_mode) == "reader_prompt":
        return freeze_prompt
    if memory_method != "workspace_files":
        return freeze_prompt
    persist_on = persist is True or persist == "on"
    if _normalize_sessions(sessions) == "notes_only":
        return QA_WORKSPACE_NOTES_ONLY_V1
    if persist_on:
        return QA_WORKSPACE_PERSIST_V1
    return QA_WORKSPACE_V1


def _axis_values(
    matrix: dict[str, Any],
    freeze: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, list[Any]]:
    axes: dict[str, list[Any]] = {}
    raw_readers = matrix.get("reader")
    if raw_readers is None:
        raw_readers = [freeze.get("reader")] if freeze.get("reader") else None
    if raw_readers is None:
        raise ValueError("matrix.reader or experiment.freeze.reader is required")
    axes["reader"] = [_parse_reader(item, catalog) for item in _as_list(raw_readers)]

    memories = matrix.get("memory_method") or matrix.get("memory_methods")
    if not memories:
        raise ValueError("matrix.memory_method is required")
    axes["memory_method"] = [str(m) for m in _as_list(memories)]

    if matrix.get("writer") is not None:
        axes["writer"] = [
            _parse_writer(item, catalog) for item in _as_list(matrix.get("writer"))
        ]
    seeds = matrix.get("seed") or matrix.get("seeds")
    if seeds is not None:
        axes["seed"] = [int(s) for s in _as_list(seeds)]
    else:
        axes["seed"] = [1]
    if matrix.get("thinking") is not None:
        axes["thinking"] = [_as_bool(v) for v in _as_list(matrix.get("thinking"))]
    if matrix.get("agent") is not None:
        axes["agent"] = [_normalize_agent(v) or "none" for v in _as_list(matrix.get("agent"))]
    if matrix.get("agent_persist") is not None:
        axes["agent_persist"] = [
            _as_bool(v) for v in _as_list(matrix.get("agent_persist"))
        ]
    if matrix.get("agent_sessions") is not None:
        axes["agent_sessions"] = [
            _normalize_sessions(v) for v in _as_list(matrix.get("agent_sessions"))
        ]
    if matrix.get("agent_tools") is not None:
        axes["agent_tools"] = [str(v).strip().lower() for v in _as_list(matrix.get("agent_tools"))]
    if matrix.get("agent_prompt_mode") is not None:
        axes["agent_prompt_mode"] = [
            _normalize_agent_prompt_mode(v)
            for v in _as_list(matrix.get("agent_prompt_mode"))
        ]
    return axes


def _apply_thinking_axis(
    reader: ReaderModelRef,
    writer: WriterModelRef | None,
    thinking: bool,
) -> tuple[ReaderModelRef, WriterModelRef | None]:
    """Matched on/off + headroom. Sandwich applies to the writer; sweeps to the reader."""
    if writer is not None:
        return reader, replace(
            writer,
            thinking=thinking,
            max_tokens=THINKING_OUTPUT_TOKENS[("writer", thinking)],
        )
    return (
        replace(
            reader,
            thinking=thinking,
            max_tokens=THINKING_OUTPUT_TOKENS[("reader", thinking)],
        ),
        writer,
    )


def _parse_reader(item: Any, catalog: dict[str, Any]) -> ReaderModelRef:
    if isinstance(item, str):
        item = {"catalog": item}
    if not isinstance(item, dict):
        raise ValueError(f"reader assignment must be a mapping or catalog id, got {item!r}")
    data = dict(item)
    if "catalog" in data:
        data = _merge_catalog(str(data["catalog"]), catalog, data)
    api_id = str(data.get("api_model_id") or data.get("model") or "")
    if not api_id:
        raise ValueError(f"reader is missing api_model_id: {item!r}")
    return ReaderModelRef(
        display_name=str(data.get("display_name") or api_id),
        provider=str(data.get("provider") or "openai"),
        family=str(data.get("family") or data.get("provider") or "openai"),
        generation=data.get("generation"),
        api_model_id=api_id,
        model_snapshot=_as_str(data.get("model_snapshot")),
        catalog_id=_as_str(data.get("catalog") or data.get("id")),
        status=str(data.get("status") or "runnable"),
    )


def _memory_uses_writer(memory_method: str) -> bool:
    """True when matrix.writer is the live writer (not a shared Mem0/RAG dump)."""
    key = str(memory_method)
    return key in {
        "graph",
        "session_summaries",
        "agent_codex_mem0_facts",
    } or is_removed_multi_model_method(key)


def _parse_writer(item: Any, catalog: dict[str, Any]) -> WriterModelRef | None:
    if item in (None, "none", False):
        return None
    if isinstance(item, str):
        item = {"catalog": item}
    data = dict(item)
    if "catalog" in data:
        data = _merge_catalog(str(data["catalog"]), catalog, data)
    api_id = str(data.get("api_model_id") or data.get("model") or "")
    if not api_id:
        raise ValueError(f"writer is missing api_model_id: {item!r}")
    thinking, max_tokens = _writer_knobs(data)
    return WriterModelRef(
        display_name=str(data.get("display_name") or api_id),
        provider=str(data.get("provider") or "openai"),
        api_model_id=api_id,
        catalog_id=_as_str(data.get("catalog") or data.get("id")),
        status=str(data.get("status") or "runnable"),
        thinking=thinking,
        max_tokens=max_tokens,
    )


def _writer_knobs(data: dict[str, Any]) -> tuple[bool | None, int | None]:
    """Optional write-path overlay from catalog ``writer:`` (GPT-5/GPT-6)."""
    block = data.get("writer") if isinstance(data.get("writer"), dict) else {}
    thinking = block.get("thinking")
    if thinking is None:
        thinking = data.get("thinking")
    max_tokens = block.get("max_tokens")
    if max_tokens is None:
        max_tokens = data.get("writer_max_tokens")
    think_flag = None
    if thinking is not None:
        think_flag = _as_bool(thinking)
    n = None if max_tokens is None or max_tokens == "" else int(max_tokens)
    return think_flag, n


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid thinking {value!r}; use on/off or true/false")


def _merge_catalog(key: str, catalog: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    if key not in catalog:
        raise ValueError(f"Unknown model catalog id {key!r}")
    merged = dict(catalog[key])
    for field, value in overlay.items():
        if field == "catalog":
            continue
        if value is not None:
            merged[field] = value
    merged["catalog"] = key
    return merged


def _cell_status(reader: ReaderModelRef, writer: WriterModelRef | None) -> str:
    statuses = [reader.status]
    if writer is not None:
        statuses.append(writer.status)
    if any(s != "runnable" for s in statuses):
        return "to_confirm"
    return "runnable"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _len_axis(value: Any) -> int:
    return len(_as_list(value))


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _as_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
