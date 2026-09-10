"""Expand an experiment YAML into an ordered list of ``ExperimentRunSpec``.

The same YAML always yields the same order and the same hashed run ids.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from src.memorybench.experiment_run_spec import (
    ExperimentRunSpec,
    ReaderModelRef,
    WriterModelRef,
)
from src.memorybench.experiment_types import (
    MATRIX_AXIS_ORDER,
    SANDWICH,
    require_known_type,
)
from src.memorybench.hashed_run_id import hashed_run_id
from src.memorybench.locomo_method_yaml import method_yaml_for
from src.memorybench.shared_index_paths import index_run_ids_for_memory
from src.locomo_eval.prompts import QA_MEM0_V1


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

    if exp_type == SANDWICH:
        matrix = _apply_sandwich_freeze(matrix, freeze)

    axes = _axis_values(matrix, freeze, catalog)
    keys = [key for key in MATRIX_AXIS_ORDER if key in axes]
    extra = sorted(k for k in axes if k not in keys)
    keys.extend(extra)
    value_lists = [axes[k] for k in keys]
    combos = list(product(*value_lists)) if value_lists else [()]

    specs: list[ExperimentRunSpec] = []
    for run_index, combo in enumerate(combos):
        cell = dict(zip(keys, combo))
        reader = cell["reader"]
        memory_method = str(cell["memory_method"])
        writer = cell.get("writer")
        seed = int(cell.get("seed") or 1)
        indexes = index_run_ids_for_memory(memory_method, shared)
        status = _cell_status(reader, writer)
        spec = ExperimentRunSpec(
            experiment_name=name,
            experiment_type=exp_type,
            run_index=run_index,
            run_id="",  # filled after identity is known
            benchmark=benchmark,
            memory_method=memory_method,
            reader=reader,
            seed=seed,
            prompt_path=prompt_path,
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
            "sandwich experiments freeze the reader; put one reader in "
            "experiment.freeze.reader, not a list under matrix.reader"
        )
    frozen = freeze.get("reader")
    if frozen is None:
        raise ValueError("sandwich experiments require experiment.freeze.reader")
    out["reader"] = [frozen]
    return out


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
    return axes


def _parse_reader(item: Any, catalog: dict[str, Any]) -> ReaderModelRef:
    if isinstance(item, str):
        item = {"catalog": item}
    if not isinstance(item, dict):
        raise ValueError(f"reader cell must be a mapping or catalog id, got {item!r}")
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
    return WriterModelRef(
        display_name=str(data.get("display_name") or api_id),
        provider=str(data.get("provider") or "openai"),
        api_model_id=api_id,
        catalog_id=_as_str(data.get("catalog") or data.get("id")),
        status=str(data.get("status") or "runnable"),
    )


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
