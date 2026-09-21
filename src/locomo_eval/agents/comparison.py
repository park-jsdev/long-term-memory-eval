"""Versioned, canonical controls for comparable agent-harness runs.

The contract records what can be pinned across CLI adapters.  It deliberately
marks vendor-owned system prompts and unsupported limits as non-comparable
instead of silently treating them as equal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "agent_comparison.v1"
NOT_APPLICABLE = "not_applicable"


def canonical_json(value: Any) -> str:
    """Stable serialization used for the comparison contract identity."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def contract_sha256(contract: dict[str, Any]) -> str:
    """Hash a contract excluding its self-referential hash field."""
    body = {key: value for key, value in contract.items() if key != "sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Content hash for a pinned prompt or snapshot file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workspace_manifest_sha256(workspace: Path) -> str:
    """Hash relative paths and bytes so every adapter sees the same corpus."""
    digest = hashlib.sha256()
    for path in sorted(p for p in workspace.rglob("*") if p.is_file()):
        # POSIX relative paths so Windows vs POSIX dumps hash equal.
        digest.update(path.relative_to(workspace).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def adapter_capabilities(adapter: str) -> dict[str, bool]:
    """Hard limits an adapter can enforce without trusting its task prompt."""
    key = str(adapter or "").strip().lower()
    if key == "mock":
        return {
            "allowed_tools": True,
            "max_tool_calls": True,
            "max_retrieved_tokens": True,
            "wall_clock_s": True,
            "workspace_preflight": True,
        }
    # Codex's CLI exposes a wall-clock timeout to this wrapper, but does not
    # offer a reliable per-tool-call hook or a common retrieval service.
    if key == "codex":
        return {
            "allowed_tools": False,
            "max_tool_calls": False,
            "max_retrieved_tokens": False,
            "wall_clock_s": True,
            "workspace_preflight": True,
        }
    return {
        "allowed_tools": False,
        "max_tool_calls": False,
        "max_retrieved_tokens": False,
        "wall_clock_s": False,
        "workspace_preflight": False,
    }


def validate_strict_contract(
    contract_cfg: dict[str, Any],
    *,
    adapter: str | None,
    model_snapshot: str | None,
) -> None:
    """Reject a strict cell before it is assigned a reproducible run id."""
    if not contract_cfg or not bool(contract_cfg.get("strict")):
        return
    if not adapter:
        raise ValueError("strict agent comparison requires a harness adapter")
    snapshot = str(model_snapshot or "").strip()
    if not snapshot or snapshot == "TO_CONFIRM":
        raise ValueError(
            "strict agent comparison requires a pinned immutable model_snapshot"
        )
    budget = contract_cfg.get("tool_budget") or {}
    capabilities = adapter_capabilities(adapter)
    required = {
        "allowed_tools": bool(budget.get("allowed_tools")),
        "max_tool_calls": budget.get("max_tool_calls") is not None,
        "max_retrieved_tokens": budget.get("max_retrieved_tokens") is not None,
        "wall_clock_s": budget.get("wall_clock_s") is not None,
    }
    missing = [name for name, enabled in required.items() if enabled and not capabilities[name]]
    if missing:
        raise ValueError(
            f"strict agent comparison cannot use adapter {adapter!r}: "
            f"it cannot enforce {', '.join(missing)}"
        )


def resolve_contract(
    *,
    config: dict[str, Any],
    adapter: str | None,
    model: str | None,
    model_snapshot: str | None,
    prompt_path: Path,
    memory_type: str,
) -> dict[str, Any]:
    """Build the on-disk comparison contract for one QA run."""
    agent = config.get("agent") or {}
    source = dict(agent.get("comparison") or {})
    budget = dict(source.get("tool_budget") or agent.get("budget") or {})
    retrieval = dict(source.get("retrieval") or {})
    memory_write = dict(source.get("memory_write") or {})
    if memory_type == "workspace_files":
        retrieval.setdefault("mode", "agent_file_search")
        retrieval.setdefault("embedding_model", NOT_APPLICABLE)
        retrieval.setdefault("k", NOT_APPLICABLE)
        memory_write.setdefault("method", NOT_APPLICABLE)
        memory_write.setdefault("model", NOT_APPLICABLE)
        memory_write.setdefault("index_sha256", NOT_APPLICABLE)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "strict": bool(source.get("strict", False)),
        "status": "pending",
        "backbone": {
            "adapter": adapter or NOT_APPLICABLE,
            "model": model or NOT_APPLICABLE,
            "model_snapshot": model_snapshot or NOT_APPLICABLE,
            "adapter_version": source.get("adapter_version", "unknown"),
        },
        "task_prompt": {
            "path": str(prompt_path),
            "sha256": file_sha256(prompt_path),
            "adapter_system_prompt": source.get("adapter_system_prompt", "unknown"),
        },
        "context": {
            "max_available_tokens": source.get("context", {}).get("max_available_tokens"),
            "max_input_tokens": source.get("context", {}).get("max_input_tokens"),
            "workspace_manifest_sha256": None,
        },
        "retrieval": retrieval,
        "memory_write": memory_write,
        "judge": dict(source.get("judge") or {}),
        "tool_budget": {
            "allowed_tools": list(budget.get("allowed_tools") or []),
            "max_tool_calls": budget.get("max_tool_calls"),
            "max_retrieved_tokens": budget.get("max_retrieved_tokens"),
            "wall_clock_s": budget.get("wall_clock_s"),
        },
        "capabilities": adapter_capabilities(adapter or ""),
    }
    contract["sha256"] = contract_sha256(contract)
    return contract


def comparison_status(
    contract: dict[str, Any],
    *,
    harness_failed: bool,
) -> str:
    """Map one run onto comparable / incomparable / harness_failed."""
    if harness_failed:
        return "harness_failed"
    if not contract.get("strict"):
        return "incomparable"
    return "comparable"
