"""Map a memory-method id to the existing locomo_eval YAML (not a second experiment language)."""

from __future__ import annotations

from src.locomo_eval.eval_pipeline import METHOD_CONFIGS

_RESOLVE = {
    "workspace_files": "configs/writers/workspace_files.yaml",
    "agent_codex_mem0_facts": "configs/writers/agent_codex_mem0_facts.yaml",
}


def method_yaml_for(memory_method: str) -> str:
    key = str(memory_method).strip()
    path = METHOD_CONFIGS.get(key) or _RESOLVE.get(key)
    if not path:
        raise ValueError(
            f"No locomo_eval YAML for memory_method={key!r}. "
            f"Known: {sorted(set(METHOD_CONFIGS) | set(_RESOLVE))}"
        )
    return path
