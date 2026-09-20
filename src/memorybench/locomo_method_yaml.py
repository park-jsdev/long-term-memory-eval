"""Map a memory-method id to the existing locomo_eval YAML (not a second experiment language)."""

from __future__ import annotations

from src.locomo_eval.eval_pipeline import METHOD_CONFIGS

_RESOLVE = {
    "fused_teacher_graph_resolve_top_voted": (
        "configs/writers/fused_teacher_graph_resolve_top_voted.yaml"
    ),
    "fused_teacher_graph_resolve_first": (
        "configs/writers/fused_teacher_graph_resolve_first.yaml"
    ),
    "fused_teacher_graph_resolve_random": (
        "configs/writers/fused_teacher_graph_resolve_random.yaml"
    ),
    "fused_teacher_graph_resolve_round_robin": (
        "configs/writers/fused_teacher_graph_resolve_round_robin.yaml"
    ),
    "fused_teacher_graph_resolve_confidence": (
        "configs/writers/fused_teacher_graph_resolve_confidence.yaml"
    ),
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
