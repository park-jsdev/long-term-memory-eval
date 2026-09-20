"""Agent-harness eval: workspace files + coding-agent adapters + trajectories."""

from __future__ import annotations

from typing import Any

from .adapters.codex import CodexAgentRunner, resolve_codex_bin
from .adapters.mock import MockAgentRunner
from .protocol import (
    KNOWN_ADAPTERS,
    STUB_ADAPTERS,
    AgentRequest,
    AgentResult,
    AgentRunner,
)
from .workspace import WorkspaceFilesMemoryBuilder, render_agent_prompt


def get_agent_runner(
    adapter: str,
    *,
    model: str = "gpt-5",
    persist_memory: bool = False,
    tools: str = "native",
    **kwargs: Any,
) -> AgentRunner:
    """Construct a harness adapter. Stub ids fail loudly until implemented."""
    key = str(adapter or "").strip().lower()
    if key in ("", "none"):
        raise ValueError("get_agent_runner requires an adapter id (mock, codex, …)")
    if key == "mock":
        return MockAgentRunner(model_name=model or "mock")
    if key == "codex":
        return CodexAgentRunner(
            model_name=model,
            persist_memory=persist_memory,
            tools=tools,
            **{k: v for k, v in kwargs.items() if k in ("timeout_s", "codex_bin", "extra_args")},
        )
    if key in STUB_ADAPTERS:
        raise NotImplementedError(
            f"{key} adapter is stubbed. Codex is the first implemented harness; "
            f"add src/locomo_eval/agents/adapters/{key}.py when wiring it."
        )
    raise ValueError(
        f"Unknown agent adapter {adapter!r}. Known: {sorted(KNOWN_ADAPTERS)}"
    )


__all__ = [
    "AgentRequest",
    "AgentResult",
    "AgentRunner",
    "WorkspaceFilesMemoryBuilder",
    "get_agent_runner",
    "render_agent_prompt",
    "resolve_codex_bin",
]
