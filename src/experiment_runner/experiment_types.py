"""Named experiment designs the harness can expand.

``type: frozen_reader`` holds the reader fixed and varies the memory system.
Agent eval freezes the harness model and varies adapter, persist, or tools.
The harness does not assume a frozen reader unless the YAML says
``type: frozen_reader`` or ``type: agent``.
"""

from __future__ import annotations

FROZEN_READER = "frozen_reader"
SWEEP = "sweep"
ABLATION = "ablation"
CALIBRATION = "calibration"
AGENT = "agent"

KNOWN_TYPES = (FROZEN_READER, SWEEP, ABLATION, CALIBRATION, AGENT)

# Cartesian axes, in this order, so expansion is stable across processes.
MATRIX_AXIS_ORDER = (
    "reader",
    "memory_method",
    "writer",
    "thinking",
    "agent",
    "agent_persist",
    "agent_sessions",
    "agent_tools",
    "agent_prompt_mode",
    "seed",
)


def require_known_type(name: str) -> str:
    key = str(name).strip().lower()
    if key not in KNOWN_TYPES:
        raise ValueError(
            f"Unknown experiment type {name!r}. Use one of: {', '.join(KNOWN_TYPES)}"
        )
    return key
