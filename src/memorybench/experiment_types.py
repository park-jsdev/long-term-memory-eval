"""Named experiment designs the harness can expand.

Sandwich is one design, not the only one. The harness does not assume a frozen
reader unless the YAML says ``type: sandwich``.
"""

from __future__ import annotations

SANDWICH = "sandwich"
SWEEP = "sweep"
ABLATION = "ablation"
CALIBRATION = "calibration"

KNOWN_TYPES = (SANDWICH, SWEEP, ABLATION, CALIBRATION)

# Cartesian axes, in this order, so expansion is stable across processes.
MATRIX_AXIS_ORDER = (
    "reader",
    "memory_method",
    "writer",
    "thinking",
    "seed",
)


def require_known_type(name: str) -> str:
    key = str(name).strip().lower()
    if key not in KNOWN_TYPES:
        raise ValueError(
            f"Unknown experiment type {name!r}. Use one of: {', '.join(KNOWN_TYPES)}"
        )
    return key
