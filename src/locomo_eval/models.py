"""Known answer/teacher model ids, families, and Chat Completions kwargs.

The sandwich still treats the *reader* as frozen for raw_chunks vs session_summaries claims.
This catalog exists so a *separate* axis can swap models within a family
(cross-model robustness) without scattering API quirks through run.py.

Unknown ids pass through with family inferred from the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Canonical ids used in configs, logs, and tests.
BASELINE_READER_MODEL = "gpt-4.1-mini"
GPT4O_MINI = "gpt-4o-mini"
GPT56_LUNA = "gpt-5.6-luna"
GPT56_TERRA = "gpt-5.6-terra"
GPT56_SOL = "gpt-5.6-sol"

FAMILY_GPT41 = "gpt-4.1"
FAMILY_GPT4O = "gpt-4o"
FAMILY_GPT56 = "gpt-5.6"


@dataclass(frozen=True)
class ModelSpec:
    """What it is: one API model id plus the request shape it needs.

    Who consumes it: OpenAIChatCaller (readers/teachers) and run_meta logging.
    """

    model_id: str
    family: str
    display_name: str
    # GPT-5.x rejects legacy max_tokens; GPT-4.1 still accepts it.
    max_tokens_field: str
    supports_temperature: bool
    reasoning_effort: str | None = None


_CATALOG: dict[str, ModelSpec] = {
    BASELINE_READER_MODEL: ModelSpec(
        model_id=BASELINE_READER_MODEL,
        family=FAMILY_GPT41,
        display_name="GPT-4.1 mini",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
    "gpt-4.1": ModelSpec(
        model_id="gpt-4.1",
        family=FAMILY_GPT41,
        display_name="GPT-4.1",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
    GPT4O_MINI: ModelSpec(
        model_id=GPT4O_MINI,
        family=FAMILY_GPT4O,
        display_name="GPT-4o mini",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
    GPT56_LUNA: ModelSpec(
        model_id=GPT56_LUNA,
        family=FAMILY_GPT56,
        display_name="GPT-5.6 Luna",
        max_tokens_field="max_completion_tokens",
        supports_temperature=False,
        reasoning_effort="none",
    ),
    GPT56_TERRA: ModelSpec(
        model_id=GPT56_TERRA,
        family=FAMILY_GPT56,
        display_name="GPT-5.6 Terra",
        max_tokens_field="max_completion_tokens",
        supports_temperature=False,
        reasoning_effort="none",
    ),
    GPT56_SOL: ModelSpec(
        model_id=GPT56_SOL,
        family=FAMILY_GPT56,
        display_name="GPT-5.6 Sol",
        max_tokens_field="max_completion_tokens",
        supports_temperature=False,
        reasoning_effort="none",
    ),
}

# Short names for CLI / tests. Keys are lowercase.
_ALIASES: dict[str, str] = {
    "luna": GPT56_LUNA,
    "gpt-5.6-luna": GPT56_LUNA,
    "terra": GPT56_TERRA,
    "sol": GPT56_SOL,
    "gpt-5.6": GPT56_SOL,  # OpenAI alias routes gpt-5.6 → sol
    "mini": BASELINE_READER_MODEL,
    "baseline": BASELINE_READER_MODEL,
    "gpt-4.1-mini": BASELINE_READER_MODEL,
    "gpt-4o-mini": GPT4O_MINI,
    "4o-mini": GPT4O_MINI,
}


def infer_family(model_id: str) -> str:
    """Prefix-based family when the id is not in the catalog."""
    mid = model_id.strip().lower()
    if mid.startswith("gpt-5.6"):
        return FAMILY_GPT56
    if mid.startswith("gpt-4.1"):
        return FAMILY_GPT41
    if mid.startswith("gpt-4o"):
        return "gpt-4o"
    if mid.startswith("gpt-5"):
        return "gpt-5"
    return model_id


def resolve_model(name: str) -> ModelSpec:
    """Map alias or id → ModelSpec. Unknown ids pass through with inferred family."""
    if not name or not str(name).strip():
        raise ValueError("Model name is empty.")
    raw = str(name).strip()
    key = _ALIASES.get(raw.lower(), raw)
    if key in _CATALOG:
        return _CATALOG[key]
    family = infer_family(key)
    gpt5 = family.startswith("gpt-5")
    return ModelSpec(
        model_id=key,
        family=family,
        display_name=key,
        max_tokens_field="max_completion_tokens" if gpt5 else "max_tokens",
        supports_temperature=not gpt5,
        reasoning_effort="none" if gpt5 else None,
    )


def same_family(a: str, b: str) -> bool:
    return resolve_model(a).family == resolve_model(b).family


def models_in_family(family: str) -> list[str]:
    return [spec.model_id for spec in _CATALOG.values() if spec.family == family]


def chat_create_kwargs(
    spec: ModelSpec,
    *,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Kwargs for ``client.chat.completions.create`` for this model."""
    n = int(max_tokens)
    if spec.max_tokens_field == "max_completion_tokens":
        n = max(n, 16)  # GPT-5.6 Luna rejects values below 16
    kwargs: dict[str, Any] = {
        "model": spec.model_id,
        "messages": messages,
        spec.max_tokens_field: n,
    }
    if spec.supports_temperature:
        kwargs["temperature"] = float(temperature)
    if spec.reasoning_effort is not None:
        kwargs["reasoning_effort"] = spec.reasoning_effort
    return kwargs
