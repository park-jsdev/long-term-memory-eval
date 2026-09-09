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
GPT41_MINI = "gpt-4.1-mini"
GPT4O = "gpt-4o"
GPT4O_MINI = "gpt-4o-mini"
BASELINE_READER_MODEL = GPT4O_MINI
GPT56_LUNA = "gpt-5.6-luna"
GPT56_TERRA = "gpt-5.6-terra"
GPT56_SOL = "gpt-5.6-sol"
CLAUDE_HAIKU_45 = "claude-haiku-4-5"
DEEPSEEK_V4_FLASH = "deepseek-v4-flash"

# Cheap teacher ids for multi-teacher plumbing (not the frozen reader).
SANITY_TEACHER_OPENAI = GPT4O_MINI
SANITY_TEACHER_ANTHROPIC = CLAUDE_HAIKU_45
SANITY_TEACHER_DEEPSEEK = DEEPSEEK_V4_FLASH

FAMILY_GPT41 = "gpt-4.1"
FAMILY_GPT4O = "gpt-4o"
FAMILY_GPT56 = "gpt-5.6"
FAMILY_CLAUDE_HAIKU = "claude-haiku"
FAMILY_CLAUDE = "claude"
FAMILY_DEEPSEEK_V4 = "deepseek-v4"
FAMILY_DEEPSEEK = "deepseek"

# Mem0 paper/released evaluation defaults for both answering and judging.
DEFAULT_AUTORATER_MODEL = GPT4O_MINI


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
    GPT41_MINI: ModelSpec(
        model_id=GPT41_MINI,
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
    GPT4O: ModelSpec(
        model_id=GPT4O,
        family=FAMILY_GPT4O,
        display_name="GPT-4o",
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
    CLAUDE_HAIKU_45: ModelSpec(
        model_id=CLAUDE_HAIKU_45,
        family=FAMILY_CLAUDE_HAIKU,
        display_name="Claude Haiku 4.5",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
    DEEPSEEK_V4_FLASH: ModelSpec(
        model_id=DEEPSEEK_V4_FLASH,
        family=FAMILY_DEEPSEEK_V4,
        display_name="DeepSeek V4 Flash",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
    "deepseek-chat": ModelSpec(
        model_id="deepseek-chat",
        family=FAMILY_DEEPSEEK,
        display_name="DeepSeek Chat",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
    "deepseek-reasoner": ModelSpec(
        model_id="deepseek-reasoner",
        family=FAMILY_DEEPSEEK,
        display_name="DeepSeek Reasoner",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
    "claude-3-5-haiku-latest": ModelSpec(
        model_id="claude-3-5-haiku-latest",
        family=FAMILY_CLAUDE,
        display_name="Claude 3.5 Haiku",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
    "claude-3-5-sonnet-latest": ModelSpec(
        model_id="claude-3-5-sonnet-latest",
        family=FAMILY_CLAUDE,
        display_name="Claude 3.5 Sonnet",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
    "claude-sonnet-4-5": ModelSpec(
        model_id="claude-sonnet-4-5",
        family=FAMILY_CLAUDE,
        display_name="Claude Sonnet 4.5",
        max_tokens_field="max_tokens",
        supports_temperature=True,
    ),
}

# Short names for CLI / tests. Keys are lowercase.
_ALIASES: dict[str, str] = {
    "luna": GPT56_LUNA,
    "gpt-5.6-luna": GPT56_LUNA,
    "terra": GPT56_TERRA,
    "sol": GPT56_SOL,
    "gpt-5.6": GPT56_SOL,  # OpenAI alias routes gpt-5.6 → sol
    "mini": GPT41_MINI,
    "baseline": BASELINE_READER_MODEL,
    "gpt-4.1-mini": GPT41_MINI,
    "gpt-4o": GPT4O,
    "4o": GPT4O,
    "gpt-4o-mini": GPT4O_MINI,
    "4o-mini": GPT4O_MINI,
    "haiku": CLAUDE_HAIKU_45,
    "claude-haiku": CLAUDE_HAIKU_45,
    "claude-haiku-4-5": CLAUDE_HAIKU_45,
    "claude-haiku-4-5-20251001": CLAUDE_HAIKU_45,
    "claude": "claude-3-5-haiku-latest",
    "claude-sonnet": "claude-3-5-sonnet-latest",
    "deepseek": DEEPSEEK_V4_FLASH,
    "deepseek-v4-flash": DEEPSEEK_V4_FLASH,
    "deepseek-chat": "deepseek-chat",
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
    if mid.startswith("claude"):
        if "haiku-4-5" in mid or mid == "claude-haiku-4-5":
            return FAMILY_CLAUDE_HAIKU
        if "haiku" in mid:
            return FAMILY_CLAUDE
        if "sonnet" in mid:
            return "claude-sonnet"
        if "opus" in mid:
            return "claude-opus"
        return FAMILY_CLAUDE
    if mid.startswith("deepseek"):
        return FAMILY_DEEPSEEK_V4 if "v4" in mid else FAMILY_DEEPSEEK
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


# Write-path overlay only. The frozen GPT-5.6 *reader* stays catalog `none`.
TEACHER_THINKING_EFFORT = "high"
TEACHER_THINKING_MIN_OUTPUT_TOKENS = 1024


def supports_reasoning_effort(spec: ModelSpec) -> bool:
    """True for GPT-5.x / o-series. gpt-4o-mini has no reasoning_effort knob."""
    if spec.reasoning_effort is not None:
        return True
    family = (spec.family or "").lower()
    mid = (spec.model_id or "").lower()
    return (
        family.startswith("gpt-5")
        or family.startswith("o1")
        or family.startswith("o3")
        or mid.startswith("o1")
        or mid.startswith("o3")
    )


def apply_openai_thinking(
    spec: ModelSpec,
    kwargs: dict[str, Any],
    thinking: bool | None,
) -> dict[str, Any]:
    """Overlay teacher thinking onto Chat Completions kwargs.

    ``None`` leaves the catalog pin (reader: GPT-5.6 ``reasoning_effort=none``).
    ``True`` sets teacher effort on models that support it; ``False`` forces
    ``none``. gpt-4o-mini is a no-op either way.
    """
    out = dict(kwargs)
    if thinking is None or not supports_reasoning_effort(spec):
        return out
    out["reasoning_effort"] = TEACHER_THINKING_EFFORT if thinking else "none"
    if thinking:
        field = spec.max_tokens_field
        n = int(out.get(field) or 16)
        out[field] = max(n, TEACHER_THINKING_MIN_OUTPUT_TOKENS)
    return out


def chat_create_kwargs(
    spec: ModelSpec,
    *,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Kwargs for ``client.chat.completions.create`` for this model."""
    kwargs: dict[str, Any] = {
        "model": spec.model_id,
        "messages": messages,
    }
    if max_tokens is not None:
        n = int(max_tokens)
        if spec.max_tokens_field == "max_completion_tokens":
            n = max(n, 16)  # GPT-5.6 Luna rejects values below 16
        kwargs[spec.max_tokens_field] = n
    if spec.supports_temperature:
        kwargs["temperature"] = float(temperature)
    if spec.reasoning_effort is not None:
        kwargs["reasoning_effort"] = spec.reasoning_effort
    return kwargs
