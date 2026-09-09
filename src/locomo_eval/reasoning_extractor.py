"""Extract visible reasoning / thinking text from provider message objects.

Shared by write-path teacher callers and the frozen answer reader (OpenAI
family). Kept separate from ``teacher_callers`` to avoid import cycles.
"""

from __future__ import annotations

from typing import Any


def openai_reasoning_text(message: Any) -> str:
    """Visible chain-of-thought from DeepSeek/OpenAI chat messages, if returned.

    GPT-5.x often hides the chain and only reports ``reasoning_tokens``.
    """
    text = getattr(message, "reasoning_content", None)
    if text:
        return str(text).strip()
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    if reasoning is not None and not isinstance(reasoning, str):
        for attr in ("content", "summary", "text"):
            val = getattr(reasoning, attr, None)
            if not val:
                continue
            if isinstance(val, list):
                return "\n".join(str(part) for part in val if part).strip()
            return str(val).strip()
    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        return str(extra.get("reasoning_content") or extra.get("reasoning") or "").strip()
    return ""


def openai_reasoning_tokens(usage: Any) -> int | None:
    """OpenAI ``completion_tokens_details.reasoning_tokens``, if present."""
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return None
    n = getattr(details, "reasoning_tokens", None)
    try:
        return int(n) if n is not None else None
    except (TypeError, ValueError):
        return None


def split_anthropic_content(content: Any) -> tuple[str, str]:
    """Return (visible_text, thinking_text) from Messages API content blocks."""
    texts: list[str] = []
    thoughts: list[str] = []
    for block in content or []:
        btype = getattr(block, "type", None)
        if btype == "thinking":
            thoughts.append(str(getattr(block, "thinking", "") or ""))
        elif btype == "redacted_thinking":
            thoughts.append("[redacted_thinking]")
        else:
            texts.append(str(getattr(block, "text", "") or ""))
    return (
        "\n".join(part for part in texts if part).strip(),
        "\n".join(part for part in thoughts if part).strip(),
    )
