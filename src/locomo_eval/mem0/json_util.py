"""Parse JSON objects from LLM text (fences / leading prose)."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """First JSON object in ``text``. Raises ValueError if none / not a dict."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty LLM JSON.")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM JSON is not parseable: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON must be an object.")
    return obj
