"""Parse harness events into a retrieval trajectory and score evidence hits.

Gold evidence ids (LoCoMo ``dia_id`` strings on ``Question.evidence``) are
matched as substrings of retrieved text. Catalog reads (INDEX.md) are
navigation, not evidence. Token counts use tiktoken cl100k when installed,
else whitespace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .protocol import (
    EVENT_CATALOG,
    EVENT_ERROR,
    EVENT_OTHER,
    EVENT_RETRIEVE,
    FAILURE_HARNESS,
    FAILURE_NONE,
    FAILURE_PARAMETRIC,
    FAILURE_REASONING,
    FAILURE_RETRIEVAL,
    RetrievalEvent,
    RetrievalTrajectory,
)
from .workspace import INDEX_NAME

_RETRIEVE_CMDS = re.compile(
    r"\b(cat|head|tail|less|more|rg|grep|findstr|find|type|Get-Content|"
    r"Select-String|sed|awk|nl)\b",
    re.I,
)
_CATALOG_HINTS = re.compile(r"(INDEX\.md|\bls\b|\bdir\b|\bglob\b)", re.I)


def count_retrieved_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        from src.locomo_eval.rag.chunk import count_tokens

        return int(count_tokens(text))
    except Exception:
        return len(text.split())


def evidence_ids_in_text(text: str, evidence_ids: list[str]) -> list[str]:
    hits: list[str] = []
    blob = text or ""
    for eid in evidence_ids:
        token = str(eid or "").strip()
        if token and token in blob and token not in hits:
            hits.append(token)
    return hits


def classify_event(tool: str, target: str, retrieved_text: str) -> str:
    if str(tool or "").lower() in ("error", "reasoning"):
        return EVENT_ERROR
    hay = f"{tool} {target} {retrieved_text[:200]}"
    if INDEX_NAME.lower() in hay.lower() or (
        _CATALOG_HINTS.search(target or "") and INDEX_NAME.lower() in hay.lower()
    ):
        return EVENT_CATALOG
    if (target or "").replace("\\", "/").rstrip("/").endswith("/" + INDEX_NAME.lower()):
        return EVENT_CATALOG
    if INDEX_NAME in (target or ""):
        return EVENT_CATALOG
    if tool in ("read", "grep", "glob", "search") or _RETRIEVE_CMDS.search(target or ""):
        return EVENT_RETRIEVE
    if tool in ("shell", "command", "command_execution") and _RETRIEVE_CMDS.search(
        target or ""
    ):
        return EVENT_RETRIEVE
    if tool in ("shell", "command", "command_execution") and _CATALOG_HINTS.search(
        target or ""
    ):
        return EVENT_CATALOG
    if tool in ("web_search", "mcp"):
        return EVENT_OTHER
    if tool:
        # Unknown items are tool telemetry, not evidence retrieval.  Counting
        # them as retrieval hid failed Codex calls in the first live smoke.
        return EVENT_OTHER
    return EVENT_OTHER


def build_trajectory(
    events: list[RetrievalEvent],
    *,
    evidence_ids: list[str],
    usage: dict[str, Any] | None = None,
) -> RetrievalTrajectory:
    """Join tool events to gold evidence ids and compute recall/precision."""
    usage = usage or {}
    required = [str(e).strip() for e in evidence_ids if str(e).strip()]
    retrieve_events = [e for e in events if e.kind == EVENT_RETRIEVE]
    hit_ids: list[str] = []
    for event in events:
        found = evidence_ids_in_text(event.retrieved_text, required)
        event.evidence_ids_hit = found
        for eid in found:
            if eid not in hit_ids:
                hit_ids.append(eid)
    useful = [e for e in retrieve_events if e.evidence_ids_hit]
    unnecessary = len(retrieve_events) - len(useful)
    recall = None
    precision = None
    evidence_retrieved: bool | None = None
    if required:
        recall = len(hit_ids) / len(required)
        evidence_retrieved = len(hit_ids) == len(required)
        if retrieve_events:
            precision = len(useful) / len(retrieve_events)
        else:
            precision = 0.0
    prompt, completion, reasoning = _usage_parts(usage)
    return RetrievalTrajectory(
        n_retrieval_calls=len(retrieve_events),
        retrieved_tokens=sum(int(e.retrieved_tokens or 0) for e in retrieve_events),
        input_tokens=prompt,
        output_tokens=completion,
        reasoning_tokens=reasoning,
        evidence_retrieved=evidence_retrieved,
        evidence_ids_required=required,
        evidence_ids_hit=hit_ids,
        memory_recall=recall,
        memory_precision=precision,
        unnecessary_retrievals=max(0, unnecessary),
        events=events,
    )


def failure_mode(
    *,
    correct: bool,
    evidence_retrieved: bool | None,
    harness_failed: bool = False,
) -> str:
    """Split wrong answers into retrieval vs reasoning failure.

    ``parametric_success`` is a correct answer that never retrieved required
    evidence (lucky / parametric). ``none`` is correct with evidence in hand
    or when the item has no gold evidence ids.
    """
    if harness_failed:
        return FAILURE_HARNESS
    if correct:
        if evidence_retrieved is False:
            return FAILURE_PARAMETRIC
        return FAILURE_NONE
    if evidence_retrieved is False:
        return FAILURE_RETRIEVAL
    if evidence_retrieved is True:
        return FAILURE_REASONING
    return FAILURE_REASONING


def parse_codex_jsonl(lines: list[str] | str) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Return (raw events, usage, last agent message) from ``codex exec --json``."""
    if isinstance(lines, str):
        rows = [ln for ln in lines.splitlines() if ln.strip()]
    else:
        rows = list(lines)
    events: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    last_message = ""
    for line in rows:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        events.append(obj)
        item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
        typ = str(obj.get("type") or "")
        if typ == "turn.completed" and isinstance(obj.get("usage"), dict):
            usage = normalize_usage(obj["usage"])
        text = item.get("text") or item.get("aggregated_output") or item.get("output")
        if item.get("type") == "agent_message" and text:
            last_message = str(text)
        if typ == "item.completed" and item.get("type") == "agent_message" and text:
            last_message = str(text)
    return events, usage, last_message


def events_from_codex_raw(raw: list[dict[str, Any]]) -> list[RetrievalEvent]:
    """Lift Codex JSONL items into ``RetrievalEvent`` rows (completed items only)."""
    out: list[RetrievalEvent] = []
    step = 0
    seen: set[str] = set()
    for obj in raw:
        item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
        typ = str(obj.get("type") or "")
        if typ not in ("item.completed", "item.started"):
            continue
        item_id = str(item.get("id") or "")
        item_type = str(item.get("type") or "")
        if item_type == "agent_message":
            continue
        if typ == "item.started":
            continue
        if item_id and item_id in seen:
            continue
        if item_id:
            seen.add(item_id)
        tool, target, text = _codex_item_fields(item)
        if not tool and not target:
            continue
        step += 1
        kind = classify_event(tool, target, text)
        out.append(
            RetrievalEvent(
                step=step,
                kind=kind,
                tool=tool,
                target=target,
                retrieved_text=text,
                retrieved_tokens=count_retrieved_tokens(text),
            )
        )
    return out


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Map Codex / OpenAI / Anthropic usage onto prompt/completion/total."""
    usage = dict(usage or {})
    prompt = int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or 0
    )
    completion = int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0
    )
    reasoning = int(
        usage.get("reasoning_tokens")
        or usage.get("reasoning_output_tokens")
        or 0
    )
    cached = int(usage.get("cached_input_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "cached_input_tokens": cached,
        "input_tokens": prompt,
        "output_tokens": completion,
    }


def parse_structured_answer(text: str) -> str:
    """Best-effort extract ``answer`` from Codex ``--output-schema`` JSON."""
    blob = (text or "").strip()
    if not blob:
        return ""
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict) and obj.get("answer") is not None:
            return str(obj.get("answer")).strip()
    except json.JSONDecodeError:
        pass
    start = blob.find("{")
    end = blob.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(blob[start : end + 1])
            if isinstance(obj, dict) and obj.get("answer") is not None:
                return str(obj.get("answer")).strip()
        except json.JSONDecodeError:
            pass
    return blob


def load_answer_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return parse_structured_answer(path.read_text(encoding="utf-8"))


def _codex_item_fields(item: dict[str, Any]) -> tuple[str, str, str]:
    item_type = str(item.get("type") or "")
    command = str(item.get("command") or item.get("cmd") or "")
    output = str(
        item.get("aggregated_output")
        or item.get("output")
        or item.get("content")
        or item.get("text")
        or ""
    )
    if item_type in ("command_execution", "command"):
        return "shell", command, output
    if item_type in ("mcp_tool_call", "mcp"):
        name = str(item.get("name") or item.get("tool") or "mcp")
        return "mcp", name, output
    if item_type in ("web_search", "web_search_call"):
        q = str(item.get("query") or item.get("text") or "")
        return "web_search", q, output
    if item_type:
        target = str(item.get("path") or item.get("query") or command or item_type)
        return item_type, target, output
    return "", "", ""


def _usage_parts(usage: dict[str, Any]) -> tuple[int, int, int]:
    norm = normalize_usage(usage)
    return (
        int(norm.get("prompt_tokens") or 0),
        int(norm.get("completion_tokens") or 0),
        int(norm.get("reasoning_tokens") or 0),
    )
