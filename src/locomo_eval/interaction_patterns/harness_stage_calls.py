"""Model Stage 1 and Stage 2. Gold If/When/How/Then are not inputs.

The completer is any object with complete(messages) -> text. Tests pass
a fake. The live runner passes the Chat Completions caller. Stage 1
returns the same selection shape the unigram stub returns, so the
decision files stay one format.
"""

from __future__ import annotations

import json
from typing import Any

from .hypothesis_record import HypothesisRecord
from .hypothesis_rendering import render_record_for_decision

Stage1Result = tuple[list[HypothesisRecord], bool, str]


def stage1_messages(
    prompt: str,
    records: list[HypothesisRecord],
    request: str,
) -> list[dict[str, str]]:
    """Stage 1 sees record ids. It has to copy them into selected_ids."""
    body = "\n\n".join(render_record_for_decision(record) for record in records)
    return [
        {"role": "system", "content": prompt.strip()},
        {
            "role": "user",
            "content": f"Request:\n{request.strip()}\n\nRecords:\n{body}\n",
        },
    ]


def stage2_messages(
    prompt: str,
    selected: list[HypothesisRecord],
    request: str,
) -> list[dict[str, str]]:
    """Stage 2 sees only the records Stage 1 kept, still with ids.

    The ids are how this stage knows which rows it was given. The
    instruction it returns is stripped of those ids afterwards.
    """
    body = "\n\n".join(render_record_for_decision(record) for record in selected)
    return [
        {"role": "system", "content": prompt.strip()},
        {
            "role": "user",
            "content": (
                f"Request:\n{request.strip()}\n\n"
                f"Selected records:\n{body}\n"
            ),
        },
    ]


def parse_stage1_response(text: str, known_ids: list[str]) -> tuple[list[str], bool, str]:
    """Ids the model named, whether it abstained, and its reason.

    Unusable JSON abstains. An unknown id is dropped. Abstain wins over
    a non-empty id list so a confused object does not inject memory.
    """
    parsed = _json_object(text)
    if parsed is None:
        return [], True, "parse_error"
    known = set(known_ids)
    raw_ids = parsed.get("selected_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    ids = [str(record_id) for record_id in raw_ids if str(record_id) in known]
    abstain = _as_bool(parsed.get("abstain")) or not ids
    if abstain:
        ids = []
    reason = str(parsed.get("reason") or "model_stage1").strip() or "model_stage1"
    return ids, abstain, reason


def strip_record_ids(text: str, record_ids: list[str]) -> str:
    """Remove bookkeeping ids from the instruction the reply model sees.

    Longer ids go first so plus-101 is not left as plus-1 with a tail.
    """
    cleaned = text.strip()
    for record_id in sorted(record_ids, key=len, reverse=True):
        if record_id:
            cleaned = cleaned.replace(record_id, "")
    return cleaned.strip()


def run_harness_stages(
    completer: Any,
    records: list[HypothesisRecord],
    request: str,
    *,
    stage1_prompt: str,
    stage2_prompt: str,
) -> tuple[Stage1Result, str, dict[str, str]]:
    """One Stage 1 call, then Stage 2 only when Stage 1 selected.

    The trace keeps both prompts so a pack can be checked for gold
    decision text. Stage 2 is skipped on abstain because an instruction
    would invent a use the selector refused.
    """
    known = [record.record_id for record in records]
    stage1_prompt_text = stage1_messages(stage1_prompt, records, request)
    stage1_raw = complete_text(completer, stage1_prompt_text, json_mode=True)
    ids, abstain, reason = parse_stage1_response(stage1_raw, known)
    by_id = {record.record_id: record for record in records}
    selected = [by_id[record_id] for record_id in ids]
    stage2_raw = ""
    stage2_user = ""
    if not abstain and selected:
        stage2_prompt_text = stage2_messages(stage2_prompt, selected, request)
        stage2_user = stage2_prompt_text[-1]["content"]
        stage2_raw = complete_text(completer, stage2_prompt_text, json_mode=False)
        instruction = strip_record_ids(stage2_raw, known)
    else:
        instruction = ""
    trace = {
        "stage1_prompt": stage1_prompt_text[-1]["content"],
        "stage1_system": stage1_prompt,
        "stage1_raw": stage1_raw,
        "stage2_prompt": stage2_user,
        "stage2_raw": stage2_raw,
        "stage2_instruction": instruction,
    }
    return (selected, abstain, reason), instruction, trace


def complete_text(completer: Any, messages: list[dict[str, str]], *, json_mode: bool) -> str:
    """Accept a bare string or the (text, meta) pair the chat caller returns."""
    try:
        result = completer.complete(messages, json_mode=json_mode)
    except TypeError:
        result = completer.complete(messages)
    if isinstance(result, tuple):
        return str(result[0]).strip()
    return str(result).strip()


def _json_object(text: str) -> dict | None:
    body = text.strip()
    if body.startswith("```"):
        lines = [line for line in body.splitlines() if not line.strip().startswith("```")]
        body = "\n".join(lines).strip()
    parsed = _loads_object(body)
    if parsed is not None:
        return parsed
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return None
    return _loads_object(body[start : end + 1])


def _loads_object(body: str) -> dict | None:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)
