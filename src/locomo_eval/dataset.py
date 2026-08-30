"""Load locomo10.json into Conversation / Question objects (pipeline step 1).

Does not modify the source file. Source data stay in data/raw/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schemas import Conversation, Question, Session, Turn

_SESSION_KEY = re.compile(r"^session_(\d+)$")
_SUMMARY_KEY = re.compile(r"^session_(\d+)_summary$")
_OBS_KEY = re.compile(r"^session_(\d+)_observation$")


def load_raw(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of samples in {path}")
    return data


def _parse_sessions(conversation: dict[str, Any]) -> list[Session]:
    sessions: list[Session] = []
    for key, value in conversation.items():
        m = _SESSION_KEY.match(key)
        if not m or not isinstance(value, list):
            continue
        sid = int(m.group(1))
        date_time = str(conversation.get(f"session_{sid}_date_time", ""))
        turns = []
        for t in value:
            turns.append(
                Turn(
                    dia_id=str(t.get("dia_id", "")),
                    speaker=str(t.get("speaker", "")),
                    text=str(t.get("text", "")),
                    blip_caption=t.get("blip_caption"),
                )
            )
        sessions.append(Session(session_id=sid, date_time=date_time, turns=turns))
    sessions.sort(key=lambda s: s.session_id)
    return sessions


def _parse_summaries(block: dict[str, Any] | None) -> dict[int, str]:
    out: dict[int, str] = {}
    if not isinstance(block, dict):
        return out
    for key, value in block.items():
        m = _SUMMARY_KEY.match(key)
        if m:
            out[int(m.group(1))] = str(value)
    return out


def _parse_observations(block: dict[str, Any] | None) -> dict[int, Any]:
    out: dict[int, Any] = {}
    if not isinstance(block, dict):
        return out
    for key, value in block.items():
        m = _OBS_KEY.match(key)
        if m:
            out[int(m.group(1))] = value
    return out


def parse_sample(sample: dict[str, Any]) -> Conversation:
    sample_id = str(sample["sample_id"])
    conv = sample.get("conversation") or {}
    speaker_a = str(conv.get("speaker_a", "A"))
    speaker_b = str(conv.get("speaker_b", "B"))
    sessions = _parse_sessions(conv)

    questions: list[Question] = []
    for i, qa in enumerate(sample.get("qa") or []):
        questions.append(
            Question(
                sample_id=sample_id,
                question_id=f"{sample_id}-q-{i}",
                question=str(qa["question"]),
                answer=str(qa.get("answer", "")),
                category=int(qa["category"]),
                evidence=list(qa.get("evidence") or []),
                qa_index=i,
            )
        )

    return Conversation(
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=sessions,
        session_summaries=_parse_summaries(sample.get("session_summary")),
        observations=_parse_observations(sample.get("observation")),
        questions=questions,
    )


def load_conversations(path: str | Path) -> list[Conversation]:
    return [parse_sample(s) for s in load_raw(path)]


def iter_questions(conversations: list[Conversation]):
    for conv in conversations:
        for q in conv.questions:
            yield conv, q


QUESTION_SAMPLE_ROUND_ROBIN = "round_robin"
QUESTION_SAMPLE_PREFIX = "prefix"
QUESTION_SAMPLE_MODES = (QUESTION_SAMPLE_ROUND_ROBIN, QUESTION_SAMPLE_PREFIX)


def select_question_pairs(
    pairs: list[tuple[Conversation, Question]],
    max_questions: int | None,
    *,
    mode: str = QUESTION_SAMPLE_ROUND_ROBIN,
) -> list[tuple[Conversation, Question]]:
    """Cap QA pairs for a smoke/subset run.

    ``round_robin`` (default): first question of each conversation, then the
    second of each, and so on. With 10 LoCoMo samples and ``max_questions=10``
    that is one question per conversation.

    ``prefix``: first N pairs in file order (old behavior; can sit entirely
    inside sample 1).
    """
    if max_questions is None:
        return list(pairs)
    n = int(max_questions)
    if n <= 0:
        return []
    key = (mode or QUESTION_SAMPLE_ROUND_ROBIN).strip().lower()
    if key == QUESTION_SAMPLE_PREFIX:
        return list(pairs[:n])
    if key != QUESTION_SAMPLE_ROUND_ROBIN:
        raise ValueError(
            f"Unknown question sample mode '{mode}'. Known: {QUESTION_SAMPLE_MODES}"
        )
    buckets: dict[str, list[tuple[Conversation, Question]]] = {}
    order: list[str] = []
    for conv, question in pairs:
        sid = conv.sample_id
        if sid not in buckets:
            buckets[sid] = []
            order.append(sid)
        buckets[sid].append((conv, question))
    out: list[tuple[Conversation, Question]] = []
    depth = 0
    while len(out) < n:
        progressed = False
        for sid in order:
            bucket = buckets[sid]
            if depth < len(bucket):
                out.append(bucket[depth])
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
        depth += 1
    return out
