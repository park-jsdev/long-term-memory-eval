"""Deterministic preprocess steps: segment, stamp ids, normalize speakers/times.

Order is fixed. The diagram's three boxes are not parallel services.
"""

from __future__ import annotations

import re
from datetime import datetime

from ..schemas import (
    Conversation,
    ProcessedConversation,
    ProcessedTurn,
    Session,
    SessionBlock,
)

# LoCoMo release uses "1:56 pm on 8 May, 2023"; fixtures use "1 Jan 2023".
_DATE_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p, %d %B, %Y",
    "%d %B, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%d %b, %Y",
)

_AMPM = re.compile(r"\b([AaPp])[Mm]\b")


def parse_session_datetime(raw: str) -> str | None:
    """Best-effort ISO-8601. Unparseable strings return None; raw is always kept elsewhere."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text_ampm = _AMPM.sub(lambda m: m.group(1).upper() + "M", text)
    for fmt in _DATE_FORMATS:
        for candidate in (text, text_ampm):
            try:
                dt = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            if "%I" in fmt or "%H" in fmt:
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
            return dt.strftime("%Y-%m-%d")
    return None


def _speaker_role(speaker_raw: str, speaker_a: str, speaker_b: str) -> str:
    name = speaker_raw.strip().casefold()
    if name and name == speaker_a.strip().casefold():
        return "a"
    if name and name == speaker_b.strip().casefold():
        return "b"
    return "other"


def segment_sessions(conversation: Conversation) -> list[SessionBlock]:
    """Split on LoCoMo session_N keys. Empty sessions are dropped, not re-cut."""
    blocks: list[SessionBlock] = []
    session_index = 0
    for session in conversation.sessions:
        if not session.turns:
            continue
        blocks.append(_block_from_session(conversation, session, session_index))
        session_index += 1
    return blocks


def _block_from_session(
    conversation: Conversation, session: Session, session_index: int
) -> SessionBlock:
    turns = [
        ProcessedTurn(
            turn_id="",
            source_dia_id=turn.dia_id,
            turn_index=-1,
            speaker_raw=turn.speaker,
            speaker_role="",
            text=turn.text,
            blip_caption=turn.blip_caption,
        )
        for turn in session.turns
    ]
    return SessionBlock(
        sample_id=conversation.sample_id,
        session_id=session.session_id,
        session_index=session_index,
        source_key=f"session_{session.session_id}",
        date_time_raw=session.date_time,
        date_time_normalized=None,
        speaker_a=conversation.speaker_a,
        speaker_b=conversation.speaker_b,
        turns=turns,
    )


def assign_turn_ids(blocks: list[SessionBlock]) -> list[SessionBlock]:
    """Stamp stable turn_id / turn_index. Does not change session order."""
    out: list[SessionBlock] = []
    for block in blocks:
        stamped: list[ProcessedTurn] = []
        for i, turn in enumerate(block.turns):
            stamped.append(
                ProcessedTurn(
                    turn_id=f"{block.sample_id}:s{block.session_id}:t{i:03d}",
                    source_dia_id=turn.source_dia_id,
                    turn_index=i,
                    speaker_raw=turn.speaker_raw,
                    speaker_role=turn.speaker_role,
                    text=turn.text,
                    blip_caption=turn.blip_caption,
                )
            )
        out.append(
            SessionBlock(
                sample_id=block.sample_id,
                session_id=block.session_id,
                session_index=block.session_index,
                source_key=block.source_key,
                date_time_raw=block.date_time_raw,
                date_time_normalized=block.date_time_normalized,
                speaker_a=block.speaker_a,
                speaker_b=block.speaker_b,
                turns=stamped,
                schema_version=block.schema_version,
            )
        )
    return out


def normalize_speakers_and_times(blocks: list[SessionBlock]) -> list[SessionBlock]:
    """Map speakers to a/b/other; ISO-8601 dates when parseable."""
    out: list[SessionBlock] = []
    for block in blocks:
        turns = [
            ProcessedTurn(
                turn_id=turn.turn_id,
                source_dia_id=turn.source_dia_id,
                turn_index=turn.turn_index,
                speaker_raw=turn.speaker_raw,
                speaker_role=_speaker_role(
                    turn.speaker_raw, block.speaker_a, block.speaker_b
                ),
                text=turn.text,
                blip_caption=turn.blip_caption,
            )
            for turn in block.turns
        ]
        out.append(
            SessionBlock(
                sample_id=block.sample_id,
                session_id=block.session_id,
                session_index=block.session_index,
                source_key=block.source_key,
                date_time_raw=block.date_time_raw,
                date_time_normalized=parse_session_datetime(block.date_time_raw),
                speaker_a=block.speaker_a,
                speaker_b=block.speaker_b,
                turns=turns,
                schema_version=block.schema_version,
            )
        )
    return out


class PreprocessingPipeline:
    """HLD (i): Conversation → ProcessedConversation with recorded session blocks."""

    def process(self, conversation: Conversation) -> ProcessedConversation:
        blocks = segment_sessions(conversation)
        blocks = assign_turn_ids(blocks)
        blocks = normalize_speakers_and_times(blocks)
        return ProcessedConversation(
            sample_id=conversation.sample_id,
            speaker_a=conversation.speaker_a,
            speaker_b=conversation.speaker_b,
            session_blocks=blocks,
            question_ids=[q.question_id for q in conversation.questions],
        )
