"""LoCoMo → Mem0 message pairs (eval add.py ingest protocol).

Does not re-cut sessions. Pairs overlay SessionBlock.turns:
  batch_size=2 consecutive turns, dual speaker indexes, role-flip,
  speaker name prefixed into content, timestamp = date_time_raw.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..schemas import ProcessedConversation, SessionBlock
from .schemas import MessagePair, RoleMessage


def user_messages_text(pair: MessagePair) -> str:
    """User-role lines only — extract must not see assistant text."""
    lines = [m.content for m in pair.messages if m.role == "user"]
    return "\n".join(lines).strip()


def _role_for(speaker_raw: str, index_speaker: str) -> str:
    if speaker_raw.strip().casefold() == index_speaker.strip().casefold():
        return "user"
    return "assistant"


def pairs_for_session_block(
    block: SessionBlock,
    *,
    speaker_index: str,
    speaker_name: str,
    batch_size: int = 2,
) -> list[MessagePair]:
    """Consecutive turns in ``batch_size`` (last remainder is a singleton)."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    turns = list(block.turns)
    out: list[MessagePair] = []
    pair_n = 0
    for start in range(0, len(turns), batch_size):
        chunk = turns[start : start + batch_size]
        messages = [
            RoleMessage(
                role=_role_for(turn.speaker_raw, speaker_name),
                content=f"{turn.speaker_raw.strip()}: {turn.text}",
                speaker=turn.speaker_raw,
                turn_id=turn.turn_id,
                source_dia_id=turn.source_dia_id,
            )
            for turn in chunk
        ]
        turn_ids = [t.turn_id for t in chunk]
        out.append(
            MessagePair(
                pair_id=(
                    f"{block.sample_id}:s{block.session_id}:"
                    f"p{pair_n:03d}:{speaker_index}"
                ),
                sample_id=block.sample_id,
                session_id=block.session_id,
                session_index=block.session_index,
                speaker_index=speaker_index,
                speaker_name=speaker_name,
                timestamp=block.date_time_raw,
                messages=messages,
                turn_ids=turn_ids,
            )
        )
        pair_n += 1
    return out


def iter_speaker_pairs(
    processed: ProcessedConversation,
    *,
    batch_size: int = 2,
    max_sessions: int | None = None,
) -> Iterator[MessagePair]:
    """Yield A-index pairs then B-index pairs, session order inside each index.

    Eval add.py walks sessions once and role-flips into two user_ids. Order
    within an index is session order; we finish speaker a then speaker b so
    dumps are deterministic.
    """
    blocks = list(processed.session_blocks)
    if max_sessions is not None:
        blocks = blocks[: int(max_sessions)]
    for speaker_index, speaker_name in (
        ("a", processed.speaker_a),
        ("b", processed.speaker_b),
    ):
        for block in blocks:
            yield from pairs_for_session_block(
                block,
                speaker_index=speaker_index,
                speaker_name=speaker_name,
                batch_size=batch_size,
            )
