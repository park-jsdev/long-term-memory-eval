"""Load the deterministic preprocess dump and format memory strings.

Write path is ``run_index`` (no LLM). Retrieve/format is a separate seam:
  * ``all`` (default) — concatenate every session unit (current sanity)
  * ``top_k`` — naive lexical rank over session documents (later cosine)

``raw_chunks`` formats speaker turns from SessionBlocks.
``session_summaries`` formats dataset ``session_summary`` from SessionDocuments.
Swap this ranker later without re-parsing locomo10.json.
"""

from __future__ import annotations

from pathlib import Path

from ..memory import format_session_turns
from ..schemas import Conversation, Question, Session, SessionBlock, Turn
from .dump import load_session_blocks, load_session_documents, require_sample_dump
from .session_documents import MemoryField, SessionDocument, naive_rank_session_ids


def session_from_block(block: SessionBlock) -> Session:
    """SessionBlock → dataset Session so raw_chunks layout stays one formatter."""
    turns = [
        Turn(
            dia_id=turn.source_dia_id,
            speaker=turn.speaker_raw,
            text=turn.text,
            blip_caption=turn.blip_caption,
        )
        for turn in block.turns
    ]
    return Session(
        session_id=block.session_id, date_time=block.date_time_raw, turns=turns
    )


def conversation_from_dump(
    index_dir: str | Path,
    conversation: Conversation,
) -> Conversation:
    """Replace sessions/summaries from the dump; keep gold questions on Conversation."""
    sample_dir = require_sample_dump(Path(index_dir), conversation.sample_id)
    blocks = load_session_blocks(sample_dir)
    docs = load_session_documents(sample_dir)
    speaker_a = blocks[0].speaker_a if blocks else conversation.speaker_a
    speaker_b = blocks[0].speaker_b if blocks else conversation.speaker_b
    summaries = {
        doc.session_id: doc.session_summary
        for doc in docs
        if doc.session_summary and str(doc.session_summary).strip()
    }
    return Conversation(
        sample_id=conversation.sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=[session_from_block(b) for b in blocks],
        session_summaries=summaries,
        observations=conversation.observations,
        questions=conversation.questions,
    )


def select_documents(
    documents: list[SessionDocument],
    query: str,
    field: MemoryField,
    top_k: int | None,
) -> list[SessionDocument]:
    """Chronological all units, or naive lexical top-k. Does not use gold answers."""
    if top_k is None:
        return list(documents)
    ranked_ids = naive_rank_session_ids(query, documents, field)
    by_sid = {doc.session_id: doc for doc in documents}
    picked: list[SessionDocument] = []
    for sid in ranked_ids[: max(0, int(top_k))]:
        doc = by_sid.get(sid)
        if doc is not None:
            picked.append(doc)
    return picked


def format_raw_chunks_from_blocks(
    blocks: list[SessionBlock],
    *,
    speaker_a: str,
    speaker_b: str,
    max_chars: int | None = None,
) -> tuple[str, list[str]]:
    chunks: list[str] = [
        f"Conversation between {speaker_a} and {speaker_b}.\n",
    ]
    source_ids: list[str] = []
    for block in blocks:
        session = session_from_block(block)
        if not session.turns:
            continue
        block_lines = [
            f"DATE: {session.date_time}",
            f"SESSION {session.session_id}:",
            format_session_turns(session),
        ]
        for turn in session.turns:
            source_ids.append(turn.dia_id or f"session_{session.session_id}")
        chunks.append("\n".join(block_lines))
    text = "\n\n".join(chunks).strip()
    if not text:
        text = "(No dialogue turns available for this conversation.)"
    if max_chars is not None and len(text) > max_chars:
        text = (
            "[... earlier turns truncated to max_chars ...]\n" + text[-max_chars:]
        )
    return text, source_ids


def format_session_summaries_from_documents(
    documents: list[SessionDocument],
) -> tuple[str, list[str]]:
    chunks: list[str] = []
    source_ids: list[str] = []
    for doc in documents:
        summary = (doc.session_summary or "").strip()
        if not summary:
            continue
        chunks.append(f"[Session {doc.session_id}]\n{summary}")
        source_ids.append(f"session_{doc.session_id}_summary")
    if not chunks:
        text = "(No session summaries available for this conversation.)"
    else:
        text = "\n\n".join(chunks)
    return text, source_ids


def build_raw_chunks_from_index(
    index_dir: str | Path,
    conversation: Conversation,
    question: Question,
    *,
    max_chars: int | None = None,
    top_k: int | None = None,
) -> tuple[str, list[str]]:
    sample_dir = require_sample_dump(Path(index_dir), conversation.sample_id)
    blocks = load_session_blocks(sample_dir)
    docs = load_session_documents(sample_dir)
    if top_k is not None:
        keep = {d.session_id for d in select_documents(docs, question.question, "turns", top_k)}
        blocks = [b for b in blocks if b.session_id in keep]
    speaker_a = blocks[0].speaker_a if blocks else conversation.speaker_a
    speaker_b = blocks[0].speaker_b if blocks else conversation.speaker_b
    return format_raw_chunks_from_blocks(
        blocks, speaker_a=speaker_a, speaker_b=speaker_b, max_chars=max_chars
    )


def build_session_summaries_from_index(
    index_dir: str | Path,
    conversation: Conversation,
    question: Question,
    *,
    top_k: int | None = None,
) -> tuple[str, list[str]]:
    sample_dir = require_sample_dump(Path(index_dir), conversation.sample_id)
    docs = load_session_documents(sample_dir)
    docs = select_documents(docs, question.question, "summary", top_k)
    return format_session_summaries_from_documents(docs)
