"""HLD (ii) teacher orchestration — thin passthrough seam.

Iterates one SessionBlock at a time so later batch teachers can attribute
the same session/turn ids preprocess recorded.

This file is not a write/ package. No LLM, no fusion, no LlmResponseHash.
Promote when teachers actually generate memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .schemas import ProcessedConversation, SessionBlock


def iter_session_blocks(processed: ProcessedConversation) -> Iterator[SessionBlock]:
    """Yield session blocks in preprocess order (one block at a time)."""
    yield from processed.session_blocks


def process_session_block(block: SessionBlock) -> dict[str, Any]:
    """Identity record for one session. Downstream can swap this for a teacher call."""
    return {
        "sample_id": block.sample_id,
        "session_id": block.session_id,
        "session_index": block.session_index,
        "source_key": block.source_key,
        "n_turns": len(block.turns),
        "turn_ids": [t.turn_id for t in block.turns],
        "status": "passthrough",
    }


class TeacherOrchestrator:
    """Software controller stub: walk session blocks, do not call an LLM."""

    def iter_session_blocks(self, processed: ProcessedConversation) -> Iterator[SessionBlock]:
        return iter_session_blocks(processed)

    def process_session_block(self, block: SessionBlock) -> dict[str, Any]:
        return process_session_block(block)
