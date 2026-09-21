"""Dump a LoCoMo conversation as session files for a coding-agent harness.

This is the agent-eval analogue of full_context: every turn is *available*
on disk, but the harness must retrieve it. Gold answers never enter these
files. Schema: ``docs/schemas/agent_runtime.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..memory import MemoryBuilder, format_session_turns
from ..schemas import Conversation, Memory, Question

INDEX_NAME = "INDEX.md"
SESSIONS_DIR = "sessions"
PERSIST_DIR = "memory"
PERSIST_NOTES = "memory/notes.md"
PERSIST_AGENTS = "AGENTS.md"


@dataclass(frozen=True)
class WorkspaceManifest:
    """Paths + Memory.text pointer for one conversation workspace."""

    sample_id: str
    root: Path | None
    index_rel: str
    session_rels: tuple[str, ...]
    source_ids: tuple[str, ...]
    text: str


def session_file_name(session_id: int) -> str:
    """Stable ``sessions/session_N.md`` basename used in INDEX.md links."""
    return f"session_{session_id}.md"


def write_conversation_workspace(
    conversation: Conversation,
    dest: Path | None,
    *,
    persist_memory: bool = False,
) -> WorkspaceManifest:
    """Write session markdown under ``dest`` (or format only when dest is None)."""
    session_rels: list[str] = []
    source_ids: list[str] = []
    index_lines = [
        f"# Conversation {conversation.sample_id}",
        "",
        f"Speakers: {conversation.speaker_a}, {conversation.speaker_b}",
        "",
        "Sessions:",
    ]
    session_bodies: list[tuple[str, str]] = []
    for session in conversation.sessions:
        rel = f"{SESSIONS_DIR}/{session_file_name(session.session_id)}"
        session_rels.append(rel)
        body_lines = [
            f"# Session {session.session_id}",
            f"Date: {session.date_time or 'unknown'}",
            "",
            format_session_turns(session) if session.turns else "(empty session)",
            "",
        ]
        session_bodies.append((rel, "\n".join(body_lines)))
        index_lines.append(
            f"- `{rel}` — {session.date_time or 'unknown'} "
            f"({len(session.turns)} turns)"
        )
        for turn in session.turns:
            source_ids.append(turn.dia_id or f"session_{session.session_id}")
    index_lines.extend(
        [
            "",
            "Each turn in a session file ends with `(dia_id)` when LoCoMo "
            "provided one. Answer questions using only these files.",
            "",
        ]
    )
    index_text = "\n".join(index_lines)
    if dest is not None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / INDEX_NAME).write_text(index_text, encoding="utf-8")
        sessions_dir = dest / SESSIONS_DIR
        sessions_dir.mkdir(parents=True, exist_ok=True)
        for rel, body in session_bodies:
            (dest / rel).write_text(body, encoding="utf-8")
        if persist_memory:
            (dest / PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            notes = dest / PERSIST_NOTES
            if not notes.is_file():
                notes.write_text(
                    "# Harness memory\n\nDurable notes for this conversation.\n",
                    encoding="utf-8",
                )
            agents_md = dest / PERSIST_AGENTS
            if not agents_md.is_file():
                agents_md.write_text(
                    "# Persistent memory is on for this workspace.\n"
                    "You may write notes under memory/.\n",
                    encoding="utf-8",
                )
    pointer_lines = [
        f"[workspace_files] conversation={conversation.sample_id}",
        f"index={INDEX_NAME}",
        "sessions:",
        *[f"  - {rel}" for rel in session_rels],
    ]
    if persist_memory:
        pointer_lines.append(f"persist={PERSIST_NOTES}")
    return WorkspaceManifest(
        sample_id=conversation.sample_id,
        root=dest,
        index_rel=INDEX_NAME,
        session_rels=tuple(session_rels),
        source_ids=tuple(source_ids),
        text="\n".join(pointer_lines),
    )


def render_agent_prompt(template: str, question: str) -> str:
    """Fill ``prompts/agents/qa_workspace_v1.txt``. Gold must not appear."""
    return template.replace("{question}", str(question))


class WorkspaceFilesMemoryBuilder(MemoryBuilder):
    """Sandwich middle for harness eval: files, not stuffed transcript text.

    ``Memory.text`` is a short manifest so claim-audit still has a pointer.
    The harness reads the files; the one-shot reader is not used.
    """

    name = "workspace_files"

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        persist_memory: bool = False,
    ):
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.persist_memory = bool(persist_memory)
        self.workspaces: dict[str, Path] = {}

    def build(self, conversation: Conversation, question: Question) -> Memory:
        dest = None
        if self.workspace_root is not None:
            dest = self.workspace_root / conversation.sample_id
        manifest = write_conversation_workspace(
            conversation, dest, persist_memory=self.persist_memory
        )
        if dest is not None:
            self.workspaces[conversation.sample_id] = dest
        return Memory(
            memory_type=self.name,
            text=manifest.text,
            source_ids=list(manifest.source_ids),
            search_latency_s=0.0,
        )
