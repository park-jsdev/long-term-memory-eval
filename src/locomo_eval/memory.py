"""Memory builders: conversation → Memory context for the answer LLM.

The memory stage. Readers and metrics stay independent of which system is built.

Current conditions (ids are what reviewers see in logs):
  raw_chunks                  — raw dialogue turns (optional char budget)
  session_summaries           — dataset summaries, or one writer model when writer is set
  graph                       — one writer model writes locked Mem0GraphMemory
  full_context                — entire timestamped dialog (Mem0 paper full-context)
  rag                         — token-chunk cosine retrieve from a RAG write-index
  openai_memory               — extract-all dump, retrieve-all (paper OpenAI protocol clone)
  mem0 / mem0g                — load a Mem0 write-index dump + cosine retrieve (no re-extract)
  workspace_files             — conversation as session files for a coding-agent harness

See docs/reports/engineering_notebook.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import Conversation, Memory, Question, Session

if TYPE_CHECKING:
    from .writer_model import Writer
    from .model_orchestrator import ModelOrchestrator
    from .mem0.embeddings import Embedder


class MemoryBuilder(ABC):
    """Memory stage: Conversation (+ Question) → Memory.

    Called from run_locomo_pipeline_with_memory_config (one YAML per call).
    Swap the subclass without touching the reader or metrics.
    """

    name: str = "base"

    @abstractmethod
    def build(self, conversation: Conversation, question: Question) -> Memory:
        ...


def format_session_turns(session: Session) -> str:
    """Speaker lines for one session. Shared by raw_chunks dump and teacher input."""
    lines: list[str] = []
    for turn in session.turns:
        line = f"{turn.speaker}: {turn.text}"
        if turn.blip_caption:
            line += f" [image: {turn.blip_caption}]"
        if turn.dia_id:
            line += f" ({turn.dia_id})"
        lines.append(line)
    return "\n".join(lines)


class RawConversationMemoryBuilder(MemoryBuilder):
    """Chronological raw turns, grouped into session chunks.

    Optional max_chars keeps very long conversations within a rough budget
    when the full dialog would blow the answer-model context. Truncation is
    from the *start* (keep recent sessions) so late evidence is more likely
    retained — simple heuristic, not true retrieval.
    """

    name = "raw_chunks"

    def __init__(
        self,
        max_chars: int | None = None,
        preprocess_index_dir: str | Path | None = None,
        retrieve_top_k: int | None = None,
    ):
        self.max_chars = max_chars
        self.preprocess_index_dir = (
            Path(preprocess_index_dir) if preprocess_index_dir else None
        )
        self.retrieve_top_k = retrieve_top_k
        self.retrieve_log: list[dict] = []

    def build(self, conversation: Conversation, question: Question) -> Memory:
        if self.preprocess_index_dir is not None:
            from .experiment_pack.claim_audit import retrieve_rank_row
            from .preprocess.retrieve import build_raw_chunks_from_index, rank_raw_chunks_from_index

            text, source_ids = build_raw_chunks_from_index(
                self.preprocess_index_dir,
                conversation,
                question,
                max_chars=self.max_chars,
                top_k=self.retrieve_top_k,
            )
            self.retrieve_log.append(
                retrieve_rank_row(
                    sample_id=conversation.sample_id,
                    question_id=question.question_id,
                    retriever="preprocess_turns",
                    top_k=self.retrieve_top_k,
                    candidates=rank_raw_chunks_from_index(
                        self.preprocess_index_dir,
                        conversation,
                        question,
                        top_k=self.retrieve_top_k,
                    ),
                )
            )
            return Memory(
                memory_type=self.name,
                text=text,
                source_ids=source_ids,
            )

        chunks: list[str] = []
        source_ids: list[str] = []
        header = (
            f"Conversation between {conversation.speaker_a} and "
            f"{conversation.speaker_b}.\n"
        )
        chunks.append(header)

        for session in conversation.sessions:
            if not session.turns:
                continue
            block_lines = [
                f"DATE: {session.date_time}",
                f"SESSION {session.session_id}:",
            ]
            for turn in session.turns:
                source_ids.append(turn.dia_id or f"session_{session.session_id}")
            block_lines.append(format_session_turns(session))
            chunks.append("\n".join(block_lines))

        text = "\n\n".join(chunks).strip()
        if not text:
            text = "(No dialogue turns available for this conversation.)"

        if self.max_chars is not None and len(text) > self.max_chars:
            # Keep the tail (more recent); mark truncation for audit.
            text = (
                "[... earlier turns truncated to max_chars ...]\n"
                + text[-self.max_chars :]
            )

        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
        )


class SessionSummaryMemoryBuilder(MemoryBuilder):
    """Concatenate LoCoMo session summaries chronologically.

    Dataset text when no writer is passed. A writer model uses
    WriterSessionMemoryBuilder and the same memory id.
    """

    name = "session_summaries"

    def __init__(
        self,
        preprocess_index_dir: str | Path | None = None,
        retrieve_top_k: int | None = None,
    ):
        self.preprocess_index_dir = (
            Path(preprocess_index_dir) if preprocess_index_dir else None
        )
        self.retrieve_top_k = retrieve_top_k
        self.retrieve_log: list[dict] = []

    def build(self, conversation: Conversation, question: Question) -> Memory:
        if self.preprocess_index_dir is not None:
            from .experiment_pack.claim_audit import retrieve_rank_row
            from .preprocess.retrieve import (
                build_session_summaries_from_index,
                rank_session_summaries_from_index,
            )

            text, source_ids = build_session_summaries_from_index(
                self.preprocess_index_dir,
                conversation,
                question,
                top_k=self.retrieve_top_k,
            )
            self.retrieve_log.append(
                retrieve_rank_row(
                    sample_id=conversation.sample_id,
                    question_id=question.question_id,
                    retriever="preprocess_summary",
                    top_k=self.retrieve_top_k,
                    candidates=rank_session_summaries_from_index(
                        self.preprocess_index_dir,
                        conversation,
                        question,
                        top_k=self.retrieve_top_k,
                    ),
                )
            )
            return Memory(
                memory_type=self.name,
                text=text,
                source_ids=source_ids,
            )

        chunks: list[str] = []
        source_ids: list[str] = []
        for sid, text in conversation.chronological_summaries():
            if not text or not str(text).strip():
                continue
            chunks.append(f"[Session {sid}]\n{text.strip()}")
            source_ids.append(f"session_{sid}_summary")

        if not chunks:
            text = "(No session summaries available for this conversation.)"
        else:
            text = "\n\n".join(chunks)

        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
        )


class WriterSessionMemoryBuilder(MemoryBuilder):
    """Per-session summaries written by one model.

    Same inject shape as dataset session summaries. The variable is the
    writer model. Gold answers never enter the writer prompt.
    """

    name = "session_summaries"

    def __init__(self, writer: Writer):
        self.writer = writer
        self.teacher_call_log: list[dict] = []
        self.session_texts: list[dict] = []

    @property
    def writer_model(self) -> str:
        return self.writer.model_name

    @property
    def writer_provider(self) -> str:
        return self.writer.provider

    def build(self, conversation: Conversation, question: Question) -> Memory:
        chunks: list[str] = []
        source_ids: list[str] = []
        for session in conversation.sessions:
            if not session.turns:
                continue
            session_text = format_session_turns(session)
            self.session_texts.append(
                {
                    "sample_id": conversation.sample_id,
                    "session_id": session.session_id,
                    "session_index": None,
                    "text": session_text,
                }
            )
            summary, meta = self.writer.summarize_session(
                session_text=session_text,
                date_time=session.date_time,
                speaker_a=conversation.speaker_a,
                speaker_b=conversation.speaker_b,
            )
            from .writer_model import writer_call_record

            self.teacher_call_log.append(
                writer_call_record(
                    writer=self.writer,
                    meta=meta,
                    sample_id=conversation.sample_id,
                    session_id=session.session_id,
                    output_text=summary,
                    session_text=session_text,
                )
            )
            summary = (summary or "").strip()
            if not summary:
                continue
            chunks.append(f"[Session {session.session_id}]\n{summary}")
            source_ids.append(f"session_{session.session_id}_summary")

        if not chunks:
            text = "(No session summaries available for this conversation.)"
        else:
            text = "\n\n".join(chunks)

        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
            writer_model=self.writer.model_name,
            writer_provider=self.writer.provider,
        )


class AgentFactMemoryBuilder(WriterSessionMemoryBuilder):
    """Codex-produced session fact lists, injected as a fixed reader memory."""

    name = "agent_codex_mem0_facts"


class OrchestratedGraphMemoryBuilder(MemoryBuilder):
    """Session blocks → ModelOrchestrator → locked Mem0GraphMemory → Memory.text.

    ``name`` is ``graph``. One writer model. Graph schema is Mem0g
    (nodes + labeled edges with valid=false invalidation).
    """

    def __init__(self, orchestrator: ModelOrchestrator, name: str):
        self.orchestrator = orchestrator
        self.name = name
        self.graphs_by_sample: dict = {}

    @property
    def writer_model(self) -> str | None:
        return self.orchestrator.writer_model

    @property
    def writer_provider(self) -> str | None:
        return self.orchestrator.writer_provider

    def build(self, conversation: Conversation, question: Question) -> Memory:
        from .preprocess.preprocessing_pipeline import PreprocessingPipeline
        from .model_orchestrator import format_graph_memory_text

        processed = PreprocessingPipeline().process(conversation)
        graph = self.orchestrator.build_graph(processed)
        self.graphs_by_sample[conversation.sample_id] = graph
        text = format_graph_memory_text(
            graph,
            speaker_a=conversation.speaker_a,
            speaker_b=conversation.speaker_b,
        )
        source_ids = [e.edge_id for e in graph.edges if e.valid]
        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
            writer_model=self.orchestrator.writer_model,
            writer_provider=self.orchestrator.writer_provider,
        )


GRAPH = "graph"
ORCHESTRATED_GRAPH_NAMES = (GRAPH,)

# Registry name → constructor kwargs factory
_BUILDERS: dict[str, type[MemoryBuilder]] = {
    RawConversationMemoryBuilder.name: RawConversationMemoryBuilder,
    SessionSummaryMemoryBuilder.name: SessionSummaryMemoryBuilder,
    AgentFactMemoryBuilder.name: AgentFactMemoryBuilder,
}

# Legacy / short names. Canonical ids are the builder ``name`` values above.
_ALIASES: dict[str, str] = {
    "c0": RawConversationMemoryBuilder.name,
    "c0_raw": RawConversationMemoryBuilder.name,
    "raw": RawConversationMemoryBuilder.name,
    "raw_dialog": RawConversationMemoryBuilder.name,
    "c1": SessionSummaryMemoryBuilder.name,
    "c1_session_summary": SessionSummaryMemoryBuilder.name,
    "session_summary": SessionSummaryMemoryBuilder.name,
    "mem0": "mem0",
    "mem0g": "mem0g",
    GRAPH: GRAPH,
    "rag": "rag",
    "full_context": "full_context",
    "full-context": "full_context",
    "openai_memory": "openai_memory",
    "openai": "openai_memory",
}

_INDEX_BUILDERS = ("mem0", "mem0g", "rag", "full_context", "openai_memory", "workspace_files")


def resolve_memory_name(name: str) -> str:
    key = name.strip().lower() if name else ""
    if key in _BUILDERS or key in _INDEX_BUILDERS or key in ORCHESTRATED_GRAPH_NAMES:
        return "full_context" if key in ("full_context", "full-context") else key
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"Unknown memory builder '{name}'. "
        f"Known: {sorted(list(_BUILDERS) + list(_INDEX_BUILDERS) + list(ORCHESTRATED_GRAPH_NAMES))} aliases={sorted(_ALIASES)}"
    )


def get_memory_builder(
    name: str,
    max_chars: int | None = None,
    writer: Writer | None = None,
    orchestrator: ModelOrchestrator | None = None,
    mem0_index_dir: str | Path | None = None,
    mem0_top_k: int = 30,
    mem0_embedder: Embedder | None = None,
    preprocess_index_dir: str | Path | None = None,
    retrieve_top_k: int | None = None,
    rag_index_dir: str | Path | None = None,
    rag_top_k: int = 2,
    rag_embedder: Embedder | None = None,
    openai_memory_index_dir: str | Path | None = None,
    agent_workspace_root: str | Path | None = None,
    agent_persist_memory: bool = False,
) -> MemoryBuilder:
    resolved = resolve_memory_name(name)
    if resolved == "workspace_files":
        from .agents.workspace import WorkspaceFilesMemoryBuilder

        return WorkspaceFilesMemoryBuilder(
            workspace_root=agent_workspace_root,
            persist_memory=agent_persist_memory,
        )
    if resolved == "full_context":
        from .rag.builders import FullContextMemoryBuilder

        return FullContextMemoryBuilder()
    if resolved == "rag":
        from .rag.builders import RagMemoryBuilder

        if rag_index_dir is None:
            raise ValueError(
                "rag requires a write-index directory. "
                "Set rag.index_run_id in YAML (experiments/<id>/rag_index) "
                "after python -m src.locomo_eval.rag.run_index."
            )
        return RagMemoryBuilder(
            index_dir=rag_index_dir,
            embedder=rag_embedder,
            top_k=rag_top_k,
        )
    if resolved == "openai_memory":
        from .openai_memory.builders import OpenAIMemoryBuilder

        if openai_memory_index_dir is None:
            raise ValueError(
                "openai_memory requires a write-index directory. "
                "Set openai_memory.index_run_id in YAML after "
                "python -m src.locomo_eval.openai_memory.run_index."
            )
        return OpenAIMemoryBuilder(index_dir=openai_memory_index_dir)
    if resolved in ("mem0", "mem0g"):
        from .mem0.builders import Mem0IndexMemoryBuilder, Mem0gIndexMemoryBuilder

        if mem0_index_dir is None:
            raise ValueError(
                f"{resolved} requires a write-index directory. "
                "Set mem0.index_run_id in YAML (experiments/<id>/mem0_index) "
                "after python -m src.locomo_eval.mem0.run_index."
            )
        cls = Mem0IndexMemoryBuilder if resolved == "mem0" else Mem0gIndexMemoryBuilder
        return cls(
            index_dir=mem0_index_dir,
            embedder=mem0_embedder,
            top_k=mem0_top_k,
        )
    if resolved in ORCHESTRATED_GRAPH_NAMES:
        if orchestrator is None:
            raise ValueError(
                f"{resolved} requires a ModelOrchestrator "
                "(set writer.model in YAML or --writer-model)."
            )
        return OrchestratedGraphMemoryBuilder(orchestrator=orchestrator, name=resolved)
    cls = _BUILDERS[resolved]
    if resolved == RawConversationMemoryBuilder.name:
        return cls(
            max_chars=max_chars,
            preprocess_index_dir=preprocess_index_dir,
            retrieve_top_k=retrieve_top_k,
        )
    if resolved == SessionSummaryMemoryBuilder.name:
        if writer is not None:
            return WriterSessionMemoryBuilder(writer=writer)
        return SessionSummaryMemoryBuilder(
            preprocess_index_dir=preprocess_index_dir,
            retrieve_top_k=retrieve_top_k,
        )
    if resolved == AgentFactMemoryBuilder.name:
        if writer is None:
            raise ValueError(
                "agent_codex_mem0_facts requires a writer "
                "(set writer.model / --writer-model)."
            )
        return AgentFactMemoryBuilder(writer=writer)
    return cls()


def is_removed_multi_model_method(memory_method: str) -> bool:
    """True for deleted pooled_* / fused_* ids so matrices fail closed."""
    key = str(memory_method or "").strip().lower()
    return key.startswith("pooled_teacher") or key.startswith("fused_teacher")


def is_question_independent(name: str, *, retrieve_top_k: int | None = None) -> bool:
    """True if memory text does not depend on the question (build once per conversation)."""
    if retrieve_top_k is not None:
        return False
    resolved = resolve_memory_name(name)
    return resolved in (
        RawConversationMemoryBuilder.name,
        SessionSummaryMemoryBuilder.name,
        AgentFactMemoryBuilder.name,
        GRAPH,
        "full_context",
        "openai_memory",
        "workspace_files",
    )
