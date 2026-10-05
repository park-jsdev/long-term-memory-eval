"""Path contract for one experiment pack: ``experiments/<run_id>/``.

This module names folders and files. It does not read or write them.

A frozen-reader run holds the dataset and the evaluation fixed and varies the memory system. The on-disk
tree is the **audit** of those layers (reader, writer, memory graph, judge)
so later analysis can map a condition to its results without re-running LLMs.

``AuditPaths`` is the shared map so the dump side (``audit_writer``) and the
analysis side (``audit_loader``) never invent different filenames.

Do not import the writer model, readers, or ``run.py`` from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

AUDIT_LAYOUT_VERSION = "audit_pack.v3"

READER = "reader"
AGENT = "agent"
MEMORY = "memory"
WRITER_ROOT = "memory/writer"
GRAPH = "memory/graph"
AUTORATER = "autorater"
PROMPTS = "prompts"
TRACE = "TRACE.md"
# Run-root copies so older compare scripts need not know about reader/ / writer/.
COMPAT_PREDICTIONS = "predictions.jsonl"
COMPAT_WRITER_CALLS = "memory/writer_calls.jsonl"
LINEAGE = "memory/lineage.jsonl"
RETRIEVE_RANKS = "memory/retrieve_ranks.jsonl"
GRAPH_INGEST = "memory/graph/ingest.jsonl"
WRITER_QUALITY = "memory/writer/quality.json"
WRITER_SESSIONS = "memory/writer/sessions.jsonl"
COST = "cost.json"
SUMMARY = "SUMMARY.md"
ATTRIBUTION = "attribution.jsonl"
ATTRIBUTION_MD = "ATTRIBUTION.md"
CONFIG_SOURCE = "config.source.yaml"
CONFIG_RESOLVED = "config.resolved.yaml"


def audit_layout_meta() -> dict[str, str]:
    """Relative paths pinned on ``run_meta.json['audit_layout']``.

    Analysis should read this dict (or ``AuditPaths``) instead of hardcoding
    filenames, so a layout bump is visible in the run pack.
    """
    return {
        "version": AUDIT_LAYOUT_VERSION,
        "reader": f"{READER}/",
        "agent": f"{AGENT}/",
        "memory": f"{MEMORY}/",
        "writer": f"{WRITER_ROOT}/",
        "graph": f"{GRAPH}/",
        "autorater": f"{AUTORATER}/",
        "prompts": f"{PROMPTS}/",
        "trace": TRACE,
        "compat_predictions": COMPAT_PREDICTIONS,
        "lineage": LINEAGE,
        "retrieve_ranks": RETRIEVE_RANKS,
        "graph_ingest": GRAPH_INGEST,
        "writer_quality": WRITER_QUALITY,
        "writer_sessions": WRITER_SESSIONS,
        "cost": COST,
        "summary": SUMMARY,
        "attribution": ATTRIBUTION,
        "attribution_md": ATTRIBUTION_MD,
        "config_source": CONFIG_SOURCE,
        "config_resolved": CONFIG_RESOLVED,
    }


def writer_dir_name(writer_id: str) -> str:
    """Filesystem-safe folder for ``memory/writer/by_writer/<id>/``.

    Writer ids can contain ``+`` or provider punctuation; those are not
    legal directory names on every OS.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(writer_id))
    return safe or "writer"


@dataclass(frozen=True)
class AuditPaths:
    """Absolute paths for one experiment-pack directory. Missing files are still listed."""

    run_dir: Path
    run_meta: Path
    metrics: Path
    predictions_root: Path  # run-root copy; reader/predictions.jsonl is the same rows
    reader_dir: Path
    reader_predictions: Path
    reader_traces: Path
    reader_metrics: Path
    agent_dir: Path
    agent_traces: Path
    agent_trajectory: Path
    agent_events: Path
    agent_metrics: Path
    memory_dir: Path
    writer_dir: Path
    writer_index: Path
    writer_calls: Path
    writer_calls_compat: Path  # memory/writer_calls.jsonl; same as writer/calls.jsonl
    writer_quality: Path
    writer_sessions_index: Path
    graph_dir: Path
    graph_index: Path
    graph_ingest: Path
    lineage: Path
    retrieve_ranks: Path
    autorater_dir: Path
    autorater_verdicts: Path
    autorater_traces: Path
    prompts_dir: Path
    prompt_index: Path
    trace: Path
    cost: Path
    summary: Path
    attribution: Path  # machine call → role → claims
    attribution_md: Path  # human ATTRIBUTION.md
    config_source: Path
    config_resolved: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> AuditPaths:
        """Bind the relative contract names to one ``experiments/<run_id>/``.

        Missing files are still listed so dump and load share one map.
        """
        root = Path(run_dir)
        reader = root / READER
        agent = root / AGENT
        writer = root / WRITER_ROOT
        graph = root / GRAPH
        autorater = root / AUTORATER
        prompts = root / PROMPTS
        return cls(
            run_dir=root,
            run_meta=root / "run_meta.json",
            metrics=root / "metrics.json",
            predictions_root=root / COMPAT_PREDICTIONS,
            reader_dir=reader,
            reader_predictions=reader / "predictions.jsonl",
            reader_traces=reader / "traces.jsonl",
            reader_metrics=reader / "metrics.json",
            agent_dir=agent,
            agent_traces=agent / "traces.jsonl",
            agent_trajectory=agent / "trajectory.jsonl",
            agent_events=agent / "events.jsonl",
            agent_metrics=agent / "metrics.json",
            memory_dir=root / MEMORY,
            writer_dir=writer,
            writer_index=writer / "index.jsonl",
            writer_calls=writer / "calls.jsonl",
            writer_calls_compat=root / COMPAT_WRITER_CALLS,
            writer_quality=root / WRITER_QUALITY,
            writer_sessions_index=root / WRITER_SESSIONS,
            graph_dir=graph,
            graph_index=graph / "index.jsonl",
            graph_ingest=root / GRAPH_INGEST,
            lineage=root / LINEAGE,
            retrieve_ranks=root / RETRIEVE_RANKS,
            autorater_dir=autorater,
            autorater_verdicts=autorater / "autorater_verdicts.jsonl",
            autorater_traces=autorater / "traces.jsonl",
            prompts_dir=prompts,
            prompt_index=prompts / "index.json",
            trace=root / TRACE,
            cost=root / COST,
            summary=root / SUMMARY,
            attribution=root / ATTRIBUTION,
            attribution_md=root / ATTRIBUTION_MD,
            config_source=root / CONFIG_SOURCE,
            config_resolved=root / CONFIG_RESOLVED,
        )

    def writer_calls_path(self, writer_id: str) -> Path:
        """Per-writer partition of ``calls.jsonl`` (same rows, one model)."""
        return self.writer_dir / "by_writer" / writer_dir_name(writer_id) / "calls.jsonl"

    def graph_sample_path(self, sample_id: str) -> Path:
        """Mem0g snapshot for one conversation (embeddings omitted)."""
        return self.graph_dir / "by_sample" / f"{sample_id}.json"

    def writer_session_text_path(self, sample_id: str, session_id: str | int) -> Path:
        """Exact session text the writer saw for this sample/session."""
        return (
            self.writer_dir
            / "sessions"
            / "by_sample"
            / str(sample_id)
            / f"session_{session_id}.txt"
        )
