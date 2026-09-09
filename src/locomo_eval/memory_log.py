"""Describe and dump runtime Memory payloads for audit.

See docs/schemas/memory_runtime.md (memory_io.v1).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.locomo_eval.experiments.write import write_teacher_module
from .prompts import render_qa_prompt
from .report import write_json
from .schemas import Memory

SCHEMA_VERSION = "memory_io.v1"

# Fixed layout grammars for known conditions (human + machine readable).
CONDITION_LAYOUTS: dict[str, dict[str, Any]] = {
    "raw_chunks": {
        "builder": "RawConversationMemoryBuilder",
        "code": "src/locomo_eval/memory.py",
        "source_fields": [
            "conversation.speaker_a/b",
            "conversation.session_k",
            "conversation.session_k_date_time",
            "turn.speaker/text/dia_id/blip_caption",
        ],
        "text_layout": (
            "Conversation between {speaker_a} and {speaker_b}.\n\n"
            "DATE: {date}\nSESSION {k}:\n"
            "{speaker}: {text} [image: …]? ({dia_id})?\n"
            "…\n"
            "Optional prefix if truncated: "
            "'[... earlier turns truncated to max_chars ...]\\n' + tail"
        ),
        "notes": "Optional memory_max_chars keeps the end (recent) of the dialog.",
    },
    "session_summaries": {
        "builder": "SessionSummaryMemoryBuilder",
        "code": "src/locomo_eval/memory.py",
        "source_fields": ["session_summary.session_k_summary"],
        "text_layout": (
            "[Session {k}]\n{session_k_summary text}\n\n"
            "[Session {k+1}]\n…"
        ),
        "notes": "LoCoMo-provided summaries; chronological by session id.",
    },
    "teacher_session_summaries": {
        "builder": "TeacherSessionMemoryBuilder",
        "code": "src/locomo_eval/memory.py",
        "source_fields": [
            "conversation.session_k turns",
            "teacher.model (per-session summarize)",
        ],
        "text_layout": (
            "[Session {k}]\n{teacher summary of session k}\n\n"
            "[Session {k+1}]\n…"
        ),
        "notes": (
            "Live single teacher; swap teacher.model within a family. "
            "Not multi-teacher fusion."
        ),
    },
    "mem0": {
        "builder": "Mem0IndexMemoryBuilder",
        "code": "src/locomo_eval/mem0/builders.py",
        "source_fields": [
            "experiments/<index_run_id>/mem0_index/by_sample/<id>/speaker_*.json",
        ],
        "text_layout": (
            "Speaker {a} memories:\n{timestamp}: {fact}\n\n"
            "Speaker {b} memories:\n{timestamp}: {fact}"
        ),
        "notes": (
            "Loads a Mem0 write-index dump; cosine top_k=30 both speakers. "
            "Does not re-extract. Not a paper-J claim."
        ),
    },
    "mem0g": {
        "builder": "Mem0gIndexMemoryBuilder",
        "code": "src/locomo_eval/mem0/builders.py",
        "source_fields": [
            "experiments/<index_run_id>/mem0_index/by_sample/<id>/speaker_*.json",
            "experiments/<index_run_id>/mem0_index/by_sample/<id>/graph.json",
        ],
        "text_layout": (
            "Speaker {a} memories:\n{timestamp}: {fact}\n\n"
            "Speaker {b} memories:\n{timestamp}: {fact}\n\n"
            "Graph relations:\n{source} -- {relationship} -- {target}"
        ),
        "notes": (
            "Mem0 vector retrieve plus in-memory graph relations. "
            "Swap GraphMemory later; freeze extract for that claim."
        ),
    },
    "teacher_graph": {
        "builder": "OrchestratedGraphMemoryBuilder",
        "code": "src/locomo_eval/memory.py",
        "source_fields": [
            "conversation sessions via PreprocessingPipeline",
            "one Teacher.extract_session_graph",
            "Mem0GraphMemory.ingest_triples",
        ],
        "text_layout": (
            "Conversation between {speaker_a} and {speaker_b}.\n\n"
            "Graph relations:\n"
            "{source} -- {relationship} -- {target}"
        ),
        "notes": (
            "Single interchangeable teacher (openai / anthropic / deepseek) "
            "writes locked Mem0GraphMemory. Swap teacher.model."
        ),
    },
    "pooled_teacher_graph": {
        "builder": "OrchestratedGraphMemoryBuilder",
        "code": "src/locomo_eval/memory.py",
        "source_fields": [
            "conversation sessions via PreprocessingPipeline",
            "TeacherOrchestrator (equal_weight | random | round_robin)",
            "Mem0GraphMemory.ingest_triples",
        ],
        "text_layout": (
            "Conversation between {speaker_a} and {speaker_b}.\n\n"
            "Graph relations:\n"
            "{source} -- {relationship} -- {target}"
        ),
        "notes": (
            "Naive multi-teacher pool into locked Mem0g. equal_weight unions "
            "triples; random/round_robin pick one teacher per session."
        ),
    },
    "fused_teacher_graph": {
        "builder": "OrchestratedGraphMemoryBuilder",
        "code": "src/locomo_eval/memory.py",
        "source_fields": [
            "conversation sessions via PreprocessingPipeline",
            "TeacherOrchestrator (K teachers, majority_vote or resolve_*)",
            "Mem0GraphMemory.ingest_triples",
        ],
        "text_layout": (
            "Conversation between {speaker_a} and {speaker_b}.\n\n"
            "Graph relations:\n"
            "{source} -- {relationship} -- {target}"
        ),
        "notes": (
            "Fusion into locked Mem0g: majority_vote, or resolve_* baselines "
            "(top_voted / first / random / round_robin / confidence) for "
            "source+relationship disagreements. Not a paper-J claim."
        ),
    },
    "rag": {
        "builder": "RagMemoryBuilder",
        "code": "src/locomo_eval/rag/builders.py",
        "source_fields": [
            "experiments/<index_run_id>/rag_index/by_sample/<id>/chunks.json",
        ],
        "text_layout": "{chunk_i}\\n<->\\n{chunk_j}  (cosine top-k token windows)",
        "notes": (
            "Mem0 paper RAG clone: tiktoken cl100k_base chunks, "
            "text-embedding-3-small, k in {1,2}. Question-dependent. Not Table 2 J."
        ),
    },
    "full_context": {
        "builder": "FullContextMemoryBuilder",
        "code": "src/locomo_eval/rag/builders.py",
        "source_fields": [
            "conversation.session_k_date_time",
            "turn.speaker/text",
        ],
        "text_layout": "{timestamp} | {speaker}: {text}\\n…",
        "notes": (
            "Mem0 paper full-context: entire transcript, no retrieval. "
            "Not a number clone of J=72.90."
        ),
    },
    "openai_memory": {
        "builder": "OpenAIMemoryBuilder",
        "code": "src/locomo_eval/openai_memory/builders.py",
        "source_fields": [
            "experiments/<index_run_id>/openai_memory_index/by_sample/<id>/memories.json",
        ],
        "text_layout": "{timestamp} | {speaker}: {extracted fact}\\n…",
        "notes": (
            "Privileged retrieve-all of an extract dump. Architecture clone of "
            "the paper's ChatGPT Memory protocol, not the product and not J=52.90."
        ),
    },
}
# Older run packs may still log the numbered ids.
CONDITION_LAYOUTS["c0_raw"] = CONDITION_LAYOUTS["raw_chunks"]
CONDITION_LAYOUTS["c1_session_summary"] = CONDITION_LAYOUTS["session_summaries"]
CONDITION_LAYOUTS["c1_teacher"] = CONDITION_LAYOUTS["teacher_session_summaries"]


def memory_to_record(memory: Memory) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "memory_type": memory.memory_type,
        "text": memory.text,
        "source_ids": list(memory.source_ids),
    }
    if memory.teacher_model:
        rec["teacher_model"] = memory.teacher_model
    if memory.teacher_provider:
        rec["teacher_provider"] = memory.teacher_provider
    if memory.search_latency_s is not None:
        rec["search_latency_s"] = memory.search_latency_s
    return rec


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_memory_run_log(
    run_dir: Path,
    memories_by_sample: dict[str, Memory],
    *,
    max_chars_cfg: int | None,
    prompt_template: str,
    example_question: str | None,
    doc_path: str = "docs/schemas/memory_runtime.md",
    memories_by_question: dict[str, Memory] | None = None,
    question_sample_ids: dict[str, str] | None = None,
) -> Path:
    """Write experiments/<run_id>/memory/ audit package.

    Question-independent builders dump one text per sample. Question-dependent
    builders (rag, mem0, mem0g) also dump ``memory/by_question/<qid>.txt`` so
    retrieved chunks are auditable per experimental item.
    """
    mem_dir = Path(run_dir) / "memory"
    sample_dir = mem_dir / "by_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if not memories_by_sample:
        write_json(
            mem_dir / "schema.json",
            {
                "schema_version": SCHEMA_VERSION,
                "note": "No memories recorded (empty run).",
            },
        )
        return mem_dir

    first = next(iter(memories_by_sample.values()))
    memory_type = first.memory_type
    layout = CONDITION_LAYOUTS.get(
        memory_type,
        {
            "builder": "unknown",
            "code": "src/locomo_eval/memory.py",
            "source_fields": [],
            "text_layout": "see builder implementation",
            "notes": f"Unregistered layout for memory_type={memory_type}",
        },
    )

    schema_doc = {
        "schema_version": SCHEMA_VERSION,
        "memory_type": memory_type,
        "max_chars_config": max_chars_cfg,
        "layout": layout,
        "documentation": doc_path,
        "json_schema": "docs/schemas/memory_io.schema.json",
        "object_fields": {
            "memory_type": "condition id",
            "text": "full string injected as prompt {memory}",
            "source_ids": "provenance list",
            "schema_version": SCHEMA_VERSION,
            "teacher_model": "write-path model id(s); '+' joins multi-teacher runs",
            "teacher_provider": "openai | anthropic | deepseek | mock | '+' joined",
        },
        "teacher_model": first.teacher_model,
        "teacher_provider": first.teacher_provider,
        "injection": {
            "prompt_placeholders": ["{memory}", "{question}"],
            "gold_answer_in_memory": False,
        },
        "n_unique_samples": len(memories_by_sample),
        "n_unique_questions": len(memories_by_question or {}),
        "question_dependent_dump": bool(memories_by_question),
    }
    write_json(mem_dir / "schema.json", schema_doc)

    readme = "\n".join(
        [
            f"# Memory dump for this run (`{memory_type}`)",
            "",
            f"- Schema: **{SCHEMA_VERSION}**",
            f"- Human doc: `{doc_path}`",
            f"- Machine schema: `docs/schemas/memory_io.schema.json`",
            f"- This run’s layout: `memory/schema.json`",
            f"- Full texts: `memory/by_sample/<sample_id>.txt`",
            f"- Per-question texts (when retrieve is question-dependent): `memory/by_question/<question_id>.txt`",
            f"- Index: `memory/index.jsonl`",
            f"- Teacher LLM traces (when used): `memory/teachers/`",
            f"- Fused graph snapshot (when used): `memory/graph/`",
            "",
            "This folder is the audit trail of **exactly** what `{memory}` contained.",
            "",
        ]
    )
    (mem_dir / "README.md").write_text(readme, encoding="utf-8")

    index_path = mem_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as idx:
        for sample_id, memory in sorted(memories_by_sample.items()):
            text = memory.text or ""
            rel = f"memory/by_sample/{sample_id}.txt"
            out_txt = sample_dir / f"{sample_id}.txt"
            out_txt.write_text(text, encoding="utf-8")
            row = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "memory_type": memory.memory_type,
                "n_source_ids": len(memory.source_ids),
                "n_chars": len(text),
                "text_sha256": _sha256_text(text),
                "text_head": text[:400],
                "text_tail": text[-200:] if len(text) > 200 else text,
                "full_text_path": rel,
                "teacher_model": memory.teacher_model,
                "teacher_provider": memory.teacher_provider,
                "search_latency_s": memory.search_latency_s,
                "key_kind": "sample",
            }
            idx.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
        if memories_by_question:
            q_dir = mem_dir / "by_question"
            q_dir.mkdir(parents=True, exist_ok=True)
            q_sample = question_sample_ids or {}
            for question_id, memory in sorted(memories_by_question.items()):
                text = memory.text or ""
                rel = f"memory/by_question/{question_id}.txt"
                (q_dir / f"{question_id}.txt").write_text(text, encoding="utf-8")
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": q_sample.get(question_id),
                    "question_id": question_id,
                    "memory_type": memory.memory_type,
                    "n_source_ids": len(memory.source_ids),
                    "n_chars": len(text),
                    "text_sha256": _sha256_text(text),
                    "text_head": text[:400],
                    "text_tail": text[-200:] if len(text) > 200 else text,
                    "full_text_path": rel,
                    "search_latency_s": memory.search_latency_s,
                    "key_kind": "question",
                }
                idx.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")

    # One filled-prompt example (may be large; still useful for review).
    example_sid, example_mem = next(iter(sorted(memories_by_sample.items())))
    q = example_question or "(example question omitted)"
    filled = render_qa_prompt(prompt_template, example_mem.text, q)
    max_prompt_dump = 120_000
    if len(filled) > max_prompt_dump:
        filled = (
            filled[: max_prompt_dump // 2]
            + "\n\n[... prompt truncated for prompt_fill_example.txt ...]\n\n"
            + filled[-(max_prompt_dump // 2) :]
        )
    header = (
        f"# Example prompt fill\n"
        f"# sample_id={example_sid} memory_type={example_mem.memory_type}\n"
        f"# schema_version={SCHEMA_VERSION}\n\n"
    )
    (mem_dir / "prompt_fill_example.txt").write_text(header + filled, encoding="utf-8")

    return mem_dir


def collect_teacher_call_log(builder: Any) -> list[dict[str, Any]]:
    """Gather teacher call rows from an orchestrator and/or session builder."""
    rows: list[dict[str, Any]] = []
    orch = getattr(builder, "orchestrator", None)
    if orch is not None:
        rows.extend(list(getattr(orch, "call_log", []) or []))
    rows.extend(list(getattr(builder, "teacher_call_log", []) or []))
    return rows


def collect_fusion_log(builder: Any) -> list[dict[str, Any]]:
    orch = getattr(builder, "orchestrator", None)
    if orch is None:
        return []
    return list(getattr(orch, "fusion_log", []) or [])


def write_teacher_call_log(run_dir: Path, rows: list[dict[str, Any]]) -> Path | None:
    """Compat wrapper: write memory/teachers/ (and teacher_calls.jsonl)."""
    return write_teacher_module(run_dir, calls=rows)