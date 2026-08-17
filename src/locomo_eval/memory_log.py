"""Describe and dump runtime Memory payloads for audit.

See docs/schemas/memory_runtime.md (memory_io.v1).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .prompts import render_qa_prompt
from .report import write_json
from .schemas import Memory

SCHEMA_VERSION = "memory_io.v1"

# Fixed layout grammars for known conditions (human + machine readable).
CONDITION_LAYOUTS: dict[str, dict[str, Any]] = {
    "c0_raw": {
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
    "c1_session_summary": {
        "builder": "SessionSummaryMemoryBuilder",
        "code": "src/locomo_eval/memory.py",
        "source_fields": ["session_summary.session_k_summary"],
        "text_layout": (
            "[Session {k}]\n{session_k_summary text}\n\n"
            "[Session {k+1}]\n…"
        ),
        "notes": "LoCoMo-provided summaries; chronological by session id.",
    },
}


def memory_to_record(memory: Memory) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "memory_type": memory.memory_type,
        "text": memory.text,
        "source_ids": list(memory.source_ids),
    }


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
) -> Path:
    """Write experiments/<run_id>/memory/ audit package."""
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
        },
        "injection": {
            "prompt_placeholders": ["{memory}", "{question}"],
            "gold_answer_in_memory": False,
        },
        "n_unique_samples": len(memories_by_sample),
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
            f"- Index: `memory/index.jsonl`",
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
