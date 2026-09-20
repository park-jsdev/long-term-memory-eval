"""Prompt templates for the fixed answer model and write-path extractors.

Repo layout mirrors ``configs/`` roles so a human can walk YAML → txt → jsonl:

- ``prompts/readers/`` — answer LLM (``pipeline.prompt_path`` / layouts)
- ``prompts/agents/`` — coding-agent harness (workspace files, not stuffed memory)
- ``prompts/writers/`` — mem0 / mem0g / openai_memory extract
- ``prompts/teachers/`` — session summary + graph teachers
- ``prompts/autoraters/`` — LLM-as-a-Judge (separate job from QA)
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

QA_MEM0_V1 = "prompts/readers/qa_mem0_v1.txt"
QA_V1 = "prompts/readers/qa_v1.txt"
QA_WORKSPACE_V1 = "prompts/agents/qa_workspace_v1.txt"
TEACHER_SESSION_V1 = "prompts/teachers/teacher_session_v1.txt"
TEACHER_GRAPH_V1 = "prompts/teachers/teacher_graph_v1.txt"
AUTORATER_MEM0_V1 = "prompts/autoraters/autorater_mem0_v1.txt"
MEM0_EXTRACT_V1 = "prompts/writers/mem0_extract_v1.txt"
MEM0_UPDATE_V1 = "prompts/writers/mem0_update_v1.txt"
MEM0G_ENTITIES_V1 = "prompts/writers/mem0g_entities_v1.txt"
MEM0G_RELATIONS_V1 = "prompts/writers/mem0g_relations_v1.txt"
MEM0G_CONFLICT_V1 = "prompts/writers/mem0g_conflict_v1.txt"
OPENAI_MEMORY_EXTRACT_V1 = "prompts/writers/openai_memory_extract_v1.txt"


def locate_prompt_file(path: str | Path) -> Path:
    """Resolve a prompt path against cwd or the repo root."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    from_root = ROOT / candidate
    if from_root.is_file():
        return from_root.resolve()
    raise FileNotFoundError(f"Prompt file not found: {path}")


def load_prompt_template(path: str | Path) -> tuple[str, str]:
    """Return (prompt_version, template_text).

    Version is derived from the filename stem (e.g. qa_v1.txt → qa_v1).
    """
    resolved = locate_prompt_file(path)
    text = resolved.read_text(encoding="utf-8")
    return resolved.stem, text


def render_qa_prompt(template: str, memory: str, question: str) -> str:
    return template.format(memory=memory, question=question)


def render_autorater_prompt(
    template: str,
    *,
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> str:
    """Fill prompts/autoraters/autorater_mem0_v1.txt. Uses replace so gold/pred braces are safe."""
    return (
        template.replace("{question}", str(question))
        .replace("{gold_answer}", str(gold_answer))
        .replace("{generated_answer}", str(generated_answer))
    )


def render_teacher_session_prompt(
    template: str,
    *,
    date: str,
    session_text: str,
    speaker_a: str,
    speaker_b: str,
) -> str:
    """Fill prompts/teachers/teacher_session_v1.txt. Gold answers must not appear here."""
    return template.format(
        date=date,
        session_text=session_text,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )


def render_teacher_graph_prompt(template: str, *, user_id: str, text: str) -> str:
    """Fill prompts/teachers/teacher_graph_v1.txt. Gold answers must not appear here."""
    return template.format(user_id=user_id, text=text)
