"""Prompt templates for the fixed answer model."""

from __future__ import annotations

from pathlib import Path


def load_prompt_template(path: str | Path) -> tuple[str, str]:
    """Return (prompt_version, template_text).

    Version is derived from the filename stem (e.g. qa_v1.txt → qa_v1).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    return path.stem, text


def render_qa_prompt(template: str, memory: str, question: str) -> str:
    return template.format(memory=memory, question=question)


def render_teacher_session_prompt(
    template: str,
    *,
    date: str,
    session_text: str,
    speaker_a: str,
    speaker_b: str,
) -> str:
    """Fill prompts/teacher_session_v1.txt. Gold answers must not appear here."""
    return template.format(
        date=date,
        session_text=session_text,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )
