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
