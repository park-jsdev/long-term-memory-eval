"""Load an OpenAI-memory dump and inject every extracted entry (no top-k)."""

from __future__ import annotations

from pathlib import Path

from ..memory import MemoryBuilder
from ..schemas import Conversation, Memory, Question
from .dump import load_sample_memories
from .extract import ExtractedMemory

RUN_INDEX_HINT = (
    "Run: python -m src.locomo_eval.openai_memory.run_index "
    "--config configs/openai_memory.yaml --run-id <index_run_id>"
)


class MissingOpenAIMemoryIndexError(FileNotFoundError):
    """Raised when the builder needs a dump that run_index has not written."""


def require_sample_dump(index_root: Path, sample_id: str) -> Path:
    sample_dir = Path(index_root) / "by_sample" / sample_id
    if not (sample_dir / "memories.json").is_file():
        raise MissingOpenAIMemoryIndexError(
            f"No OpenAI-memory dump for sample {sample_id} under {sample_dir}. "
            f"{RUN_INDEX_HINT}"
        )
    return sample_dir


def format_openai_memory_text(memories: list[ExtractedMemory]) -> str:
    if not memories:
        return "(No OpenAI-memory entries extracted for this conversation.)"
    lines = []
    for mem in memories:
        ts = mem.timestamp or "unknown"
        speaker = mem.speaker or "unknown"
        lines.append(f"{ts} | {speaker}: {mem.text}")
    return "\n".join(lines)


class OpenAIMemoryBuilder(MemoryBuilder):
    """Privileged retrieve-all of a write-index dump. Question-independent."""

    name = "openai_memory"

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)

    def build(self, conversation: Conversation, question: Question) -> Memory:
        sample_dir = require_sample_dump(self.index_dir, conversation.sample_id)
        memories = load_sample_memories(sample_dir)
        text = format_openai_memory_text(memories)
        source_ids = [
            f"{m.speaker}:{i}" for i, m in enumerate(memories)
        ]
        return Memory(
            memory_type=self.name,
            text=text,
            source_ids=source_ids,
            search_latency_s=0.0,
        )
