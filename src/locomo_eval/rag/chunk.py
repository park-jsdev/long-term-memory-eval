"""Token-chunk a LoCoMo conversation the way Mem0's RAG baseline does.

Transcript grammar matches Mem0 ``clean_chat_history``:
``{timestamp} | {speaker}: {text}``. Encoding is ``cl100k_base`` (paper:
tiktoken for ``text-embedding-3-small``). ``chunk_size < 0`` means
full-context (one chunk, no split).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import Conversation

DEFAULT_ENCODING = "cl100k_base"
CHUNK_JOIN = "\n<->\n"


@dataclass
class RagChunk:
    """One token window. Embeddings are filled at index time."""

    chunk_id: str
    text: str
    n_tokens: int
    embedding: list[float] | None = None


def format_conversation_transcript(conversation: Conversation) -> str:
    """Speaker lines with session timestamps. Gold never enters this string."""
    lines: list[str] = []
    for session in conversation.sessions:
        ts = session.date_time or "unknown"
        for turn in session.turns:
            line = f"{ts} | {turn.speaker}: {turn.text}"
            if turn.blip_caption:
                line += f" [image: {turn.blip_caption}]"
            lines.append(line)
    return "\n".join(lines)


def _encoding(name: str = DEFAULT_ENCODING):
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "RAG chunking needs tiktoken (pip install tiktoken). "
            "It is listed in requirements.txt."
        ) from exc
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    return len(_encoding(encoding_name).encode(text or ""))


def chunk_transcript(
    transcript: str,
    chunk_size: int,
    *,
    encoding_name: str = DEFAULT_ENCODING,
    sample_id: str = "",
) -> list[RagChunk]:
    """Split ``transcript`` into consecutive token windows.

    ``chunk_size < 0`` (Mem0 ``-1``) returns the whole transcript as one chunk.
    """
    text = transcript or ""
    if chunk_size < 0:
        n = count_tokens(text, encoding_name) if text else 0
        return [
            RagChunk(
                chunk_id=f"{sample_id}:full" if sample_id else "full",
                text=text,
                n_tokens=n,
            )
        ]
    if chunk_size == 0:
        raise ValueError("chunk_size must be > 0 (or < 0 for full-context).")
    enc = _encoding(encoding_name)
    tokens = enc.encode(text)
    chunks: list[RagChunk] = []
    if not tokens:
        return [
            RagChunk(
                chunk_id=f"{sample_id}:c0" if sample_id else "c0",
                text="",
                n_tokens=0,
            )
        ]
    for i, start in enumerate(range(0, len(tokens), int(chunk_size))):
        window = tokens[start : start + int(chunk_size)]
        chunk_text = enc.decode(window)
        cid = f"{sample_id}:c{i}" if sample_id else f"c{i}"
        chunks.append(RagChunk(chunk_id=cid, text=chunk_text, n_tokens=len(window)))
    return chunks
