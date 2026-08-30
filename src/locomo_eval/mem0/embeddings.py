"""Dense embeddings for Mem0 vector search (no Qdrant).

Paper/OSS default: text-embedding-3-small. MockEmbedder is bag-of-tokens so
offline tests can rank similar strings without an API.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod

import numpy as np

from ..env import load_env

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
MOCK_DIM = 32


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class Embedder(ABC):
    """text → dense vector. Used at write (facts) and read (query)."""

    model_name: str
    provider: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class MockEmbedder(Embedder):
    """Deterministic hashed bag-of-tokens. Similar wording ranks nearby."""

    provider = "mock"

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, dim: int = MOCK_DIM):
        self.model_name = model_name or DEFAULT_EMBED_MODEL
        self.dim = int(dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float64)
            for tok in (text or "").casefold().split():
                h = hashlib.sha256(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                vec[idx] += 1.0
            n = float(np.linalg.norm(vec))
            if n:
                vec = vec / n
            out.append(vec.tolist())
        return out


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings API. Same OPENAI_API_KEY as the reader."""

    provider = "openai"

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
    ):
        self.model_name = model or DEFAULT_EMBED_MODEL
        load_env()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set {api_key_env} in a repo-root .env file "
                f"(see .env.example) or export it before using OpenAI embeddings."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        self._client = OpenAI(api_key=api_key, timeout=timeout_s)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model_name, input=list(texts))
        by_idx = {item.index: item.embedding for item in resp.data}
        return [list(by_idx[i]) for i in range(len(texts))]


def get_embedder(
    name: str,
    model: str = DEFAULT_EMBED_MODEL,
) -> Embedder:
    key = (name or "mock").strip().lower()
    if key == "mock":
        return MockEmbedder(model_name=model)
    if key == "openai":
        return OpenAIEmbedder(model=model)
    raise ValueError(f"Unknown embedder '{name}'. Use openai or mock.")
