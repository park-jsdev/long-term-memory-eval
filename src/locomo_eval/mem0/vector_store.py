"""In-memory fact list + cosine search (no Qdrant)."""

from __future__ import annotations

from .embeddings import cosine_similarity
from .schemas import Fact


class VectorMemoryStore:
    """One speaker's vector memories for one LoCoMo sample."""

    def __init__(self, *, sample_id: str, speaker_index: str, speaker_name: str):
        self.sample_id = sample_id
        self.speaker_index = speaker_index
        self.speaker_name = speaker_name
        self.facts: list[Fact] = []
        self._n = 0

    def next_fact_id(self) -> str:
        fid = f"{self.speaker_index}:{self._n:04d}"
        self._n += 1
        return fid

    def add(self, fact: Fact) -> Fact:
        if not fact.fact_id:
            fact.fact_id = self.next_fact_id()
        self.facts.append(fact)
        return fact

    def update(self, fact_id: str, text: str, embedding: list[float]) -> Fact | None:
        for fact in self.facts:
            if fact.fact_id == fact_id:
                fact.text = text
                fact.embedding = list(embedding)
                return fact
        return None

    def delete(self, fact_id: str) -> bool:
        before = len(self.facts)
        self.facts = [f for f in self.facts if f.fact_id != fact_id]
        return len(self.facts) < before

    def get(self, fact_id: str) -> Fact | None:
        for fact in self.facts:
            if fact.fact_id == fact_id:
                return fact
        return None

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Fact, float]]:
        scored = [
            (fact, cosine_similarity(query_embedding, fact.embedding))
            for fact in self.facts
            if fact.embedding
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        k = max(0, int(top_k))
        return scored[:k]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "speaker_index": self.speaker_index,
            "speaker_name": self.speaker_name,
            "n_facts": len(self.facts),
            "next_n": self._n,
            "facts": [f.to_dict() for f in self.facts],
        }

    @classmethod
    def from_dict(cls, row: dict) -> VectorMemoryStore:
        store = cls(
            sample_id=str(row["sample_id"]),
            speaker_index=str(row["speaker_index"]),
            speaker_name=str(row.get("speaker_name") or ""),
        )
        store._n = int(row.get("next_n") or 0)
        store.facts = [Fact.from_dict(f) for f in (row.get("facts") or [])]
        if store.facts and store._n == 0:
            store._n = len(store.facts)
        return store
