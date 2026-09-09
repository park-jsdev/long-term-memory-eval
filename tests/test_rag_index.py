"""RAG / full-context index + retrieve tests (mock embedder; no live API)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.dataset import parse_sample
from src.locomo_eval.memory import get_memory_builder, is_question_independent, resolve_memory_name
from src.locomo_eval.mem0.embeddings import MockEmbedder
from src.locomo_eval.rag.chunk import CHUNK_JOIN, chunk_transcript, format_conversation_transcript
from src.locomo_eval.rag.dump import write_sample_dump
from src.locomo_eval.rag.retrieve import retrieve_rag_text
from src.locomo_eval.rag.run_index import run_rag_index

GOLD_LOCK = "UNIQ_GOLD_REF_ZZZ"

MINI = {
    "sample_id": "conv-rag",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting in Boston."},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice!"},
        ],
        "session_2_date_time": "2 Jan 2023",
        "session_2": [
            {"dia_id": "D2:1", "speaker": "Alice", "text": "I got a nursing job in Seattle."},
        ],
    },
    "session_summary": {"session_1_summary": "Alice paints."},
    "qa": [
        {
            "question": "Where does Alice work?",
            "answer": GOLD_LOCK,
            "category": 4,
            "evidence": ["D2:1"],
        },
        {
            "question": "What hobby did Alice begin?",
            "answer": "painting",
            "category": 4,
            "evidence": ["D1:1"],
        },
    ],
}


class TestFormatAndChunk(unittest.TestCase):
    def test_format_conversation_transcript_uses_timestamp_pipe_speaker(self):
        conv = parse_sample(MINI)
        text = format_conversation_transcript(conv)
        self.assertIn("1 Jan 2023 | Alice: I started painting in Boston.", text)
        self.assertNotIn(GOLD_LOCK, text)

    def test_chunk_transcript_negative_size_returns_one_full_chunk(self):
        chunks = chunk_transcript("hello world", -1, sample_id="s")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "hello world")
        self.assertEqual(chunks[0].chunk_id, "s:full")

    def test_chunk_transcript_splits_into_nonempty_windows(self):
        chunks = chunk_transcript("alpha beta gamma delta", 2, sample_id="s")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(c.n_tokens > 0 for c in chunks))


class TestRagRetrieveFromDump(unittest.TestCase):
    def test_retrieve_joins_top_k_chunks_with_paper_delimiter(self):
        conv = parse_sample(MINI)
        transcript = format_conversation_transcript(conv)
        chunks = chunk_transcript(transcript, 8, sample_id=conv.sample_id)
        embedder = MockEmbedder()
        for chunk, emb in zip(chunks, embedder.embed([c.text for c in chunks])):
            chunk.embedding = emb
        with tempfile.TemporaryDirectory() as tmp:
            index_root = Path(tmp)
            write_sample_dump(
                index_root,
                sample_id=conv.sample_id,
                transcript=transcript,
                chunks=chunks,
                chunk_size=8,
                encoding="cl100k_base",
            )
            text, source_ids, search_s = retrieve_rag_text(
                index_root, conv.sample_id, "nursing job", embedder, k=2
            )
            self.assertGreater(len(source_ids), 0)
            self.assertGreaterEqual(search_s, 0.0)
            if len(source_ids) > 1:
                self.assertIn(CHUNK_JOIN, text)

    def test_rag_builder_is_question_dependent_and_omits_gold(self):
        self.assertFalse(is_question_independent("rag"))
        self.assertEqual(resolve_memory_name("rag"), "rag")
        conv = parse_sample(MINI)
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "locomo.json"
            data_path.write_text(json.dumps([MINI]), encoding="utf-8")
            cfg = {
                "data": {"raw_path": str(data_path)},
                "rag": {
                    "chunk_size": 16,
                    "k": 1,
                    "embed": {"provider": "mock", "model": "mock"},
                },
                "run": {"output_dir": str(tmp), "run_id": "idx"},
            }
            run_rag_index(
                cfg,
                Namespace(
                    data=str(data_path),
                    output_dir=tmp,
                    run_id="idx",
                    sample_id=None,
                    max_samples=None,
                    embedder="mock",
                    chunk_size=16,
                ),
            )
            builder = get_memory_builder(
                "rag",
                rag_index_dir=Path(tmp) / "idx" / "rag_index",
                rag_top_k=1,
                rag_embedder=MockEmbedder(),
            )
            mem = builder.build(conv, conv.questions[0])
            self.assertEqual(mem.memory_type, "rag")
            self.assertNotIn(GOLD_LOCK, mem.text)
            self.assertIsNotNone(mem.search_latency_s)


class TestFullContextBuilder(unittest.TestCase):
    def test_full_context_is_question_independent_and_contains_all_turns(self):
        self.assertTrue(is_question_independent("full_context"))
        conv = parse_sample(MINI)
        mem = get_memory_builder("full_context").build(conv, conv.questions[0])
        self.assertEqual(mem.memory_type, "full_context")
        self.assertIn("painting", mem.text)
        self.assertIn("nursing", mem.text)
        self.assertEqual(mem.search_latency_s, 0.0)
        self.assertNotIn(GOLD_LOCK, mem.text)


if __name__ == "__main__":
    unittest.main()
