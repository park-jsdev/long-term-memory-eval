"""Tests for dataset parsing, memory build, and metrics."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.dataset import parse_sample
from src.locomo_eval.memory import (
    RawConversationMemoryBuilder,
    SessionSummaryMemoryBuilder,
    get_memory_builder,
    resolve_memory_name,
)
from src.locomo_eval.metrics import exact_match, normalize_answer, token_f1
from src.metrics.locomo_qa import score_prediction


MINI = {
    "sample_id": "conv-test",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice!"},
        ],
        "session_2_date_time": "2 Jan 2023",
        "session_2": [
            {"dia_id": "D2:1", "speaker": "Alice", "text": "I got a nursing job."},
        ],
    },
    "session_summary": {
        "session_1_summary": "Alice said she started painting.",
        "session_2_summary": "Alice got a nursing job.",
    },
    "observation": {},
    "qa": [
        {
            "question": "What hobby did Alice begin?",
            "answer": "painting",
            "category": 4,
            "evidence": ["D1:1"],
        },
        {
            "question": "When did this happen?",
            "answer": "last June",
            "category": 2,
            "evidence": [],
        },
    ],
}


class TestBaseline(unittest.TestCase):
    def test_parse_sample(self):
        conv = parse_sample(MINI)
        self.assertEqual(conv.sample_id, "conv-test")
        self.assertEqual(conv.speaker_a, "Alice")
        self.assertEqual(len(conv.sessions), 2)
        self.assertEqual(conv.sessions[0].session_id, 1)
        self.assertEqual(len(conv.sessions[0].turns), 2)
        self.assertEqual(len(conv.questions), 2)
        self.assertEqual(conv.questions[0].question_id, "conv-test-q-0")
        self.assertIn(1, conv.session_summaries)

    def test_c1_session_summary_memory(self):
        conv = parse_sample(MINI)
        mem = SessionSummaryMemoryBuilder().build(conv, conv.questions[0])
        self.assertEqual(mem.memory_type, "c1_session_summary")
        self.assertIn("Session 1", mem.text)
        self.assertIn("painting", mem.text)
        self.assertIn("nursing", mem.text)
        self.assertEqual(mem.source_ids, ["session_1_summary", "session_2_summary"])
        # Alias still resolves
        alias = get_memory_builder("session_summary")
        self.assertEqual(alias.name, "c1_session_summary")

    def test_c0_raw_memory(self):
        conv = parse_sample(MINI)
        mem = RawConversationMemoryBuilder().build(conv, conv.questions[0])
        self.assertEqual(mem.memory_type, "c0_raw")
        self.assertIn("I started painting", mem.text)
        self.assertIn("SESSION 1", mem.text)
        self.assertIn("D1:1", mem.text)
        # Truncation keeps recent tail
        short = RawConversationMemoryBuilder(max_chars=40).build(conv, conv.questions[0])
        self.assertIn("truncated", short.text.lower())
        self.assertEqual(resolve_memory_name("c0"), "c0_raw")

    def test_normalize_and_scores(self):
        self.assertEqual(normalize_answer("The Cat!"), "cat")
        self.assertEqual(exact_match("The cat", "cat"), 1.0)
        self.assertGreater(token_f1("red blue cat", "blue cat"), 0.5)
        self.assertEqual(score_prediction("May 7 2023", "7 May 2023", 2), 1.0)
        self.assertEqual(score_prediction("not mentioned", "anything", 5), 1.0)

    def test_real_locomo_if_present(self):
        path = ROOT / "data" / "raw" / "locomo10.json"
        if not path.exists():
            self.skipTest("locomo10.json not fetched")
        data = json.loads(path.read_text(encoding="utf-8"))
        conv = parse_sample(data[0])
        self.assertTrue(conv.questions)
        mem = SessionSummaryMemoryBuilder().build(conv, conv.questions[0])
        self.assertGreater(len(mem.text), 100)
        raw = RawConversationMemoryBuilder().build(conv, conv.questions[0])
        self.assertGreater(len(raw.text), 100)


if __name__ == "__main__":
    unittest.main()
