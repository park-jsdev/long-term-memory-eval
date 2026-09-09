"""OpenAI-memory extract dump + privileged retrieve-all (mock, no API)."""

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
from src.locomo_eval.memory import get_memory_builder, is_question_independent
from src.locomo_eval.openai_memory.extract import MockOpenAIMemoryExtractor
from src.locomo_eval.openai_memory.run_index import run_openai_memory_index
from src.locomo_eval.rag.chunk import format_conversation_transcript

GOLD_LOCK = "UNIQ_GOLD_REF_ZZZ"

MINI = {
    "sample_id": "conv-oai-mem",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice!"},
        ],
    },
    "session_summary": {},
    "qa": [
        {
            "question": "What hobby?",
            "answer": GOLD_LOCK,
            "category": 4,
            "evidence": ["D1:1"],
        }
    ],
}


class TestMockExtractorAndBuilder(unittest.TestCase):
    def test_mock_extractor_writes_one_memory_per_turn_without_gold(self):
        conv = parse_sample(MINI)
        transcript = format_conversation_transcript(conv)
        memories, meta = MockOpenAIMemoryExtractor().extract(conv, transcript)
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0].speaker, "Alice")
        blob = " ".join(m.text for m in memories)
        self.assertNotIn(GOLD_LOCK, blob)
        self.assertEqual(meta["role"], "openai_memory_extract")

    def test_openai_memory_builder_concatenates_all_entries_and_is_question_independent(self):
        self.assertTrue(is_question_independent("openai_memory"))
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "locomo.json"
            data_path.write_text(json.dumps([MINI]), encoding="utf-8")
            cfg = {
                "data": {"raw_path": str(data_path)},
                "openai_memory": {
                    "extract": {"provider": "mock", "model": "mock"},
                },
                "run": {"output_dir": str(tmp), "run_id": "idx"},
            }
            run_openai_memory_index(
                cfg,
                Namespace(
                    data=str(data_path),
                    output_dir=tmp,
                    run_id="idx",
                    sample_id=None,
                    max_samples=None,
                    extractor="mock",
                ),
            )
            index_dir = Path(tmp) / "idx" / "openai_memory_index"
            builder = get_memory_builder(
                "openai_memory", openai_memory_index_dir=index_dir
            )
            conv = parse_sample(MINI)
            mem = builder.build(conv, conv.questions[0])
            self.assertEqual(mem.memory_type, "openai_memory")
            self.assertIn("painting", mem.text)
            self.assertNotIn(GOLD_LOCK, mem.text)
            self.assertTrue((index_dir / "by_sample" / "conv-oai-mem" / "memories.json").is_file())


if __name__ == "__main__":
    unittest.main()
