"""Unit tests for YAML includes / deep-merge (no API)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import deep_merge, load_config
from src.locomo_eval.readers import READER_MESSAGE_LAYOUT_MEM0


class TestDeepMerge(unittest.TestCase):
    def test_nested_mappings_merge_and_later_scalar_wins(self):
        merged = deep_merge(
            {"reader": {"model": "a", "temperature": 0.0}, "keep": 1},
            {"reader": {"model": "b"}},
        )
        self.assertEqual(merged["reader"]["model"], "b")
        self.assertEqual(merged["reader"]["temperature"], 0.0)
        self.assertEqual(merged["keep"], 1)

    def test_lists_replace_rather_than_concatenate(self):
        merged = deep_merge({"items": [1]}, {"items": [2, 3]})
        self.assertEqual(merged["items"], [2, 3])


class TestIncludesComposeRunnableWriters(unittest.TestCase):
    def test_mem0_baseline_preset_keeps_parity_reader_and_session_memory(self):
        cfg = load_config(ROOT / "configs" / "presets" / "mem0_baseline.yaml")
        self.assertNotIn("includes", cfg)
        self.assertEqual(cfg["pipeline"]["memory"], "session_summaries")
        self.assertEqual(cfg["pipeline"]["prompt_path"], "prompts/readers/qa_mem0_v1.txt")
        self.assertEqual(cfg["reader"]["model"], "gpt-4o-mini")
        self.assertIsNone(cfg["reader"]["max_tokens"])
        self.assertEqual(cfg["reader"]["message_layout"], READER_MESSAGE_LAYOUT_MEM0)

    def test_session_summaries_writer_uses_default_reader_token_cap(self):
        cfg = load_config(ROOT / "configs" / "writers" / "session_summaries.yaml")
        self.assertEqual(cfg["pipeline"]["memory"], "session_summaries")
        self.assertEqual(cfg["reader"]["max_tokens"], 64)
        self.assertNotEqual(cfg["reader"].get("message_layout"), READER_MESSAGE_LAYOUT_MEM0)

    def test_luna_preset_overlays_reader_without_changing_memory(self):
        cfg = load_config(
            ROOT / "configs" / "presets" / "session_summaries_gpt-5.6-luna.yaml"
        )
        self.assertEqual(cfg["pipeline"]["memory"], "session_summaries")
        self.assertEqual(cfg["reader"]["model"], "gpt-5.6-luna")

    def test_mem0g_writer_enables_graph_on_shared_mem0_block(self):
        cfg = load_config(ROOT / "configs" / "writers" / "mem0g.yaml")
        self.assertEqual(cfg["pipeline"]["memory"], "mem0g")
        self.assertTrue(cfg["mem0"]["enable_graph"])
        self.assertEqual(cfg["mem0"]["index_run_id"], "mem0g_locomo10")
        self.assertEqual(cfg["mem0"]["extract"]["model"], "gpt-4o-mini")

    def test_graph_includes_one_writer_model(self):
        cfg = load_config(ROOT / "configs" / "writers" / "graph.yaml")
        self.assertEqual(cfg["writer"]["model"], "gpt-4o-mini")
        self.assertNotIn("teachers", cfg)
        self.assertNotIn("pool", cfg.get("orchestrator") or {})
        self.assertNotIn("fusion", cfg.get("orchestrator") or {})

    def test_include_cycle_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.yaml"
            b = Path(tmp) / "b.yaml"
            a.write_text("includes:\n  - b.yaml\n", encoding="utf-8")
            b.write_text("includes:\n  - a.yaml\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(a)


if __name__ == "__main__":
    unittest.main()
