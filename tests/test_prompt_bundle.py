"""Prompt bundle TRACE.md: config → prompt snapshot → jsonl map (no API)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import list_config_chain, load_config
from src.locomo_eval.experiments.prompt_bundle import write_prompt_bundle
from src.locomo_eval.prompts import QA_MEM0_V1, locate_prompt_file


class TestLocatePromptFile(unittest.TestCase):
    def test_repo_relative_reader_prompt_resolves(self):
        path = locate_prompt_file(QA_MEM0_V1)
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "qa_mem0_v1.txt")
        self.assertEqual(path.parent.name, "readers")


class TestListConfigChain(unittest.TestCase):
    def test_mem0_baseline_chain_ends_with_preset_and_includes_layout(self):
        entry = ROOT / "configs" / "presets" / "mem0_baseline.yaml"
        chain = [p.name for p in list_config_chain(entry)]
        self.assertEqual(chain[-1], "mem0_baseline.yaml")
        self.assertIn("qa_mem0_v1.yaml", chain)
        self.assertIn("session_summaries.yaml", chain)


class TestWritePromptBundle(unittest.TestCase):
    def test_session_summaries_run_dir_gets_trace_and_reader_snapshot(self):
        cfg = load_config(ROOT / "configs" / "writers" / "session_summaries.yaml")
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "smoke"
            write_prompt_bundle(
                run_dir,
                cfg,
                config_entry=ROOT / "configs" / "writers" / "session_summaries.yaml",
            )
            trace = (run_dir / "TRACE.md").read_text(encoding="utf-8")
            self.assertIn("pipeline.prompt_path", trace)
            self.assertIn(QA_MEM0_V1, trace)
            self.assertIn("reader/traces.jsonl", trace)
            snap = run_dir / QA_MEM0_V1
            self.assertTrue(snap.is_file())
            index = json.loads((run_dir / "prompts" / "index.json").read_text(encoding="utf-8"))
            roles = {row["role"] for row in index["prompts"]}
            self.assertEqual(roles, {"reader"})
            self.assertTrue((run_dir / "config.resolved.yaml").is_file())
            self.assertTrue((run_dir / "config.source.yaml").is_file())

    def test_autorater_merge_adds_judge_prompt_without_dropping_reader(self):
        cfg = load_config(ROOT / "configs" / "writers" / "session_summaries.yaml")
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "smoke"
            write_prompt_bundle(
                run_dir,
                cfg,
                config_entry=ROOT / "configs" / "writers" / "session_summaries.yaml",
            )
            write_prompt_bundle(
                run_dir,
                {"autorater": {"prompt_path": "prompts/autoraters/autorater_mem0_v1.txt"}},
                merge=True,
            )
            index = json.loads((run_dir / "prompts" / "index.json").read_text(encoding="utf-8"))
            keys = {row["config_key"] for row in index["prompts"]}
            self.assertIn("pipeline.prompt_path", keys)
            self.assertIn("autorater.prompt_path", keys)
            trace = (run_dir / "TRACE.md").read_text(encoding="utf-8")
            self.assertIn("autorater/traces.jsonl", trace)
            self.assertIn("session_summaries.yaml", trace)


if __name__ == "__main__":
    unittest.main()
