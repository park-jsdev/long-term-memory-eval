"""Matrix expansion, hashed ids, skip markers, Cloud Run index (no API)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_runner.completed_run_skip import (
    qa_success_path,
    should_skip_completed,
    write_success_marker,
)
from src.experiment_runner.expand_run_matrix import expand_run_matrix
from src.experiment_runner.gcs_object_store import gcs_run_prefix
from src.experiment_runner.hashed_run_id import hashed_run_id
from src.experiment_runner.load_experiment_yaml import load_experiment_yaml
from src.experiment_runner.resolve_task_index import resolve_task_index
from src.experiment_runner.write_run_manifest import write_run_manifest

POC = ROOT / "configs" / "experiments" / "poc.yaml"
SANDWICH = ROOT / "configs" / "experiments" / "sandwich_memory.yaml"


class TestExpandRunMatrixCounts(unittest.TestCase):
    def test_poc_expands_to_one_full_context_run(self):
        specs = expand_run_matrix(load_experiment_yaml(POC))
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].memory_method, "full_context")
        self.assertEqual(specs[0].run_index, 0)
        self.assertTrue(specs[0].run_id.startswith("locomo-poc-"))

    def test_sandwich_expands_to_three_frozen_reader_cells(self):
        specs = expand_run_matrix(load_experiment_yaml(SANDWICH))
        self.assertEqual(len(specs), 3)
        models = {s.reader.api_model_id for s in specs}
        self.assertEqual(models, {"gpt-4o-mini"})
        self.assertEqual(specs[0].experiment_type, "sandwich")

class TestHashedRunIdsAreStable(unittest.TestCase):
    def test_expanding_the_same_yaml_twice_keeps_order_and_ids(self):
        a = expand_run_matrix(load_experiment_yaml(SANDWICH))
        b = expand_run_matrix(load_experiment_yaml(SANDWICH))
        self.assertEqual([s.run_id for s in a], [s.run_id for s in b])
        self.assertEqual([s.memory_method for s in a], [s.memory_method for s in b])
        self.assertEqual(len(set(s.run_id for s in a)), 3)


class TestSandwichRejectsReaderSweep(unittest.TestCase):
    def test_sandwich_with_two_readers_raises(self):
        cfg = load_experiment_yaml(SANDWICH)
        cfg["matrix"]["reader"] = [{"catalog": "gpt-4o"}, {"catalog": "gpt-5.6-sol"}]
        with self.assertRaises(ValueError):
            expand_run_matrix(cfg)


class TestResolveTaskIndex(unittest.TestCase):
    def test_cli_index_wins_over_cloud_env(self):
        with patch.dict(os.environ, {"CLOUD_RUN_TASK_INDEX": "9"}):
            self.assertEqual(resolve_task_index(3), 3)

    def test_cloud_env_used_when_cli_index_is_none(self):
        with patch.dict(os.environ, {"CLOUD_RUN_TASK_INDEX": "4"}):
            self.assertEqual(resolve_task_index(None), 4)

    def test_default_is_zero_when_neither_cli_nor_env(self):
        env = {k: v for k, v in os.environ.items() if k != "CLOUD_RUN_TASK_INDEX"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(resolve_task_index(None), 0)


class TestCompletedRunSkip(unittest.TestCase):
    def test_skip_when_success_marker_exists_and_force_is_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "_SUCCESS"
            write_success_marker(marker)
            self.assertTrue(should_skip_completed(marker, force=False))

    def test_force_does_not_skip_even_when_marker_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "_SUCCESS"
            write_success_marker(marker)
            self.assertFalse(should_skip_completed(marker, force=True))

    def test_missing_marker_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = qa_success_path(tmp)
            self.assertFalse(should_skip_completed(marker, force=False))


class TestManifestAndGcsPaths(unittest.TestCase):
    def test_write_run_manifest_has_one_json_object_per_line(self):
        specs = expand_run_matrix(load_experiment_yaml(POC))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "runs.jsonl"
            write_run_manifest(specs, dest)
            lines = dest.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["run_index"], 0)
            self.assertEqual(row["memory_method"], "full_context")

    def test_gcs_run_prefix_is_experiment_then_run_id(self):
        self.assertEqual(
            gcs_run_prefix("locomo-poc", "locomo-poc-abc"),
            "experiments/locomo-poc/runs/locomo-poc-abc",
        )

    def test_hashed_run_id_is_stable_for_the_same_identity(self):
        identity = {"memory_method": "full_context", "reader_model": "gpt-4o-mini"}
        self.assertEqual(hashed_run_id("Locomo POC", identity), hashed_run_id("Locomo POC", identity))


class TestWriterMatricesStayOneModel(unittest.TestCase):
    def test_openai_agent_writers_is_six_cells_with_a_writer(self):
        specs = expand_run_matrix(
            load_experiment_yaml(ROOT / "configs" / "experiments" / "openai_agent_writers.yaml")
        )
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            {spec.memory_method for spec in specs},
            {"session_summaries", "graph"},
        )
        self.assertTrue(all(spec.writer is not None for spec in specs))

    def test_openai_mini_writers_structured_is_two_cells(self):
        specs = expand_run_matrix(
            load_experiment_yaml(
                ROOT / "configs" / "experiments" / "openai_mini_writers_structured.yaml"
            )
        )
        self.assertEqual(len(specs), 2)
        self.assertEqual(
            [spec.memory_method for spec in specs],
            ["session_summaries", "graph"],
        )
        self.assertTrue(all(spec.writer is not None for spec in specs))
        self.assertEqual({spec.writer.api_model_id for spec in specs}, {"gpt-4o-mini"})


if __name__ == "__main__":
    unittest.main()
