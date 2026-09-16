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

from src.memorybench.completed_run_skip import (
    qa_success_path,
    should_skip_completed,
    write_success_marker,
)
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.gcs_object_store import gcs_run_prefix
from src.memorybench.hashed_run_id import hashed_run_id
from src.memorybench.load_experiment_yaml import load_experiment_yaml
from src.memorybench.resolve_task_index import resolve_task_index
from src.memorybench.write_run_manifest import write_run_manifest

POC = ROOT / "configs" / "experiments" / "poc.yaml"
LONGITUDINAL = ROOT / "configs" / "experiments" / "longitudinal_core.yaml"
SANDWICH = ROOT / "configs" / "experiments" / "sandwich_memory.yaml"
ABLATION = ROOT / "configs" / "experiments" / "teacher_ablation.yaml"
CALIBRATION = ROOT / "configs" / "experiments" / "calibration_c01_c03.yaml"
CEILING = ROOT / "configs" / "experiments" / "ceiling_frontier.yaml"


class TestExpandRunMatrixCounts(unittest.TestCase):
    def test_poc_expands_to_one_full_context_run(self):
        specs = expand_run_matrix(load_experiment_yaml(POC))
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].memory_method, "full_context")
        self.assertEqual(specs[0].run_index, 0)
        self.assertTrue(specs[0].run_id.startswith("locomo-poc-"))

    def test_calibration_expands_to_three_memory_methods(self):
        specs = expand_run_matrix(load_experiment_yaml(CALIBRATION))
        self.assertEqual(len(specs), 3)
        self.assertEqual(
            [s.memory_method for s in specs],
            ["full_context", "rag", "mem0"],
        )

    def test_sandwich_expands_to_three_frozen_reader_cells(self):
        specs = expand_run_matrix(load_experiment_yaml(SANDWICH))
        self.assertEqual(len(specs), 3)
        models = {s.reader.api_model_id for s in specs}
        self.assertEqual(models, {"gpt-4o-mini"})
        self.assertEqual(specs[0].experiment_type, "sandwich")

    def test_longitudinal_core_expands_to_twenty_seven_runs(self):
        specs = expand_run_matrix(load_experiment_yaml(LONGITUDINAL))
        self.assertEqual(len(specs), 27)

    def test_teacher_ablation_expands_to_fifteen_runs(self):
        specs = expand_run_matrix(load_experiment_yaml(ABLATION))
        self.assertEqual(len(specs), 15)

    def test_ceiling_expands_to_three_runs(self):
        specs = expand_run_matrix(load_experiment_yaml(CEILING))
        self.assertEqual(len(specs), 3)

    def test_mem0_reader_2024_writers_is_three_runnable_sandwich_cells(self):
        path = ROOT / "configs" / "experiments" / "mem0_reader_2024_writers.yaml"
        specs = expand_run_matrix(load_experiment_yaml(path))
        self.assertEqual(len(specs), 3)
        self.assertEqual(
            [(s.memory_method, s.writer.catalog_id if s.writer else None) for s in specs],
            [
                ("full_context", None),
                ("teacher_graph", "gpt-4o"),
                ("teacher_graph", "claude-3-5-sonnet"),
            ],
        )
        self.assertEqual({s.reader.catalog_id for s in specs}, {"gpt-4o"})
        self.assertEqual(specs[0].status, "runnable")
        self.assertEqual(specs[1].status, "runnable")
        self.assertEqual(specs[2].status, "to_confirm")
        self.assertEqual(len({s.run_id for s in specs}), 3)

    def test_2025_readers_full_context_is_six_reader_by_memory_cells(self):
        path = ROOT / "configs" / "experiments" / "2025_readers_full_context.yaml"
        specs = expand_run_matrix(load_experiment_yaml(path))
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            [s.memory_method for s in specs],
            ["full_context", "rag"] * 3,
        )
        self.assertEqual(
            [s.reader.catalog_id for s in specs],
            [
                "gpt-5",
                "gpt-5",
                "claude-sonnet-4-5",
                "claude-sonnet-4-5",
                "deepseek-v3",
                "deepseek-v3",
            ],
        )
        self.assertEqual({s.status for s in specs}, {"runnable"})
        self.assertEqual({s.max_questions for s in specs}, {None})
        self.assertEqual({s.max_samples for s in specs}, {None})
        self.assertTrue(
            all(
                s.rag_index_run_id == "rag_locomo10"
                for s in specs
                if s.memory_method == "rag"
            )
        )
        self.assertTrue(
            all(s.rag_index_run_id is None for s in specs if s.memory_method == "full_context")
        )

    def test_2025_readers_full_context_smoke_keeps_subset(self):
        cfg = load_experiment_yaml(
            ROOT / "configs" / "experiments" / "2025_readers_full_context_smoke_gcs.yaml"
        )
        self.assertEqual(
            cfg["experiment"]["name"], "locomo-2025-readers-full-context-smoke"
        )
        specs = expand_run_matrix(cfg)
        self.assertEqual(len(specs), 3)
        self.assertEqual([s.memory_method for s in specs], ["full_context"] * 3)
        self.assertEqual({s.max_questions for s in specs}, {5})
        self.assertEqual({s.max_samples for s in specs}, {1})
        self.assertTrue(
            all(s.run_id.startswith("locomo-2025-readers-full-context-smoke-") for s in specs)
        )

    def test_mem0_reader_2025_writers_is_six_sandwich_cells(self):
        path = ROOT / "configs" / "experiments" / "mem0_reader_2025_writers.yaml"
        specs = expand_run_matrix(load_experiment_yaml(path))
        self.assertEqual(len(specs), 6)
        self.assertEqual({s.reader.catalog_id for s in specs}, {"gpt-4o-mini"})
        pairs = [(s.memory_method, s.writer.catalog_id) for s in specs]
        self.assertEqual(
            pairs,
            [
                ("teacher_session_summaries", "gpt-5"),
                ("teacher_session_summaries", "claude-sonnet-4-5"),
                ("teacher_session_summaries", "deepseek-v3"),
                ("teacher_graph", "gpt-5"),
                ("teacher_graph", "claude-sonnet-4-5"),
                ("teacher_graph", "deepseek-v3"),
            ],
        )
        self.assertEqual({s.status for s in specs}, {"runnable"})

    def test_2025_readers_openai_deepseek_is_eight_thinking_cells(self):
        path = ROOT / "configs" / "experiments" / "2025_readers_openai_deepseek.yaml"
        specs = expand_run_matrix(load_experiment_yaml(path))
        self.assertEqual(len(specs), 8)
        self.assertEqual(
            [(s.reader.catalog_id, s.memory_method, s.thinking_label()) for s in specs],
            [
                ("gpt-5", "full_context", "off"),
                ("gpt-5", "full_context", "on"),
                ("gpt-5", "rag", "off"),
                ("gpt-5", "rag", "on"),
                ("deepseek-v3", "full_context", "off"),
                ("deepseek-v3", "full_context", "on"),
                ("deepseek-v3", "rag", "off"),
                ("deepseek-v3", "rag", "on"),
            ],
        )
        off = [s for s in specs if s.thinking_label() == "off"]
        on = [s for s in specs if s.thinking_label() == "on"]
        self.assertTrue(all(s.reader.max_tokens == 256 for s in off))
        self.assertTrue(all(s.reader.max_tokens == 8192 for s in on))
        self.assertTrue(all("reader_thinking" in s.qa_identity() for s in specs))
        self.assertNotEqual(off[0].run_id, on[0].run_id)
        self.assertTrue(
            all(s.run_id.startswith("locomo-2025-readers-openai-deepseek-") for s in specs)
        )
        self.assertFalse(
            any(s.run_id.startswith("locomo-2025-readers-full-context-") for s in specs)
        )
        self.assertEqual({s.status for s in specs}, {"runnable"})

    def test_2025_readers_openai_deepseek_smoke_is_four_full_context_cells(self):
        cfg = load_experiment_yaml(
            ROOT / "configs" / "experiments" / "2025_readers_openai_deepseek_smoke.yaml"
        )
        self.assertEqual(
            cfg["experiment"]["name"], "locomo-2025-readers-openai-deepseek-smoke"
        )
        self.assertEqual((cfg.get("storage") or {}).get("backend"), "local")
        specs = expand_run_matrix(cfg)
        self.assertEqual(len(specs), 4)
        self.assertEqual([s.memory_method for s in specs], ["full_context"] * 4)
        self.assertEqual(
            [s.reader.catalog_id for s in specs],
            ["gpt-5", "gpt-5", "deepseek-v3", "deepseek-v3"],
        )
        self.assertEqual(
            [s.thinking_label() for s in specs],
            ["off", "on", "off", "on"],
        )
        self.assertEqual({s.max_questions for s in specs}, {5})
        self.assertEqual({s.max_samples for s in specs}, {1})

    def test_mem0_reader_2025_writers_openai_deepseek_is_eight_sandwich_cells(self):
        path = (
            ROOT
            / "configs"
            / "experiments"
            / "mem0_reader_2025_writers_openai_deepseek.yaml"
        )
        specs = expand_run_matrix(load_experiment_yaml(path))
        self.assertEqual(len(specs), 8)
        self.assertEqual({s.reader.catalog_id for s in specs}, {"gpt-4o-mini"})
        self.assertTrue(all(s.reader.thinking is None for s in specs))
        self.assertEqual(
            [(s.memory_method, s.writer.catalog_id, s.thinking_label()) for s in specs],
            [
                ("teacher_session_summaries", "gpt-5", "off"),
                ("teacher_session_summaries", "gpt-5", "on"),
                ("teacher_session_summaries", "deepseek-v3", "off"),
                ("teacher_session_summaries", "deepseek-v3", "on"),
                ("teacher_graph", "gpt-5", "off"),
                ("teacher_graph", "gpt-5", "on"),
                ("teacher_graph", "deepseek-v3", "off"),
                ("teacher_graph", "deepseek-v3", "on"),
            ],
        )
        self.assertTrue(
            all(
                s.run_id.startswith("locomo-mem0-reader-2025-writers-openai-deepseek-")
                for s in specs
            )
        )
        self.assertEqual({s.status for s in specs}, {"runnable"})
        off = [s for s in specs if s.thinking_label() == "off"]
        on = [s for s in specs if s.thinking_label() == "on"]
        self.assertTrue(all(s.writer.thinking is False for s in off))
        self.assertTrue(all(s.writer.thinking is True for s in on))
        self.assertTrue(all(s.writer.max_tokens == 8192 for s in off))
        self.assertTrue(all(s.writer.max_tokens == 32768 for s in on))
        self.assertTrue(all("teacher_thinking" in s.qa_identity() for s in specs))
        gpt5 = [s for s in specs if s.writer.catalog_id == "gpt-5"]
        self.assertNotEqual(
            {s.run_id for s in gpt5 if s.thinking_label() == "off"},
            {s.run_id for s in gpt5 if s.thinking_label() == "on"},
        )

    def test_2025_openai_deepseek_gcs_overlays_keep_cell_counts(self):
        smoke = expand_run_matrix(
            load_experiment_yaml(
                ROOT
                / "configs"
                / "experiments"
                / "2025_readers_openai_deepseek_smoke_gcs.yaml"
            )
        )
        baseline = expand_run_matrix(
            load_experiment_yaml(
                ROOT / "configs" / "experiments" / "2025_readers_openai_deepseek_gcs.yaml"
            )
        )
        writers = expand_run_matrix(
            load_experiment_yaml(
                ROOT
                / "configs"
                / "experiments"
                / "mem0_reader_2025_writers_openai_deepseek_gcs.yaml"
            )
        )
        self.assertEqual(len(smoke), 4)
        self.assertEqual(len(baseline), 8)
        self.assertEqual(len(writers), 8)
        self.assertEqual(
            {s.reader.catalog_id for s in smoke},
            {"gpt-5", "deepseek-v3"},
        )
        smoke_cfg = load_experiment_yaml(
            ROOT
            / "configs"
            / "experiments"
            / "2025_readers_openai_deepseek_smoke_gcs.yaml"
        )
        self.assertEqual(smoke_cfg["storage"]["backend"], "gcs")
        self.assertEqual({s.reader.catalog_id for s in writers}, {"gpt-4o-mini"})
        self.assertEqual(
            {s.writer.catalog_id for s in writers},
            {"gpt-5", "deepseek-v3"},
        )

    def test_2026_readers_openai_deepseek_is_eight_thinking_cells(self):
        path = ROOT / "configs" / "experiments" / "2026_readers_openai_deepseek.yaml"
        specs = expand_run_matrix(load_experiment_yaml(path))
        self.assertEqual(len(specs), 8)
        self.assertEqual(
            [s.reader.catalog_id for s in specs],
            ["gpt-5.6-terra"] * 4 + ["deepseek-v4"] * 4,
        )
        self.assertEqual(
            [s.reader.api_model_id for s in specs],
            ["gpt-5.6-terra"] * 4 + ["deepseek-v4-flash"] * 4,
        )
        self.assertEqual(
            [s.thinking_label() for s in specs],
            ["off", "on"] * 4,
        )
        self.assertTrue(
            all(s.run_id.startswith("locomo-2026-readers-openai-deepseek-") for s in specs)
        )
        self.assertFalse(any("anthropic" in (s.reader.provider or "") for s in specs))
        self.assertEqual({s.status for s in specs}, {"runnable"})

    def test_2026_readers_openai_deepseek_smoke_is_four_full_context_cells(self):
        cfg = load_experiment_yaml(
            ROOT / "configs" / "experiments" / "2026_readers_openai_deepseek_smoke.yaml"
        )
        self.assertEqual(
            cfg["experiment"]["name"], "locomo-2026-readers-openai-deepseek-smoke"
        )
        self.assertEqual((cfg.get("storage") or {}).get("backend"), "local")
        specs = expand_run_matrix(cfg)
        self.assertEqual(len(specs), 4)
        self.assertEqual([s.memory_method for s in specs], ["full_context"] * 4)
        self.assertEqual(
            [s.reader.catalog_id for s in specs],
            ["gpt-5.6-terra", "gpt-5.6-terra", "deepseek-v4", "deepseek-v4"],
        )
        self.assertEqual({s.max_questions for s in specs}, {5})
        self.assertEqual({s.max_samples for s in specs}, {1})

    def test_mem0_reader_2026_writers_openai_deepseek_is_eight_sandwich_cells(self):
        path = (
            ROOT
            / "configs"
            / "experiments"
            / "mem0_reader_2026_writers_openai_deepseek.yaml"
        )
        specs = expand_run_matrix(load_experiment_yaml(path))
        self.assertEqual(len(specs), 8)
        self.assertEqual({s.reader.catalog_id for s in specs}, {"gpt-4o-mini"})
        self.assertEqual(
            [(s.memory_method, s.writer.catalog_id, s.thinking_label()) for s in specs],
            [
                ("teacher_session_summaries", "gpt-5.6-terra", "off"),
                ("teacher_session_summaries", "gpt-5.6-terra", "on"),
                ("teacher_session_summaries", "deepseek-v4", "off"),
                ("teacher_session_summaries", "deepseek-v4", "on"),
                ("teacher_graph", "gpt-5.6-terra", "off"),
                ("teacher_graph", "gpt-5.6-terra", "on"),
                ("teacher_graph", "deepseek-v4", "off"),
                ("teacher_graph", "deepseek-v4", "on"),
            ],
        )
        self.assertTrue(
            all(
                s.run_id.startswith("locomo-mem0-reader-2026-writers-openai-deepseek-")
                for s in specs
            )
        )
        self.assertEqual({s.status for s in specs}, {"runnable"})
        off = [s for s in specs if s.thinking_label() == "off"]
        on = [s for s in specs if s.thinking_label() == "on"]
        self.assertTrue(all(s.writer.max_tokens == 8192 for s in off))
        self.assertTrue(all(s.writer.max_tokens == 32768 for s in on))
        terra = [s for s in specs if s.writer.catalog_id == "gpt-5.6-terra"]
        self.assertTrue(terra)
        self.assertTrue(all(s.writer.thinking is not None for s in specs))

    def test_2026_openai_deepseek_gcs_overlays_keep_cell_counts(self):
        smoke = expand_run_matrix(
            load_experiment_yaml(
                ROOT
                / "configs"
                / "experiments"
                / "2026_readers_openai_deepseek_smoke_gcs.yaml"
            )
        )
        baseline = expand_run_matrix(
            load_experiment_yaml(
                ROOT / "configs" / "experiments" / "2026_readers_openai_deepseek_gcs.yaml"
            )
        )
        writers = expand_run_matrix(
            load_experiment_yaml(
                ROOT
                / "configs"
                / "experiments"
                / "mem0_reader_2026_writers_openai_deepseek_gcs.yaml"
            )
        )
        self.assertEqual(len(smoke), 4)
        self.assertEqual(len(baseline), 8)
        self.assertEqual(len(writers), 8)
        self.assertEqual(
            {s.reader.catalog_id for s in smoke},
            {"gpt-5.6-terra", "deepseek-v4"},
        )
        smoke_cfg = load_experiment_yaml(
            ROOT
            / "configs"
            / "experiments"
            / "2026_readers_openai_deepseek_smoke_gcs.yaml"
        )
        self.assertEqual(smoke_cfg["storage"]["backend"], "gcs")
        self.assertEqual({s.reader.catalog_id for s in writers}, {"gpt-4o-mini"})
        self.assertEqual(
            {s.writer.catalog_id for s in writers},
            {"gpt-5.6-terra", "deepseek-v4"},
        )
        self.assertEqual(
            {s.writer.api_model_id for s in writers},
            {"gpt-5.6-terra", "deepseek-v4-flash"},
        )

    def test_mem0_reader_2024_writers_gcs_keeps_live_execution(self):
        cfg = load_experiment_yaml(
            ROOT / "configs" / "experiments" / "mem0_reader_2024_writers_gcs.yaml"
        )
        self.assertEqual(cfg["storage"]["backend"], "gcs")
        self.assertNotEqual((cfg.get("execution") or {}).get("reader_provider"), "mock")
        specs = expand_run_matrix(cfg)
        self.assertEqual(len(specs), 3)

    def test_smoke_gcs_overlay_keeps_three_cells_and_subset(self):
        cfg = load_experiment_yaml(
            ROOT / "configs" / "experiments" / "mem0_reader_2024_writers_smoke_gcs.yaml"
        )
        self.assertEqual(
            cfg["experiment"]["name"], "locomo-mem0-reader-2024-writers-smoke"
        )
        specs = expand_run_matrix(cfg)
        self.assertEqual(len(specs), 3)
        self.assertEqual({s.max_samples for s in specs}, {1})
        self.assertEqual({s.max_questions for s in specs}, {5})
        self.assertTrue(
            all(
                s.run_id.startswith("locomo-mem0-reader-2024-writers-smoke-")
                for s in specs
            )
        )


class TestHashedRunIdsAreStable(unittest.TestCase):
    def test_expanding_the_same_yaml_twice_keeps_order_and_ids(self):
        a = expand_run_matrix(load_experiment_yaml(LONGITUDINAL))
        b = expand_run_matrix(load_experiment_yaml(LONGITUDINAL))
        self.assertEqual([s.run_id for s in a], [s.run_id for s in b])
        self.assertEqual([s.memory_method for s in a], [s.memory_method for s in b])
        self.assertEqual(len(set(s.run_id for s in a)), 27)

    def test_placeholder_api_ids_do_not_collide_across_catalog_rows(self):
        specs = expand_run_matrix(load_experiment_yaml(LONGITUDINAL))
        placeholders = [s for s in specs if s.reader.api_model_id == "TO_CONFIRM"]
        self.assertTrue(placeholders)
        ids = [s.run_id for s in placeholders]
        self.assertEqual(len(ids), len(set(ids)))


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


if __name__ == "__main__":
    unittest.main()
