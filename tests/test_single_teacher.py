"""Single-teacher lock for current LoCoMo / Mem0 / agent experiments.

The pooled/fused cheap_k3 plumbing exists, but memorybench matrices must stay
K=1 until a dedicated multi-teacher experiment is added. Offline / mock only.
"""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.locomo_eval.fusion import (
    POOL_EQUAL_WEIGHT,
    POOL_SINGLE,
    is_multi_teacher_memory_method,
    select_teacher_ids,
)
from src.locomo_eval.run import (
    _build_orchestrator,
    _teacher_cardinality,
    _teacher_roster,
    build_parser,
    run_locomo_pipeline_with_memory_config,
)
from src.locomo_eval.teachers import MockTeacher
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.load_experiment_yaml import load_experiment_yaml

WRITERS = ROOT / "configs" / "writers"
EXPERIMENTS = ROOT / "configs" / "experiments"
MINI = [
    {
        "sample_id": "conv-eval",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1 Jan 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
                {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice!"},
            ],
        },
        "session_summary": {"session_1_summary": "Alice paints."},
        "qa": [
            {
                "question": "What hobby did Alice begin?",
                "answer": "painting",
                "category": 4,
                "evidence": ["D1:1"],
            }
        ],
    }
]

SINGLE_TEACHER_WRITERS = (
    "teacher_graph.yaml",
    "teacher_session_summaries.yaml",
    "agent_codex_mem0_facts.yaml",
)
NO_TEACHER_WRITERS = (
    "raw_chunks.yaml",
    "session_summaries.yaml",
    "full_context.yaml",
    "rag.yaml",
    "openai_memory.yaml",
    "mem0.yaml",
    "mem0g.yaml",
    "workspace_files.yaml",
)
MULTI_TEACHER_WRITERS = (
    "pooled_teacher_graph.yaml",
    "fused_teacher_graph.yaml",
    "fused_teacher_graph_resolve_top_voted.yaml",
    "fused_teacher_graph_resolve_first.yaml",
    "fused_teacher_graph_resolve_random.yaml",
    "fused_teacher_graph_resolve_round_robin.yaml",
    "fused_teacher_graph_resolve_confidence.yaml",
)


def _yaml_slots(cfg: dict) -> list:
    roster = list(cfg.get("teachers") or [])
    if roster:
        return roster
    teacher = cfg.get("teacher") or {}
    if teacher.get("model"):
        return [teacher]
    return []


class TestIsMultiTeacherMemoryMethod(unittest.TestCase):
    def test_teacher_graph_and_session_summaries_are_single_teacher(self):
        self.assertFalse(is_multi_teacher_memory_method("teacher_graph"))
        self.assertFalse(is_multi_teacher_memory_method("teacher_session_summaries"))
        self.assertFalse(is_multi_teacher_memory_method("mem0"))
        self.assertFalse(is_multi_teacher_memory_method("workspace_files"))

    def test_pooled_and_fused_ids_are_multi_teacher(self):
        self.assertTrue(is_multi_teacher_memory_method("pooled_teacher_graph"))
        self.assertTrue(is_multi_teacher_memory_method("fused_teacher_graph"))
        self.assertTrue(
            is_multi_teacher_memory_method("fused_teacher_graph_resolve_first")
        )


class TestSelectTeacherIdsSinglePool(unittest.TestCase):
    def test_pool_single_returns_only_the_first_id_from_a_three_teacher_roster(self):
        ids = select_teacher_ids(
            ["openai", "anthropic", "deepseek"],
            pool=POOL_SINGLE,
            session_index=4,
            rng=__import__("random").Random(0),
        )
        self.assertEqual(ids, ["openai"])


class TestWriterYamlTeacherCardinality(unittest.TestCase):
    def test_single_teacher_writer_yamls_have_one_slot_and_pool_single(self):
        for name in SINGLE_TEACHER_WRITERS:
            with self.subTest(name=name):
                cfg = load_config(WRITERS / name)
                slots = _yaml_slots(cfg)
                self.assertEqual(len(slots), 1, name)
                self.assertFalse(is_multi_teacher_memory_method(cfg["pipeline"]["memory"]))
                if cfg["pipeline"]["memory"] == "teacher_graph":
                    self.assertEqual(cfg["orchestrator"]["pool"], POOL_SINGLE)
                    self.assertNotIn("teachers", cfg)

    def test_mem0_extract_is_one_model_not_a_teacher_roster(self):
        for name in ("mem0.yaml", "mem0g.yaml"):
            with self.subTest(name=name):
                cfg = load_config(WRITERS / name)
                self.assertNotIn("teachers", cfg)
                extract = (cfg.get("mem0") or {}).get("extract") or {}
                self.assertIsInstance(extract.get("model"), str)
                self.assertTrue(extract["model"])

    def test_non_teacher_writers_have_no_roster(self):
        for name in NO_TEACHER_WRITERS:
            with self.subTest(name=name):
                cfg = load_config(WRITERS / name)
                self.assertEqual(_yaml_slots(cfg), [], name)
                self.assertFalse(is_multi_teacher_memory_method(cfg["pipeline"]["memory"]))

    def test_pooled_and_fused_writer_yamls_keep_cheap_k3_roster(self):
        cheap = load_config(ROOT / "configs" / "teachers" / "cheap_k3.yaml")
        self.assertEqual(len(cheap["teachers"]), 3)
        for name in MULTI_TEACHER_WRITERS:
            with self.subTest(name=name):
                cfg = load_config(WRITERS / name)
                self.assertEqual(len(cfg["teachers"]), 3, name)
                self.assertTrue(is_multi_teacher_memory_method(cfg["pipeline"]["memory"]))
                self.assertEqual(cfg["orchestrator"]["pool"], POOL_EQUAL_WEIGHT)

    def test_every_writer_yaml_is_classified_as_single_or_multi(self):
        known = set(
            SINGLE_TEACHER_WRITERS + NO_TEACHER_WRITERS + MULTI_TEACHER_WRITERS
        )
        known.add("preprocess.yaml")
        found = {path.name for path in WRITERS.glob("*.yaml")}
        self.assertEqual(found, known)


class TestTeacherRosterRuntime(unittest.TestCase):
    def _fake_teacher(self, provider, model=None, **kwargs):
        return MockTeacher(
            model_name=model or "mock",
            teacher_id=kwargs.get("teacher_id") or provider,
        )

    def test_teacher_graph_yaml_builds_one_teacher_with_pool_single(self):
        cfg = load_config(WRITERS / "teacher_graph.yaml")
        args = build_parser().parse_args(
            ["--config", str(WRITERS / "teacher_graph.yaml")]
        )
        seen = []

        def fake_get_teacher(provider, model=None, **kwargs):
            seen.append((provider, model))
            return self._fake_teacher(provider, model=model, **kwargs)

        with patch("src.locomo_eval.run.get_teacher", side_effect=fake_get_teacher):
            orch = _build_orchestrator(cfg, args, "teacher_graph", "openai")
        self.assertEqual(len(seen), 1)
        self.assertEqual(orch.pool, POOL_SINGLE)
        n_teachers, ids = _teacher_cardinality(orch, None)
        self.assertEqual(n_teachers, 1)
        self.assertEqual(len(ids), 1)

    def test_teacher_graph_teacher_model_flag_stays_one_overlaid_model(self):
        cfg = load_config(WRITERS / "teacher_graph.yaml")
        args = build_parser().parse_args(
            [
                "--config",
                str(WRITERS / "teacher_graph.yaml"),
                "--teacher-model",
                "gpt-5",
            ]
        )
        roster = _teacher_roster(cfg, args, "teacher_graph", "openai")
        self.assertEqual(len(roster), 1)
        self.assertEqual(roster[0]["model"], "gpt-5")

    def test_teacher_model_flag_does_not_collapse_pooled_cheap_k3_roster(self):
        cfg = load_config(WRITERS / "pooled_teacher_graph.yaml")
        args = build_parser().parse_args(
            [
                "--config",
                str(WRITERS / "pooled_teacher_graph.yaml"),
                "--teacher-model",
                "gpt-4o-mini",
            ]
        )
        roster = _teacher_roster(cfg, args, "pooled_teacher_graph", "openai")
        self.assertEqual(len(roster), 3)
        self.assertEqual(
            [slot["id"] for slot in roster],
            ["openai", "anthropic", "deepseek"],
        )


class TestMockTeacherGraphRunIsSingleTeacher(unittest.TestCase):
    def test_mock_teacher_graph_run_meta_records_one_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "locomo.json"
            data_path.write_text(json.dumps(MINI), encoding="utf-8")
            out = Path(tmp) / "experiments"
            argv = [
                "--config",
                str(WRITERS / "teacher_graph.yaml"),
                "--reader",
                "mock",
                "--teacher",
                "mock",
                "--data",
                str(data_path),
                "--output-dir",
                str(out),
                "--run-id",
                "smoke_single_teacher",
                "--max-questions",
                "1",
            ]
            args = build_parser().parse_args(argv)
            cfg = load_config(args.config)
            cfg["data"]["raw_path"] = str(data_path)
            with redirect_stdout(io.StringIO()):
                run_dir = run_locomo_pipeline_with_memory_config(cfg, args)
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["teacher_pool"], "single")
            self.assertEqual(meta["n_teachers"], 1)
            self.assertEqual(len(meta["teacher_ids"]), 1)
            calls = [
                json.loads(line)
                for line in (run_dir / "memory" / "teacher_calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            call_ids = {row["teacher_id"] for row in calls}
            self.assertEqual(len(call_ids), 1)


class TestExperimentMatricesStaySingleTeacher(unittest.TestCase):
    def test_no_experiment_yaml_schedules_pooled_or_fused_methods(self):
        leaked = []
        for path in sorted(EXPERIMENTS.glob("*.yaml")):
            cfg = load_experiment_yaml(path)
            methods = (cfg.get("matrix") or {}).get("memory_method") or []
            if isinstance(methods, str):
                methods = [methods]
            for method in methods:
                if is_multi_teacher_memory_method(method):
                    leaked.append((path.name, method))
        self.assertEqual(leaked, [])

    def test_expand_rejects_pooled_teacher_graph_in_a_matrix(self):
        cfg = copy.deepcopy(
            load_experiment_yaml(EXPERIMENTS / "sandwich_memory.yaml")
        )
        cfg["matrix"]["memory_method"] = ["pooled_teacher_graph"]
        with self.assertRaises(ValueError) as ctx:
            expand_run_matrix(cfg)
        self.assertIn("single-teacher", str(ctx.exception))

    def test_openai_agent_writers_is_six_single_teacher_cells(self):
        specs = expand_run_matrix(
            load_experiment_yaml(EXPERIMENTS / "openai_agent_writers.yaml")
        )
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            {s.memory_method for s in specs},
            {"teacher_session_summaries", "teacher_graph"},
        )
        self.assertTrue(all(s.writer is not None for s in specs))

    def test_openai_codex_and_mem0_writer_matrices_stay_single_teacher(self):
        paths = [
            EXPERIMENTS / "openai_codex_writers.yaml",
            EXPERIMENTS / "openai_codex_poc_writers.yaml",
            EXPERIMENTS / "openai_agent_readers.yaml",
            EXPERIMENTS / "mem0_reader_2025_writers.yaml",
            EXPERIMENTS / "sandwich_memory.yaml",
            EXPERIMENTS / "teacher_ablation.yaml",
        ]
        for path in paths:
            with self.subTest(path=path.name):
                specs = expand_run_matrix(load_experiment_yaml(path))
                self.assertTrue(specs)
                for spec in specs:
                    self.assertFalse(
                        is_multi_teacher_memory_method(spec.memory_method),
                        spec.memory_method,
                    )


if __name__ == "__main__":
    unittest.main()
