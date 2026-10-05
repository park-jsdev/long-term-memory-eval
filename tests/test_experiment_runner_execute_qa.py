"""Mock QA + autorater through the harness (no paid API)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_runner.execute_autorater_run import execute_autorater_run
from src.experiment_runner.execute_qa_run import _qa_argv, execute_qa_run
from src.experiment_runner.experiment_run_spec import (
    ExperimentRunSpec,
    ReaderModelRef,
    WriterModelRef,
)
from src.experiment_runner.report_experiment_status import report_experiment_status

GOLD = "UNIQ_GOLD_REF_ZZZ"
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
                "answer": GOLD,
                "category": 4,
                "evidence": ["D1:1"],
            }
        ],
    }
]


def _write_poc(tmp: Path) -> Path:
    data = tmp / "locomo.json"
    data.write_text(json.dumps(MINI), encoding="utf-8")
    yaml_path = tmp / "poc.yaml"
    yaml_path.write_text(
        f"""
experiment:
  name: harness-smoke
  type: calibration
  freeze:
    prompt_path: prompts/readers/qa_mem0_v1.txt
    reader:
      display_name: GPT-4o mini
      provider: openai
      family: openai
      generation: 2024
      api_model_id: gpt-4o-mini
      status: runnable
benchmark:
  name: locomo
  dataset_path: {data.as_posix()}
matrix:
  memory_method:
    - full_context
  seed:
    - 1
subset:
  max_questions: 1
judge:
  provider: mock
  api_model_id: mock
execution:
  reader_provider: mock
  judge_provider: mock
storage:
  backend: local
  path: {(tmp / "experiments").as_posix()}
""",
        encoding="utf-8",
    )
    return yaml_path


class TestExecuteQaWritesAuditAndParquet(unittest.TestCase):
    def test_mock_qa_writes_success_jsonl_and_parquet_then_skips(self):
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            config = _write_poc(tmp)
            run_dir = execute_qa_run(config, run_index=0)
            self.assertTrue((run_dir / "_SUCCESS").is_file())
            self.assertTrue((run_dir / "predictions.jsonl").is_file())
            self.assertTrue((run_dir / "examples.parquet").is_file())
            self.assertTrue((run_dir / "summary.parquet").is_file())
            self.assertTrue((run_dir / "run.json").is_file())
            table = pq.read_table(run_dir / "examples.parquet")
            self.assertEqual(table.num_rows, 1)
            self.assertIn("retrieved_memories", table.column_names)
            self.assertIn("generated_answer", table.column_names)

            with patch(
                "src.experiment_runner.execute_qa_run.run_locomo_pipeline_with_memory_config"
            ) as mocked:
                execute_qa_run(config, run_index=0)
                mocked.assert_not_called()

            execute_qa_run(config, run_index=0, force=True)
            self.assertTrue((run_dir / "_SUCCESS").is_file())

    def test_autorater_requires_qa_success_then_writes_judge_marker(self):
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            config = _write_poc(tmp)
            with self.assertRaises(SystemExit):
                execute_autorater_run(config, run_index=0)
            run_dir = execute_qa_run(config, run_index=0)
            execute_autorater_run(config, run_index=0)
            self.assertTrue((run_dir / "autorater" / "_SUCCESS").is_file())
            self.assertTrue((run_dir / "autorater" / "autorater_verdicts.jsonl").is_file())
            report = report_experiment_status(config)
            self.assertEqual(report["expected_runs"], 1)
            self.assertEqual(report["qa_completed"], 1)
            self.assertEqual(report["autorater_completed"], 1)


def _reader() -> ReaderModelRef:
    return ReaderModelRef(
        display_name="GPT-4o",
        provider="openai",
        family="openai",
        generation=2024,
        api_model_id="gpt-4o-2024-08-06",
        catalog_id="gpt-4o",
    )


def _spec(*, memory_method: str, writer: WriterModelRef | None) -> ExperimentRunSpec:
    return ExperimentRunSpec(
        experiment_name="locomo-mem0-reader-2024-writers",
        experiment_type="frozen_reader",
        run_index=0,
        run_id="cell-1",
        benchmark="locomo",
        memory_method=memory_method,
        reader=_reader(),
        seed=1,
        prompt_path="prompts/readers/qa_mem0_v1.txt",
        judge_provider="openai",
        judge_model="gpt-4o-mini",
        writer=writer,
        method_yaml="configs/writers/graph.yaml"
        if memory_method == "graph"
        else "configs/writers/session_summaries.yaml"
        if memory_method == "session_summaries"
        else "configs/writers/full_context.yaml",
    )


class TestQaArgvPassesWriter(unittest.TestCase):
    def test_graph_argv_includes_provider_and_model(self):
        writer = WriterModelRef(
            display_name="Claude 3.5 Sonnet",
            provider="anthropic",
            api_model_id="claude-3-5-sonnet-20241022",
            catalog_id="claude-3-5-sonnet",
        )
        argv = _qa_argv(
            _spec(memory_method="graph", writer=writer),
            {"benchmark": {"dataset_path": "data/raw/locomo10.json"}},
            Path("/tmp/out"),
            None,
        )
        self.assertEqual(argv[argv.index("--writer") + 1], "anthropic")
        self.assertEqual(
            argv[argv.index("--writer-model") + 1], "claude-3-5-sonnet-20241022"
        )
        self.assertEqual(argv[argv.index("--model") + 1], "gpt-4o-2024-08-06")
        self.assertNotIn("--thinking", argv)
        self.assertNotIn("--writer-max-tokens", argv)

    def test_gpt5_writer_argv_turns_thinking_off_and_raises_completion_cap(self):
        writer = WriterModelRef(
            display_name="GPT-5",
            provider="openai",
            api_model_id="gpt-5",
            catalog_id="gpt-5",
            thinking=False,
            max_tokens=8192,
        )
        argv = _qa_argv(
            _spec(memory_method="session_summaries", writer=writer),
            {},
            Path("/tmp/out"),
            None,
        )
        self.assertEqual(argv[argv.index("--thinking") + 1], "off")
        self.assertEqual(argv[argv.index("--writer-max-tokens") + 1], "8192")

    def test_full_context_argv_passes_reader_thinking_and_max_tokens(self):
        reader = ReaderModelRef(
            display_name="GPT-5",
            provider="openai",
            family="openai",
            generation="2025",
            api_model_id="gpt-5",
            catalog_id="gpt-5",
            thinking=True,
            max_tokens=8192,
        )
        spec = ExperimentRunSpec(
            experiment_name="locomo-2025-readers-openai-deepseek",
            experiment_type="sweep",
            run_index=0,
            run_id="cell-think",
            benchmark="locomo",
            memory_method="full_context",
            reader=reader,
            seed=1,
            prompt_path="prompts/readers/qa_mem0_v1.txt",
            judge_provider="openai",
            judge_model="gpt-4o-mini",
            method_yaml="configs/writers/full_context.yaml",
        )
        argv = _qa_argv(spec, {}, Path("/tmp/out"), None)
        self.assertEqual(argv[argv.index("--reader-thinking") + 1], "on")
        self.assertEqual(argv[argv.index("--max-tokens") + 1], "8192")
        self.assertNotIn("--thinking", argv)

    def test_full_context_argv_omits_writer_flags(self):
        argv = _qa_argv(
            _spec(memory_method="full_context", writer=None),
            {},
            Path("/tmp/out"),
            None,
        )
        self.assertNotIn("--writer", argv)
        self.assertNotIn("--writer-model", argv)


if __name__ == "__main__":
    unittest.main()
