"""Collect finished harness cells into experiments/<name>/aggregate/ (no API)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memorybench.aggregate_successful_runs import (
    collect_experiment_results,
    collect_full_run_packs,
)
from src.memorybench.execute_autorater_run import execute_autorater_run
from src.memorybench.execute_qa_run import execute_qa_run
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.gcs_run_workspace import (
    aggregate_prefix,
    collected_prefix,
    remote_run_prefix,
    upload_tree,
)
from src.memorybench.load_experiment_yaml import load_experiment_yaml

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


class FakeStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def exists(self, path: str) -> bool:
        return path.replace("\\", "/") in self.blobs

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.blobs[remote_path.replace("\\", "/")] = Path(local_path).read_bytes()

    def download(self, remote_path: str, local_path: Path) -> None:
        key = remote_path.replace("\\", "/")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.blobs[key])

    def list(self, prefix: str) -> list[str]:
        marker = prefix.strip("/").replace("\\", "/")
        out = []
        for key in self.blobs:
            if key == marker or key.startswith(marker + "/"):
                out.append(key)
        return sorted(out)


class TestCollectExperimentResultsWritesCatalogAndParquet(unittest.TestCase):
    def test_local_collect_after_qa_and_autorater_copies_audit_and_concat_parquet(self):
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            config = _write_poc(tmp)
            run_dir = execute_qa_run(config, run_index=0)
            execute_autorater_run(config, run_index=0)
            written = collect_experiment_results(config)
            agg = written["aggregate_dir"]
            self.assertTrue((agg / "_SUCCESS").is_file())
            self.assertTrue((agg / "SUMMARY.md").is_file())
            self.assertTrue((agg / "status.json").is_file())
            self.assertTrue((agg / "cells.jsonl").is_file())
            self.assertTrue((agg / "examples.parquet").is_file())
            self.assertTrue((agg / "runs.parquet").is_file())
            catalog = agg / "by_run" / run_dir.name
            self.assertTrue((catalog / "SUMMARY.md").is_file())
            self.assertTrue((catalog / "TRACE.md").is_file())
            self.assertTrue((catalog / "ATTRIBUTION.md").is_file())
            self.assertTrue((catalog / "autorater" / "SUMMARY.md").is_file())
            self.assertFalse((catalog / "memory" / "schema.json").is_file())
            status = json.loads((agg / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["expected_runs"], 1)
            self.assertEqual(status["qa_completed"], 1)
            self.assertEqual(status["autorater_completed"], 1)
            summary = (agg / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn(run_dir.name, summary)
            table = pq.read_table(agg / "examples.parquet")
            self.assertEqual(table.num_rows, 1)

    def test_gcs_collect_downloads_run_pack_and_uploads_aggregate_prefix(self):
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            config = _write_poc(tmp)
            run_dir = execute_qa_run(config, run_index=0)
            execute_autorater_run(config, run_index=0)
            cfg = load_experiment_yaml(config)
            spec = expand_run_matrix(cfg)[0]
            store = FakeStore()
            upload_tree(store, run_dir, remote_run_prefix(cfg, spec.run_id))

            gcs_yaml = tmp / "poc_gcs.yaml"
            gcs_yaml.write_text(
                config.read_text(encoding="utf-8").replace(
                    "backend: local",
                    "backend: gcs\n  bucket: test-bucket",
                ),
                encoding="utf-8",
            )
            scratch = tmp / "scratch"
            scratch.mkdir()
            gcs_cfg = load_experiment_yaml(gcs_yaml)
            with patch(
                "src.memorybench.aggregate_successful_runs.local_experiments_root",
                return_value=scratch,
            ):
                written = collect_experiment_results(gcs_yaml, store=store)
            agg = written["aggregate_dir"]
            self.assertTrue((scratch / spec.run_id / "_SUCCESS").is_file())
            self.assertTrue((agg / "by_run" / spec.run_id / "SUMMARY.md").is_file())
            prefix = aggregate_prefix(gcs_cfg)
            self.assertIn(f"{prefix}/SUMMARY.md", store.blobs)
            self.assertIn(f"{prefix}/examples.parquet", store.blobs)
            self.assertIn(f"{prefix}/_SUCCESS", store.blobs)

    def test_full_collect_copies_memory_dump_and_predictions_jsonl(self):
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            config = _write_poc(tmp)
            run_dir = execute_qa_run(config, run_index=0)
            written = collect_full_run_packs(config)
            dest = written["aggregate_dir"] / "runs" / run_dir.name
            self.assertTrue((dest / "memory" / "schema.json").is_file())
            self.assertTrue((dest / "predictions.jsonl").is_file())
            self.assertTrue((dest / "reader" / "traces.jsonl").is_file())
            summary = (written["aggregate_dir"] / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("full", summary)
            self.assertFalse(
                (written["aggregate_dir"] / "by_run" / run_dir.name / "SUMMARY.md").is_file()
            )

    def test_gcs_full_collect_uploads_memory_under_collected_prefix(self):
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            config = _write_poc(tmp)
            run_dir = execute_qa_run(config, run_index=0)
            cfg = load_experiment_yaml(config)
            spec = expand_run_matrix(cfg)[0]
            store = FakeStore()
            upload_tree(store, run_dir, remote_run_prefix(cfg, spec.run_id))
            gcs_yaml = tmp / "poc_gcs.yaml"
            gcs_yaml.write_text(
                config.read_text(encoding="utf-8").replace(
                    "backend: local",
                    "backend: gcs\n  bucket: test-bucket",
                ),
                encoding="utf-8",
            )
            scratch = tmp / "scratch"
            scratch.mkdir()
            gcs_cfg = load_experiment_yaml(gcs_yaml)
            with patch(
                "src.memorybench.aggregate_successful_runs.local_experiments_root",
                return_value=scratch,
            ):
                collect_full_run_packs(gcs_yaml, store=store)
            prefix = collected_prefix(gcs_cfg)
            self.assertIn(f"{prefix}/runs/{spec.run_id}/memory/schema.json", store.blobs)
            self.assertIn(f"{prefix}/runs/{spec.run_id}/predictions.jsonl", store.blobs)
            self.assertNotIn(f"{prefix}/by_run/{spec.run_id}/SUMMARY.md", store.blobs)


if __name__ == "__main__":
    unittest.main()
