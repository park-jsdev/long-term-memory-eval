"""Read-only experiment pack loaders (no API, no run.py)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.experiments.layout import PackPaths, audit_layout_meta
from src.locomo_eval.experiments.load import (
    load_experiment_pack,
    load_qa_pack,
    predictions_jsonl,
    resolve_predictions_jsonl,
)
from src.locomo_eval.experiments.write import write_reader_module, write_teacher_module


class TestPredictionsJsonlPrefersRootThenReader(unittest.TestCase):
    def test_predictions_jsonl_returns_root_file_when_both_exist(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "predictions.jsonl").write_text(
                json.dumps({"question_id": "root"}) + "\n", encoding="utf-8"
            )
            reader = run / "reader"
            reader.mkdir()
            (reader / "predictions.jsonl").write_text(
                json.dumps({"question_id": "reader"}) + "\n", encoding="utf-8"
            )
            path = predictions_jsonl(run)
            self.assertEqual(path, run / "predictions.jsonl")

    def test_predictions_jsonl_falls_back_to_reader_when_root_missing(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            reader = run / "reader"
            reader.mkdir(parents=True)
            (reader / "predictions.jsonl").write_text(
                json.dumps({"question_id": "reader"}) + "\n", encoding="utf-8"
            )
            path = predictions_jsonl(run)
            self.assertEqual(path, reader / "predictions.jsonl")
            resolved = resolve_predictions_jsonl(run)
            self.assertEqual(resolved, path)


class TestLoadQaPackAndExperimentPack(unittest.TestCase):
    def test_load_qa_pack_returns_compare_full_runs_keys(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "metrics.json").write_text(
                json.dumps({"metrics": {"locomo_f1": 0.5}}), encoding="utf-8"
            )
            (run / "run_meta.json").write_text(
                json.dumps({"run_id": "r1", "audit_layout": audit_layout_meta()}),
                encoding="utf-8",
            )
            (run / "predictions.jsonl").write_text(
                json.dumps({"question_id": "q0", "predicted_answer": "a"}) + "\n",
                encoding="utf-8",
            )
            pack = load_qa_pack(run)
            self.assertEqual(
                set(pack),
                {"dir", "run_id", "metrics", "meta", "predictions", "by_qid"},
            )
            self.assertEqual(pack["run_id"], "r1")
            self.assertEqual(pack["by_qid"]["q0"]["predicted_answer"], "a")

    def test_load_experiment_pack_indexes_teacher_calls_and_fusion(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "metrics.json").write_text("{}", encoding="utf-8")
            (run / "run_meta.json").write_text("{}", encoding="utf-8")
            write_reader_module(
                run,
                prediction_rows=[{"question_id": "q0", "sample_id": "s1"}],
                traces=[{"question_id": "q0", "role": "reader", "reasoning": ""}],
            )
            write_teacher_module(
                run,
                calls=[
                    {
                        "sample_id": "s1",
                        "session_id": 1,
                        "teacher_id": "openai",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "reasoning": "because",
                        "relations": [
                            {"source": "alice", "relationship": "likes", "target": "pizza"}
                        ],
                    }
                ],
                fusion_rows=[
                    {
                        "sample_id": "s1",
                        "session_id": 1,
                        "teacher_ids": ["openai"],
                        "relations": [
                            {
                                "source": "alice",
                                "relationship": "likes",
                                "target": "pizza",
                                "proposed_by": ["openai"],
                                "votes": 1,
                                "kept": True,
                            }
                        ],
                    }
                ],
            )
            pack = load_experiment_pack(run)
            self.assertEqual(len(pack.teacher_calls_for("openai")), 1)
            self.assertEqual(pack.teacher_calls_for("openai")[0]["reasoning"], "because")
            kept = pack.fusion_kept_for(sample_id="s1")
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0]["proposed_by"], ["openai"])
            self.assertTrue(PackPaths.from_run_dir(run).teacher_calls.is_file())


if __name__ == "__main__":
    unittest.main()
