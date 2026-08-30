"""Deterministic preprocess write-index (no LLM) + dump-backed memory format."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.dataset import parse_sample
from src.locomo_eval.memory import (
    RawConversationMemoryBuilder,
    SessionSummaryMemoryBuilder,
    get_memory_builder,
)
from src.locomo_eval.preprocess.dump import MissingPreprocessIndexError
from src.locomo_eval.preprocess.run_index import run_preprocess_index
from src.locomo_eval.run import run_locomo_pipeline_with_memory_config

GOLD_LOCK = "UNIQ_GOLD_REF_ZZZ"

MINI = {
    "sample_id": "conv-pre-idx",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice!"},
        ],
        "session_2_date_time": "2 Jan 2023",
        "session_2": [],
        "session_3_date_time": "1:56 pm on 8 May, 2023",
        "session_3": [
            {"dia_id": "D3:1", "speaker": "Alice", "text": "I got a nursing job."},
        ],
    },
    "session_summary": {
        "session_1_summary": "Alice said she started painting.",
        "session_3_summary": "Alice got a nursing job.",
    },
    "observation": {},
    "qa": [
        {
            "question": "What hobby did Alice begin?",
            "answer": GOLD_LOCK,
            "category": 4,
            "evidence": ["D1:1"],
        }
    ],
}


def _write_mini(path: Path) -> Path:
    path.write_text(json.dumps([MINI]), encoding="utf-8")
    return path


def _index_cfg(data_path: Path, output_dir: Path) -> dict:
    return {
        "data": {"raw_path": str(data_path), "locomo_commit": "test"},
        "preprocess": {"index_run_id": "locomo_preprocess", "top_k": None},
        "run": {"run_id": "locomo_preprocess", "output_dir": str(output_dir)},
    }


def _index_ns(data_path: Path, output_dir: Path, **extra) -> argparse.Namespace:
    kwargs = {
        "data": str(data_path),
        "output_dir": str(output_dir),
        "run_id": "locomo_preprocess",
        "sample_id": None,
        "max_samples": None,
        "eval_questions": None,
        "eval_reader": None,
        "eval_configs": None,
    }
    kwargs.update(extra)
    return argparse.Namespace(**kwargs)


class TestPreprocessRunIndexWritesGoldFreeDump(unittest.TestCase):
    def test_run_preprocess_index_writes_sessions_and_documents_without_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = _write_mini(root / "locomo.json")
            out = root / "experiments"
            index_root = run_preprocess_index(
                _index_cfg(data_path, out), _index_ns(data_path, out)
            )
            sample_dir = index_root / "by_sample" / "conv-pre-idx"
            self.assertTrue((sample_dir / "sessions.jsonl").is_file())
            self.assertTrue((sample_dir / "documents.jsonl").is_file())
            self.assertTrue((index_root / "run_meta.json").is_file())
            blob = (sample_dir / "sessions.jsonl").read_text(
                encoding="utf-8"
            ) + (sample_dir / "documents.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(GOLD_LOCK, blob)
            meta = json.loads((index_root / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["llm_calls"], 0)
            self.assertFalse(meta["gold_answer_in_index"])
            docs = [
                json.loads(line)
                for line in (sample_dir / "documents.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual([d["session_id"] for d in docs], [1, 3])
            self.assertIn("painting", docs[0]["session_summary"])

    def test_run_preprocess_index_regenerates_when_run_id_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = _write_mini(root / "locomo.json")
            out = root / "experiments"
            cfg = _index_cfg(data_path, out)
            first = run_preprocess_index(cfg, _index_ns(data_path, out))
            meta1 = json.loads((first / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta1["n_samples"], 1)
            stale = first / "stale.txt"
            stale.write_text("old", encoding="utf-8")
            second = run_preprocess_index(cfg, _index_ns(data_path, out))
            self.assertFalse(stale.exists())
            meta2 = json.loads((second / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta2["n_samples"], 1)
            self.assertNotIn("regenerated_from_scratch", meta2)
            self.assertNotIn("n_skipped", meta2)


class TestDumpBackedMemoryMatchesConversationBuilders(unittest.TestCase):
    def test_raw_chunks_from_dump_matches_conversation_builder_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = _write_mini(root / "locomo.json")
            out = root / "experiments"
            index_root = run_preprocess_index(
                _index_cfg(data_path, out), _index_ns(data_path, out)
            )
            conv = parse_sample(MINI)
            q = conv.questions[0]
            live = RawConversationMemoryBuilder().build(conv, q)
            dumped = get_memory_builder(
                "raw_chunks", preprocess_index_dir=index_root
            ).build(conv, q)
            self.assertEqual(dumped.text, live.text)
            self.assertEqual(dumped.memory_type, "raw_chunks")

    def test_session_summaries_from_dump_matches_conversation_builder_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = _write_mini(root / "locomo.json")
            out = root / "experiments"
            index_root = run_preprocess_index(
                _index_cfg(data_path, out), _index_ns(data_path, out)
            )
            conv = parse_sample(MINI)
            q = conv.questions[0]
            live = SessionSummaryMemoryBuilder().build(conv, q)
            dumped = get_memory_builder(
                "session_summaries", preprocess_index_dir=index_root
            ).build(conv, q)
            self.assertEqual(dumped.text, live.text)
            self.assertNotIn(GOLD_LOCK, dumped.text)

    def test_builder_raises_missing_index_error_that_names_run_index(self):
        conv = parse_sample(MINI)
        builder = get_memory_builder(
            "raw_chunks", preprocess_index_dir=Path("/no/such/preprocess")
        )
        with self.assertRaises(MissingPreprocessIndexError) as ctx:
            builder.build(conv, conv.questions[0])
        self.assertIn("run_index", str(ctx.exception))


class TestDumpBackedEvalSmoke(unittest.TestCase):
    def test_run_pipeline_uses_preprocess_dump_for_session_summaries_mock_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = _write_mini(root / "locomo.json")
            out = root / "experiments"
            run_preprocess_index(_index_cfg(data_path, out), _index_ns(data_path, out))
            cfg = {
                "data": {"raw_path": str(data_path), "locomo_commit": "test"},
                "pipeline": {
                    "memory": "session_summaries",
                    "prompt_path": str(ROOT / "prompts" / "qa_mem0_v1.txt"),
                    "max_questions": 1,
                },
                "reader": {
                    "provider": "mock",
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "max_retries": 1,
                    "min_request_interval_s": 0.0,
                    "max_wait_s": 1.0,
                },
                "run": {"run_id": None, "output_dir": str(out)},
            }
            ns = argparse.Namespace(
                data=str(data_path),
                memory=None,
                reader="mock",
                model=None,
                teacher=None,
                teacher_model=None,
                prompt=None,
                output_dir=str(out),
                run_id="eval_ss_n1",
                max_questions=1,
                sample_id=None,
                mem0_index_run_id=None,
                preprocess_index_run_id="locomo_preprocess",
                retrieve_top_k=None,
            )
            with redirect_stdout(io.StringIO()):
                run_dir = run_locomo_pipeline_with_memory_config(cfg, ns)
            rows = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["memory_type"], "session_summaries")
            self.assertIn("painting", rows[0]["memory_text"])
            self.assertNotIn(GOLD_LOCK, rows[0]["memory_text"])
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["preprocess_index_run_id"], "locomo_preprocess")
            self.assertEqual(meta["n_new_api_calls"], 1)

    def test_run_index_eval_questions_writes_one_prediction_per_memory_with_mock_reader(self):
        from src.locomo_eval.preprocess.run_index import main as preprocess_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = _write_mini(root / "locomo.json")
            out = root / "experiments"
            prompt = ROOT / "prompts" / "qa_mem0_v1.txt"
            yamls = []
            for memory in ("raw_chunks", "session_summaries"):
                yml = root / f"{memory}.yaml"
                yml.write_text(
                    "\n".join(
                        [
                            "data:",
                            f"  raw_path: {data_path.as_posix()}",
                            "  locomo_commit: test",
                            "pipeline:",
                            f"  memory: {memory}",
                            f"  prompt_path: {prompt.as_posix()}",
                            "  max_questions: null",
                            "reader:",
                            "  provider: mock",
                            "  model: gpt-4o-mini",
                            "  temperature: 0.0",
                            "  max_tokens: 64",
                            "  max_retries: 1",
                            "  min_request_interval_s: 0.0",
                            "  max_wait_s: 1.0",
                            "run:",
                            "  run_id: null",
                            f"  output_dir: {out.as_posix()}",
                        ]
                    ),
                    encoding="utf-8",
                )
                yamls.append(str(yml))
            index_cfg = root / "preprocess.yaml"
            index_cfg.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_path: {data_path.as_posix()}",
                        "  locomo_commit: test",
                        "run:",
                        "  run_id: smoke_preprocess",
                        f"  output_dir: {out.as_posix()}",
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                preprocess_main(
                    [
                        "--config",
                        str(index_cfg),
                        "--run-id",
                        "smoke_preprocess",
                        "--eval-questions",
                        "1",
                        "--eval-reader",
                        "mock",
                        "--eval-configs",
                        *yamls,
                    ]
                )
            for memory in ("raw_chunks", "session_summaries"):
                pred = out / f"smoke_preprocess_{memory}_n1" / "predictions.jsonl"
                self.assertTrue(pred.is_file(), pred)
                rows = [
                    json.loads(line)
                    for line in pred.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["memory_type"], memory)
                self.assertNotIn(GOLD_LOCK, rows[0]["memory_text"])


if __name__ == "__main__":
    unittest.main()
