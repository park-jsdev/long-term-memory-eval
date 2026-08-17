"""Regression locks for sandwich contracts that must not change silently.

If a test here fails, treat it as a redesign (update SPEC / AGENTS / HUMANS),
not a green-light to weaken the assertion.

Frozen contracts:
  - run_locomo_pipeline_with_memory_config runs **one** memory YAML → one audit pack
  - A vs B is two of those calls, then scripts/compare_runs.py
  - Gold answers are scorer-only (never in Memory.text or the reader prompt)
  - Swapping C0/C1 changes memory, not reader_model / prompt_version / metric names
  - Audit pack, prediction fields, and dual metrics (EM / token F1 / LoCoMo F1)
  - Same function for 1 question or N; resume from predictions.jsonl
  - offline_evaluate.py rescores strings only (does not rewrite predicted answers)

Offline / mock only — no live API.
"""

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

from scripts.compare_runs import load_pack, pair_analysis
from src.locomo_eval.utils.llm_response_cache import (
    PIPELINE_STAGE_ANSWER_READER,
    PIPELINE_STAGE_TEACHER,
    LlmResponseCache,
    llm_response_cache_dir_from_run_cfg,
)
from src.locomo_eval.offline_evaluate import main as offline_evaluate_main
from src.locomo_eval.run import run_locomo_pipeline_with_memory_config


# Token that must never leak into memory or the filled reader prompt.
GOLD_LOCK = "UNIQ_GOLD_REF_ZZZ"

FROZEN_PREDICTION_FIELDS = (
    "sample_id",
    "question_id",
    "question",
    "reference_answer",
    "predicted_answer",
    "category",
    "memory_type",
    "memory_text",
    "reader_model",
    "prompt_version",
)

FROZEN_AUDIT_FILES = (
    "predictions.jsonl",
    "predictions.csv",
    "metrics.json",
    "metrics_by_category.csv",
    "run_meta.json",
)

FROZEN_METRIC_NAMES = ("exact_match", "token_f1", "locomo_f1")

MINI = {
    "sample_id": "conv-regress",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
        ],
    },
    "session_summary": {
        "session_1_summary": "Alice said she started painting.",
    },
    "qa": [
        {
            "question": "What hobby did Alice begin?",
            "answer": GOLD_LOCK,
            "category": 4,
            "evidence": ["D1:1"],
        },
        {
            "question": "Who is the other speaker?",
            "answer": GOLD_LOCK,
            "category": 4,
            "evidence": [],
        },
    ],
}


def _write_mini(path: Path) -> Path:
    path.write_text(json.dumps([MINI]), encoding="utf-8")
    return path


def _cfg(*, data_path: Path, memory: str) -> dict:
    return {
        "data": {"raw_path": str(data_path), "locomo_commit": "regression-lock"},
        "pipeline": {
            "memory": memory,
            "prompt_path": str(ROOT / "prompts" / "qa_v1.txt"),
            "max_questions": None,
        },
        "reader": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.0,
            "max_tokens": 64,
            "max_retries": 1,
            "min_request_interval_s": 0.0,
            "max_wait_s": 1.0,
        },
        "run": {
            "run_id": None,
            "output_dir": "experiments",
        },
    }


def _overrides(
    *,
    data_path: Path,
    output_dir: Path,
    run_id: str,
    memory: str | None = None,
    max_questions: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        data=str(data_path),
        memory=memory,
        reader="mock",
        model=None,
        prompt=None,
        output_dir=str(output_dir),
        run_id=run_id,
        max_questions=max_questions,
        sample_id=None,
    )


def _run_one(
    tmp: Path,
    *,
    run_id: str,
    memory: str,
    max_questions: int | None = 1,
) -> Path:
    data_path = tmp / "locomo.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    _write_mini(data_path)
    out = tmp / "experiments"
    cfg = _cfg(data_path=data_path, memory=memory)
    overrides = _overrides(
        data_path=data_path,
        output_dir=out,
        run_id=run_id,
        max_questions=max_questions,
    )
    with redirect_stdout(io.StringIO()):
        return run_locomo_pipeline_with_memory_config(cfg, overrides)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class TestRunLocomoPipelineWithMemoryConfig(unittest.TestCase):
    """Locks the single-config runner. Not a comparison orchestrator."""

    def test_run_locomo_pipeline_with_memory_config_writes_required_audit_pack_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_audit", memory="c1_session_summary")
            for name in FROZEN_AUDIT_FILES:
                self.assertTrue((run_dir / name).is_file(), f"missing {name}")
            self.assertTrue((run_dir / "plots").is_dir())
            self.assertTrue((run_dir / "memory").is_dir())

    def test_run_locomo_pipeline_with_memory_config_returns_a_directory_named_only_by_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_one_id", memory="c1_session_summary")
            self.assertEqual(run_dir.name, "lock_one_id")
            self.assertEqual(len(list(run_dir.parent.iterdir())), 1)

    def test_run_locomo_pipeline_with_memory_config_records_exactly_one_memory_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_mem", memory="c1_session_summary")
            meta = _load_json(run_dir / "run_meta.json")
            metrics = _load_json(run_dir / "metrics.json")
            self.assertEqual(meta["memory_type"], "c1_session_summary")
            self.assertEqual(metrics["memory_type"], "c1_session_summary")
            rows = _load_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual({r["memory_type"] for r in rows}, {"c1_session_summary"})

    def test_prediction_jsonl_rows_include_frozen_audit_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_fields", memory="c1_session_summary")
            rows = _load_jsonl(run_dir / "predictions.jsonl")
            self.assertTrue(rows)
            for field in FROZEN_PREDICTION_FIELDS:
                self.assertIn(field, rows[0], f"prediction missing {field}")

    def test_metrics_json_reports_exact_match_token_f1_and_locomo_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_metrics", memory="c1_session_summary")
            overall = _load_json(run_dir / "metrics.json")["metrics"]
            for name in FROZEN_METRIC_NAMES:
                self.assertIn(name, overall)

    def test_memory_text_and_filled_prompt_do_not_contain_the_gold_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_gold", memory="c1_session_summary")
            row = _load_jsonl(run_dir / "predictions.jsonl")[0]
            self.assertEqual(row["reference_answer"], GOLD_LOCK)
            self.assertNotIn(GOLD_LOCK, row["memory_text"])
            filled = (run_dir / "memory" / "prompt_fill_example.txt").read_text(encoding="utf-8")
            self.assertNotIn(GOLD_LOCK, filled)
            self.assertNotIn("{memory}", filled)
            self.assertNotIn("{question}", filled)
            self.assertIn("What hobby did Alice begin?", filled)

    def test_max_questions_one_and_two_both_call_run_locomo_pipeline_with_memory_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = _run_one(root / "n1", run_id="n1", memory="c1_session_summary", max_questions=1)
            two = _run_one(root / "n2", run_id="n2", memory="c1_session_summary", max_questions=2)
            self.assertEqual(_load_json(one / "run_meta.json")["n_predictions"], 1)
            self.assertEqual(_load_json(two / "run_meta.json")["n_predictions"], 2)

    def test_second_call_with_the_same_run_id_resumes_finished_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _run_one(root, run_id="resume_lock", memory="c1_session_summary", max_questions=1)
            n_first = _load_json(first / "run_meta.json")["n_new_api_calls"]
            second = _run_one(root, run_id="resume_lock", memory="c1_session_summary", max_questions=1)
            meta = _load_json(second / "run_meta.json")
            self.assertGreaterEqual(n_first, 1)
            self.assertEqual(meta["n_resumed"], 1)
            self.assertEqual(meta["n_new_api_calls"], 0)
            self.assertEqual(second, first)

    def test_run_meta_omits_llm_response_cache_fields_while_cache_is_unwired(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="no_cache", memory="c1_session_summary")
            meta = _load_json(run_dir / "run_meta.json")
            row = _load_jsonl(run_dir / "predictions.jsonl")[0]
            self.assertNotIn("n_llm_response_cache_hits", meta)
            self.assertNotIn("llm_response_cache_dir", meta)
            self.assertFalse(row.get("cached"))


class TestLlmResponseCacheIsStageSpecific(unittest.TestCase):
    """Locks the unused-but-kept cache utility (not wired into run.py)."""

    def test_llm_response_cache_dir_from_run_cfg_prefers_new_key_over_legacy_cache_dir(self):
        path = llm_response_cache_dir_from_run_cfg(
            {"llm_response_cache_dir": "experiments/llm_cache", "cache_dir": "experiments/old"}
        )
        self.assertEqual(path, "experiments/llm_cache")

    def test_llm_response_cache_dir_from_run_cfg_falls_back_to_legacy_cache_dir(self):
        path = llm_response_cache_dir_from_run_cfg({"cache_dir": "experiments/old"})
        self.assertEqual(path, "experiments/old")

    def test_make_key_differs_when_pipeline_stage_is_answer_reader_vs_teacher(self):
        shared = {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.0,
            "max_tokens": 64,
            "prompt": "same body",
        }
        reader_key = LlmResponseCache.make_key(
            {"pipeline_stage": PIPELINE_STAGE_ANSWER_READER, **shared}
        )
        teacher_key = LlmResponseCache.make_key(
            {"pipeline_stage": PIPELINE_STAGE_TEACHER, **shared}
        )
        self.assertNotEqual(reader_key, teacher_key)


class TestSandwichMemorySwapDoesNotRetouchReaderOrMetrics(unittest.TestCase):
    def test_c0_and_c1_runs_share_reader_model_and_prompt_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c0 = _run_one(root / "c0", run_id="c0", memory="c0_raw")
            c1 = _run_one(root / "c1", run_id="c1", memory="c1_session_summary")
            m0 = _load_json(c0 / "run_meta.json")
            m1 = _load_json(c1 / "run_meta.json")
            self.assertEqual(m0["reader_model"], m1["reader_model"])
            self.assertEqual(m0["prompt_version"], m1["prompt_version"])
            self.assertEqual(m0["prompt_version"], "qa_v1")

    def test_c0_and_c1_runs_produce_different_memory_text_for_the_same_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c0 = _run_one(root / "c0", run_id="c0", memory="c0_raw")
            c1 = _run_one(root / "c1", run_id="c1", memory="c1_session_summary")
            t0 = _load_jsonl(c0 / "predictions.jsonl")[0]["memory_text"]
            t1 = _load_jsonl(c1 / "predictions.jsonl")[0]["memory_text"]
            self.assertNotEqual(t0, t1)
            self.assertIn("I started painting", t0)
            self.assertIn("Alice said she started painting", t1)

    def test_c0_and_c1_metrics_json_use_the_same_metric_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c0 = _run_one(root / "c0", run_id="c0", memory="c0_raw")
            c1 = _run_one(root / "c1", run_id="c1", memory="c1_session_summary")
            k0 = set(_load_json(c0 / "metrics.json")["metrics"])
            k1 = set(_load_json(c1 / "metrics.json")["metrics"])
            self.assertEqual(k0, k1)
            self.assertTrue(set(FROZEN_METRIC_NAMES) <= k0)


class TestCompareAndOfflineEvaluateAfterTwoPipelineRuns(unittest.TestCase):
    def test_compare_runs_pair_analysis_marks_c0_and_c1_memory_as_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c0 = _run_one(root / "c0", run_id="c0", memory="c0_raw")
            c1 = _run_one(root / "c1", run_id="c1", memory="c1_session_summary")
            packs = [load_pack(c0), load_pack(c1)]
            paired = pair_analysis(packs, ROOT / "prompts" / "qa_v1.txt")
            self.assertEqual(paired["n_common"], 1)
            self.assertEqual(paired["fraction_same_memory_text"], 0.0)
            self.assertNotEqual(packs[0]["meta"]["memory_type"], packs[1]["meta"]["memory_type"])

    def test_offline_evaluate_rewrites_metrics_but_not_predicted_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _run_one(root, run_id="rescore", memory="c1_session_summary")
            before = _load_jsonl(run_dir / "predictions.jsonl")
            out = root / "rescored"
            with redirect_stdout(io.StringIO()):
                offline_evaluate_main(
                    ["--predictions", str(run_dir / "predictions.jsonl"), "--output-dir", str(out)]
                )
            after = _load_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual(before[0]["predicted_answer"], after[0]["predicted_answer"])
            rescored = _load_json(out / "metrics.json")["metrics"]
            for name in FROZEN_METRIC_NAMES:
                self.assertIn(name, rescored)


if __name__ == "__main__":
    unittest.main()
