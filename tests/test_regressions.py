"""Regression locks for sandwich contracts that must not change silently.

If a test here fails, treat it as a redesign (update SPEC / AGENTS / HUMANS),
not a green-light to weaken the assertion.

Frozen contracts:
  - run_locomo_pipeline_with_memory_config runs **one** memory YAML → one audit pack
  - A vs B is two of those calls, then scripts/compare_full_runs.py
  - Gold answers are scorer-only (never in Memory.text or the reader prompt)
  - Swapping raw_chunks/session_summaries changes memory, not reader_model / prompt_version / metric names
  - Audit pack, prediction fields, and dual metrics (EM / token F1 / LoCoMo F1)
  - Same function for 1 question or N; every invocation regenerates
  - offline_evaluate.py rescores strings only (does not rewrite predicted answers)
  - Claim-audit joins are keyed by sample (and session when present); optional
    filters omitted mean "no restriction", never a phantom restriction or a leak
  - Retrieve losers never enter lineage; reusing a run id clears attribution leftovers

Offline / mock only — no live API.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_full_runs import load_pack, pair_analysis, resolve_prompt_path
from src.config import load_config
from src.locomo_eval.readers import (
    READER_MESSAGE_LAYOUT_MEM0,
    build_reader_messages,
)
from src.locomo_eval.offline_evaluate import main as offline_evaluate_main
from src.locomo_eval.experiments.audit_loader import SandwichAudit, resolve_predictions_jsonl
from src.locomo_eval.experiments.audit_layout import AuditPaths
from src.locomo_eval.experiments.claim_audit import lineage_rows
from src.locomo_eval.run import _reset_run_output, build_parser, run_locomo_pipeline_with_memory_config
from src.locomo_eval.schemas import Memory


# Token that must never leak into memory or the filled reader prompt.
GOLD_LOCK = "UNIQ_GOLD_REF_ZZZ"


class TestMem0ParityBaselineConfig(unittest.TestCase):
    def test_cli_default_selects_mem0_baseline_config(self):
        self.assertEqual(
            build_parser().parse_args([]).config,
            "configs/presets/mem0_baseline.yaml",
        )

    def test_baseline_pins_released_mem0_reader_prompt_and_message_layout(self):
        cfg = load_config(ROOT / "configs" / "presets" / "mem0_baseline.yaml")
        self.assertEqual(cfg["reader"]["model"], "gpt-4o-mini")
        self.assertEqual(cfg["reader"]["temperature"], 0.0)
        self.assertIsNone(cfg["reader"]["max_tokens"])
        self.assertEqual(
            cfg["reader"]["message_layout"], READER_MESSAGE_LAYOUT_MEM0
        )
        self.assertEqual(cfg["pipeline"]["prompt_path"], "prompts/readers/qa_mem0_v1.txt")
        prompt = (ROOT / cfg["pipeline"]["prompt_path"]).read_text(encoding="utf-8")
        self.assertEqual(
            hashlib.sha256(prompt.replace("\r\n", "\n").encode("utf-8")).hexdigest(),
            "85c626a7eaf0631e17d3fcdaa020e47afb2d85802f1d4543d6363f812fbfe31c",
        )

    def test_mem0_message_layout_sends_filled_answer_prompt_as_only_system_message(self):
        messages = build_reader_messages("filled Mem0 prompt", READER_MESSAGE_LAYOUT_MEM0)
        self.assertEqual(
            messages,
            [{"role": "system", "content": "filled Mem0 prompt"}],
        )

    def test_autorater_config_pins_released_mem0_judge(self):
        cfg = load_config(ROOT / "configs" / "autoraters" / "mem0_gpt-4o-mini.yaml")
        self.assertEqual(cfg["autorater"]["model"], "gpt-4o-mini")
        self.assertEqual(cfg["autorater"]["temperature"], 0.0)
        self.assertIsNone(cfg["autorater"]["max_tokens"])
        prompt = (ROOT / cfg["autorater"]["prompt_path"]).read_text(encoding="utf-8")
        self.assertEqual(
            hashlib.sha256(prompt.replace("\r\n", "\n").encode("utf-8")).hexdigest(),
            "94aba7f70a8fed357d7f79ff14a9abacd5d8a67558e9f6d46d99915b04384cc0",
        )

    def test_cli_overrides_effective_reader_controls_without_mutating_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = _write_mini(root / "locomo.json")
            args = build_parser().parse_args(
                [
                    "--config",
                    str(ROOT / "configs" / "presets" / "mem0_baseline.yaml"),
                    "--data",
                    str(data_path),
                    "--reader",
                    "mock",
                    "--model",
                    "gpt-4.1-mini",
                    "--prompt",
                    str(ROOT / "prompts" / "readers" / "qa_v1.txt"),
                    "--temperature",
                    "0.25",
                    "--max-tokens",
                    "17",
                    "--message-layout",
                    "default_system_user",
                    "--max-questions",
                    "1",
                    "--run-id",
                    "override_lock",
                    "--output-dir",
                    str(root / "experiments"),
                ]
            )
            cfg = load_config(args.config)
            run_dir = run_locomo_pipeline_with_memory_config(cfg, args)
            meta = _load_json(run_dir / "run_meta.json")

            self.assertEqual(args.model, "gpt-4.1-mini")
            self.assertEqual(meta["prompt_version"], "qa_v1")
            self.assertEqual(meta["temperature"], 0.25)
            self.assertEqual(meta["max_tokens"], 17)
            self.assertEqual(meta["message_layout"], "default_system_user")

            unchanged = load_config(ROOT / "configs" / "presets" / "mem0_baseline.yaml")
            self.assertEqual(unchanged["reader"]["model"], "gpt-4o-mini")
            self.assertEqual(unchanged["pipeline"]["prompt_path"], "prompts/readers/qa_mem0_v1.txt")

    def test_compare_infers_mem0_prompt_from_run_metadata(self):
        packs = [
            {"meta": {"prompt_path": "prompts/readers/qa_mem0_v1.txt"}},
            {"meta": {"prompt_path": "prompts/readers/qa_mem0_v1.txt"}},
        ]
        self.assertEqual(
            resolve_prompt_path(packs),
            Path("prompts/readers/qa_mem0_v1.txt"),
        )

    def test_compare_rejects_runs_with_different_prompt_paths(self):
        packs = [
            {"meta": {"prompt_path": "prompts/readers/qa_mem0_v1.txt"}},
            {"meta": {"prompt_path": "prompts/readers/qa_v1.txt"}},
        ]
        with self.assertRaisesRegex(ValueError, "different prompt paths"):
            resolve_prompt_path(packs)

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
            "prompt_path": str(ROOT / "prompts" / "readers" / "qa_v1.txt"),
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
            run_dir = _run_one(Path(tmp), run_id="lock_audit", memory="session_summaries")
            for name in FROZEN_AUDIT_FILES:
                self.assertTrue((run_dir / name).is_file(), f"missing {name}")
            self.assertTrue((run_dir / "plots").is_dir())
            self.assertTrue((run_dir / "memory").is_dir())

    def test_run_locomo_pipeline_with_memory_config_returns_a_directory_named_only_by_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_one_id", memory="session_summaries")
            self.assertEqual(run_dir.name, "lock_one_id")
            self.assertEqual(len(list(run_dir.parent.iterdir())), 1)

    def test_run_locomo_pipeline_with_memory_config_records_exactly_one_memory_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_mem", memory="session_summaries")
            meta = _load_json(run_dir / "run_meta.json")
            metrics = _load_json(run_dir / "metrics.json")
            self.assertEqual(meta["memory_type"], "session_summaries")
            self.assertEqual(metrics["memory_type"], "session_summaries")
            rows = _load_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual({r["memory_type"] for r in rows}, {"session_summaries"})

    def test_prediction_jsonl_rows_include_frozen_audit_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_fields", memory="session_summaries")
            rows = _load_jsonl(run_dir / "predictions.jsonl")
            self.assertTrue(rows)
            for field in FROZEN_PREDICTION_FIELDS:
                self.assertIn(field, rows[0], f"prediction missing {field}")

    def test_metrics_json_reports_exact_match_token_f1_and_locomo_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_metrics", memory="session_summaries")
            overall = _load_json(run_dir / "metrics.json")["metrics"]
            for name in FROZEN_METRIC_NAMES:
                self.assertIn(name, overall)

    def test_memory_text_and_filled_prompt_do_not_contain_the_gold_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="lock_gold", memory="session_summaries")
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
            one = _run_one(root / "n1", run_id="n1", memory="session_summaries", max_questions=1)
            two = _run_one(root / "n2", run_id="n2", memory="session_summaries", max_questions=2)
            self.assertEqual(_load_json(one / "run_meta.json")["n_predictions"], 1)
            self.assertEqual(_load_json(two / "run_meta.json")["n_predictions"], 2)

    def test_second_call_with_same_run_id_regenerates_without_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _run_one(root, run_id="fresh_lock", memory="session_summaries", max_questions=1)
            n_first = _load_json(first / "run_meta.json")["n_new_api_calls"]
            stale = first / "autorater"
            stale.mkdir()
            (stale / "old.txt").write_text("stale", encoding="utf-8")
            second = _run_one(root, run_id="fresh_lock", memory="session_summaries", max_questions=1)
            meta = _load_json(second / "run_meta.json")
            self.assertGreaterEqual(n_first, 1)
            self.assertNotIn("n_resumed", meta)
            self.assertNotIn("regenerated_from_scratch", meta)
            self.assertGreaterEqual(meta["n_new_api_calls"], 1)
            self.assertFalse(stale.exists())
            self.assertEqual(second, first)

    def test_prediction_rows_and_run_meta_omit_cache_resume_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="no_cache", memory="session_summaries")
            meta = _load_json(run_dir / "run_meta.json")
            row = _load_jsonl(run_dir / "predictions.jsonl")[0]
            self.assertNotIn("cached", row)
            self.assertNotIn("llm_request_hash", row)
            self.assertNotIn("n_resumed", meta)
            self.assertNotIn("n_llm_response_hash_hits", meta)
            self.assertNotIn("llm_response_hash_dir", meta)
            self.assertNotIn("regenerated_from_scratch", meta)
            self.assertNotIn("llm_request_hash", meta)
            csv_header = (run_dir / "predictions.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("cached", csv_header.split(","))
            self.assertNotIn("llm_request_hash", csv_header.split(","))


class TestRunsAreSelfContained(unittest.TestCase):
    """Reusing a run id replaces artifacts; it does not append or skip."""

    def test_reusing_run_id_replaces_predictions_instead_of_appending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _run_one(root, run_id="replace_lock", memory="session_summaries", max_questions=2)
            self.assertEqual(_load_json(first / "run_meta.json")["n_predictions"], 2)
            second = _run_one(root, run_id="replace_lock", memory="session_summaries", max_questions=1)
            rows = _load_jsonl(second / "predictions.jsonl")
            meta = _load_json(second / "run_meta.json")
            self.assertEqual(len(rows), 1)
            self.assertEqual(meta["n_predictions"], 1)
            self.assertEqual(meta["n_new_api_calls"], 1)

    def test_preseeded_stale_prediction_rows_are_gone_after_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _run_one(root, run_id="stale_lock", memory="session_summaries", max_questions=1)
            pred_path = run_dir / "predictions.jsonl"
            stale = {
                "sample_id": "stale-sample",
                "question_id": "stale-qid",
                "question": "stale?",
                "reference_answer": GOLD_LOCK,
                "predicted_answer": "leftover",
                "category": 4,
                "memory_type": "session_summaries",
                "memory_text": "stale memory",
                "reader_model": "mock",
                "prompt_version": "qa_v1",
            }
            with pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(stale) + "\n")
            self.assertEqual(len(_load_jsonl(pred_path)), 2)
            _run_one(root, run_id="stale_lock", memory="session_summaries", max_questions=1)
            rows = _load_jsonl(pred_path)
            self.assertEqual(len(rows), 1)
            self.assertNotEqual(rows[0]["question_id"], "stale-qid")
            self.assertNotEqual(rows[0]["predicted_answer"], "leftover")

    def test_n_new_api_calls_equals_n_predictions_for_mock_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(
                Path(tmp), run_id="api_lock", memory="session_summaries", max_questions=2
            )
            meta = _load_json(run_dir / "run_meta.json")
            rows = _load_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(meta["n_new_api_calls"], 2)
            self.assertEqual(meta["n_new_api_calls"], meta["n_predictions"])


class TestSandwichMemorySwapDoesNotRetouchReaderOrMetrics(unittest.TestCase):
    def test_raw_chunks_and_session_summaries_runs_share_reader_model_and_prompt_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = _run_one(root / "raw", run_id="raw", memory="raw_chunks")
            summ = _run_one(root / "summaries", run_id="summaries", memory="session_summaries")
            m0 = _load_json(raw / "run_meta.json")
            m1 = _load_json(summ / "run_meta.json")
            self.assertEqual(m0["reader_model"], m1["reader_model"])
            self.assertEqual(m0["prompt_version"], m1["prompt_version"])
            self.assertEqual(m0["prompt_version"], "qa_v1")

    def test_raw_chunks_and_session_summaries_runs_produce_different_memory_text_for_the_same_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = _run_one(root / "raw", run_id="raw", memory="raw_chunks")
            summ = _run_one(root / "summaries", run_id="summaries", memory="session_summaries")
            t0 = _load_jsonl(raw / "predictions.jsonl")[0]["memory_text"]
            t1 = _load_jsonl(summ / "predictions.jsonl")[0]["memory_text"]
            self.assertNotEqual(t0, t1)
            self.assertIn("I started painting", t0)
            self.assertIn("Alice said she started painting", t1)

    def test_raw_chunks_and_session_summaries_metrics_json_use_the_same_metric_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = _run_one(root / "raw", run_id="raw", memory="raw_chunks")
            summ = _run_one(root / "summaries", run_id="summaries", memory="session_summaries")
            k0 = set(_load_json(raw / "metrics.json")["metrics"])
            k1 = set(_load_json(summ / "metrics.json")["metrics"])
            self.assertEqual(k0, k1)
            self.assertTrue(set(FROZEN_METRIC_NAMES) <= k0)


class TestCompareAndOfflineEvaluateAfterTwoPipelineRuns(unittest.TestCase):
    def test_compare_full_runs_pair_analysis_marks_raw_chunks_and_session_summaries_as_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = _run_one(root / "raw", run_id="raw", memory="raw_chunks")
            summ = _run_one(root / "summaries", run_id="summaries", memory="session_summaries")
            packs = [load_pack(raw), load_pack(summ)]
            paired = pair_analysis(packs, ROOT / "prompts" / "readers" / "qa_v1.txt")
            self.assertEqual(paired["n_common"], 1)
            self.assertEqual(paired["fraction_same_memory_text"], 0.0)
            self.assertNotIn("fraction_same_llm_request_hash", paired)
            self.assertNotIn("cross_condition_request_hash_collision_ok", paired.get("sanity") or {})
            self.assertNotEqual(packs[0]["meta"]["memory_type"], packs[1]["meta"]["memory_type"])

    def test_offline_evaluate_rewrites_metrics_but_not_predicted_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _run_one(root, run_id="rescore", memory="session_summaries")
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


class TestClaimAuditTracingDoesNotLeakAcrossSamplesOrOptionalFilters(unittest.TestCase):
    def test_shared_edge_id_does_not_copy_proposed_by_from_another_sample(self):
        rows = lineage_rows(
            prediction_rows=[
                {"question_id": "q0", "sample_id": "s1"},
                {"question_id": "q1", "sample_id": "s2"},
            ],
            memories_by_sample={
                "s1": Memory(memory_type="fused_teacher_graph", text="p", source_ids=["e0000"]),
                "s2": Memory(memory_type="fused_teacher_graph", text="z", source_ids=["e0000"]),
            },
            ingest_rows=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "ops": [
                        {
                            "op": "add_edge",
                            "edge_id": "e0000",
                            "source": "alice",
                            "relationship": "started",
                            "target": "painting",
                        }
                    ],
                },
                {
                    "sample_id": "s2",
                    "session_id": 1,
                    "ops": [
                        {
                            "op": "add_edge",
                            "edge_id": "e0000",
                            "source": "bob",
                            "relationship": "likes",
                            "target": "pizza",
                        }
                    ],
                },
            ],
            fusion_rows=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "relations": [
                        {
                            "source": "alice",
                            "relationship": "started",
                            "target": "painting",
                            "proposed_by": ["openai"],
                            "kept": True,
                        }
                    ],
                },
                {
                    "sample_id": "s2",
                    "session_id": 1,
                    "relations": [
                        {
                            "source": "bob",
                            "relationship": "likes",
                            "target": "pizza",
                            "proposed_by": ["anthropic"],
                            "kept": True,
                        }
                    ],
                },
            ],
        )
        by_q = {row["question_id"]: row for row in rows}
        self.assertEqual(by_q["q0"]["proposed_by"], ["openai"])
        self.assertEqual(by_q["q1"]["proposed_by"], ["anthropic"])

    def test_retrieve_losers_never_appear_in_lineage_rows(self):
        rows = lineage_rows(
            prediction_rows=[{"question_id": "q0", "sample_id": "s1"}],
            retrieve_ranks=[
                {
                    "sample_id": "s1",
                    "question_id": "q0",
                    "candidates": [
                        {"item_id": "win", "selected": True},
                        {"item_id": "lose", "selected": False},
                    ],
                }
            ],
        )
        self.assertEqual([row["item_id"] for row in rows], ["win"])

    def test_omitted_sample_id_filter_does_not_drop_rows_and_set_filter_does_not_leak(self):
        audit = SandwichAudit(
            paths=AuditPaths.from_run_dir("unused"),
            meta={},
            metrics={},
            predictions=[],
            lineage=[
                {"question_id": "q0", "sample_id": "s1", "item_id": "a"},
                {"question_id": "q0", "sample_id": "s2", "item_id": "b"},
            ],
        )
        self.assertEqual({row["item_id"] for row in audit.lineage_for()}, {"a", "b"})
        self.assertEqual([row["item_id"] for row in audit.lineage_for(sample_id="s1")], ["a"])

    def test_reset_run_output_deletes_stale_attribution_lineage_and_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "ATTRIBUTION.md").write_text("stale teacher leak", encoding="utf-8")
            (run / "attribution.jsonl").write_text("{}\n", encoding="utf-8")
            (run / "cost.json").write_text("{}", encoding="utf-8")
            mem = run / "memory"
            mem.mkdir()
            (mem / "lineage.jsonl").write_text("{}\n", encoding="utf-8")
            _reset_run_output(run)
            self.assertFalse((run / "ATTRIBUTION.md").exists())
            self.assertFalse((run / "attribution.jsonl").exists())
            self.assertFalse((run / "cost.json").exists())
            self.assertFalse(mem.exists())

    def test_predictions_search_skips_empty_decoy_dir_instead_of_hiding_the_real_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reg_iso_run").mkdir()
            real = root / "experiments" / "reg_iso_run"
            real.mkdir(parents=True)
            (real / "predictions.jsonl").write_text(
                json.dumps({"question_id": "wanted"}) + "\n", encoding="utf-8"
            )
            found = resolve_predictions_jsonl("reg_iso_run", repo_root=root)
            self.assertEqual(found, real / "predictions.jsonl")


if __name__ == "__main__":
    unittest.main()
