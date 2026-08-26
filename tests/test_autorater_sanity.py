"""Autorater sanity: Mem0 judge protocol, GPT-4o seam, and fresh reports.

Offline only. ``MockAutorater`` replaces API calls while preserving the same
Prediction → AutoraterVerdict → benchmark-report data path.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.run_benchmark import aggregate_benchmark, run_benchmark
from src.locomo_eval.autorater import (
    MockAutorater,
    get_autorater,
    parse_judge_label,
    should_skip_category,
)
from src.locomo_eval.mem0_baselines import (
    PAPER_URL,
    TABLE1_BY_CATEGORY,
    literature_overall_j,
)
from src.locomo_eval.mem0_metrics import mem0_bleu1, mem0_f1, summarize_latencies
from src.locomo_eval.models import DEFAULT_AUTORATER_MODEL, GPT4O_MINI, resolve_model
from src.locomo_eval.prompts import render_autorater_prompt


def _prediction(
    question_id: str,
    *,
    predicted_answer: str,
    reference_answer: str = "painting",
    category: int = 4,
    latency_s: float = 0.5,
) -> dict:
    return {
        "sample_id": "conv-test",
        "question_id": question_id,
        "question": "What hobby did Alice begin?",
        "reference_answer": reference_answer,
        "predicted_answer": predicted_answer,
        "category": category,
        "memory_type": "session_summaries",
        "memory_text": "Alice began painting.",
        "reader_model": "gpt-4.1-mini",
        "prompt_version": "qa_v1",
        "latency_s": latency_s,
    }


class TestAutoraterLabelParsing(unittest.TestCase):
    def test_parse_judge_label_returns_correct_from_mem0_json(self):
        label, reasoning = parse_judge_label('{"label": "CORRECT"}')
        self.assertEqual(label, "CORRECT")
        self.assertIsNone(reasoning)

    def test_parse_judge_label_returns_wrong_and_reasoning_from_json(self):
        label, reasoning = parse_judge_label(
            '{"label": "WRONG", "reasoning": "The month differs."}'
        )
        self.assertEqual(label, "WRONG")
        self.assertEqual(reasoning, "The month differs.")

    def test_parse_judge_label_rejects_response_containing_both_labels(self):
        with self.assertRaises(ValueError):
            parse_judge_label("The answer may be CORRECT or WRONG.")

    def test_parse_judge_label_accepts_json_inside_markdown_fence(self):
        label, _ = parse_judge_label('```json\n{"label":"CORRECT"}\n```')
        self.assertEqual(label, "CORRECT")

    def test_parse_judge_label_treats_free_text_incorrect_as_wrong(self):
        label, _ = parse_judge_label("The generated answer is INCORRECT.")
        self.assertEqual(label, "WRONG")


class TestAutoraterPromptAndModel(unittest.TestCase):
    def test_default_autorater_model_matches_released_mem0_gpt4o_mini(self):
        self.assertEqual(DEFAULT_AUTORATER_MODEL, GPT4O_MINI)
        self.assertEqual(resolve_model(GPT4O_MINI).family, "gpt-4o")

    def test_get_autorater_rejects_response_cache(self):
        with self.assertRaisesRegex(ValueError, "caching is disabled"):
            get_autorater("mock", llm_response_hash=object())

    def test_render_autorater_prompt_includes_question_gold_and_prediction(self):
        rendered = render_autorater_prompt(
            "Q={question}\nG={gold_answer}\nP={generated_answer}",
            question="When?",
            gold_answer="7 May",
            generated_answer="May 7th",
        )
        self.assertEqual(rendered, "Q=When?\nG=7 May\nP=May 7th")

    def test_get_autorater_mock_does_not_impersonate_requested_gpt4o_model(self):
        rater = get_autorater("mock", model="gpt-4o")
        self.assertIsInstance(rater, MockAutorater)
        self.assertEqual(rater.model_name, "mock")
        self.assertEqual(rater.provider, "mock")

    def test_should_skip_category_returns_true_only_for_mem0_category_five(self):
        self.assertTrue(should_skip_category(5))
        self.assertFalse(should_skip_category(4))


class TestMem0Metrics(unittest.TestCase):
    def test_mem0_f1_returns_one_for_same_unique_tokens(self):
        self.assertEqual(mem0_f1("Alice likes painting", "Alice likes painting"), 1.0)

    def test_mem0_f1_returns_zero_for_disjoint_tokens(self):
        self.assertEqual(mem0_f1("painting", "running"), 0.0)

    def test_mem0_bleu1_applies_brevity_penalty_to_short_prediction(self):
        score = mem0_bleu1("shell", "a shell necklace")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_summarize_latencies_reports_p50_and_interpolated_p95(self):
        summary = summarize_latencies([1.0, 2.0, 3.0])
        self.assertEqual(summary["p50"], 2.0)
        self.assertEqual(summary["p95"], 2.9)


class TestAutoraterBenchmark(unittest.TestCase):
    def test_mock_autorater_returns_correct_for_shared_topic(self):
        verdict = MockAutorater(GPT4O_MINI).rate(
            "What hobby?", "painting", "Alice started painting last month."
        )
        self.assertEqual(verdict.label, "CORRECT")
        self.assertEqual(verdict.llm_score, 1)

    def test_aggregate_benchmark_excludes_skipped_category_from_judge_score(self):
        rows = [
            {
                "category_name": "single_hop",
                "category": 4,
                "mem0_f1": 1.0,
                "mem0_bleu1": 1.0,
                "llm_score": 1,
                "locomo_f1": 1.0,
                "reader_latency_s": 0.5,
                "autorater_latency_s": 0.2,
                "skipped": False,
            },
            {
                "category_name": "adversarial",
                "category": 5,
                "mem0_f1": 0.0,
                "mem0_bleu1": 0.0,
                "llm_score": 0,
                "locomo_f1": 1.0,
                "reader_latency_s": 0.6,
                "autorater_latency_s": 0.0,
                "skipped": True,
            },
        ]
        summary = aggregate_benchmark(rows)
        self.assertEqual(summary["n_judged"], 1)
        self.assertEqual(summary["n_skipped"], 1)
        self.assertEqual(summary["metrics"]["llm_judge"], 1.0)
        self.assertNotIn("adversarial", summary["by_category"])

    def test_run_benchmark_writes_tables_plots_metrics_and_summary(self):
        predictions = [
            _prediction("conv-test-q-0", predicted_answer="painting"),
            _prediction(
                "conv-test-q-1",
                predicted_answer="Unknown.",
                reference_answer="No information available",
                category=5,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_path = root / "predictions.jsonl"
            pred_path.write_text(
                "".join(json.dumps(row) + "\n" for row in predictions),
                encoding="utf-8",
            )
            out = root / "autorater"
            result = run_benchmark(
                pred_path,
                out_dir=out,
                autorater=MockAutorater(GPT4O_MINI),
                our_label="session_summaries",
                prompt_version="autorater_mem0_v1",
            )
            required = (
                "autorater_verdicts.jsonl",
                "autorater_metrics.json",
                "run_meta.json",
                "SUMMARY.md",
                "tables/overall.csv",
                "tables/by_category.csv",
                "tables/vs_literature.csv",
                "tables/vs_literature_by_category.csv",
                "tables/per_question.csv",
            )
            for relative in required:
                self.assertTrue((out / relative).is_file(), relative)
            self.assertEqual(result["summary"]["n_judged"], 1)
            self.assertEqual(result["summary"]["n_skipped"], 1)
            self.assertEqual(result["summary"]["autorater_provider"], "mock")
            self.assertEqual(
                result["summary"]["score_kind"], "mock_sanity_not_llm_judge"
            )
            self.assertIn(PAPER_URL, (out / "SUMMARY.md").read_text(encoding="utf-8"))
            self.assertTrue((out / "plots" / "judge_verdict_barplot.png").is_file())
            self.assertGreaterEqual(len(result["plots"]), 5)
            for plot in result["plots"]:
                self.assertTrue(Path(plot).is_file())

    def test_run_benchmark_regenerates_without_appending_prior_analysis(self):
        pred = _prediction("conv-test-q-0", predicted_answer="painting")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_path = root / "predictions.jsonl"
            pred_path.write_text(json.dumps(pred) + "\n", encoding="utf-8")
            out = root / "autorater"
            run_benchmark(
                pred_path,
                out_dir=out,
                autorater=MockAutorater(GPT4O_MINI),
                our_label="test",
            )
            stale_plot = out / "plots" / "stale_from_prior_run.png"
            stale_plot.write_text("stale", encoding="utf-8")
            pred_path.write_text(
                json.dumps(_prediction("conv-test-q-0", predicted_answer="Unknown."))
                + "\n",
                encoding="utf-8",
            )
            second = run_benchmark(
                pred_path,
                out_dir=out,
                autorater=MockAutorater(GPT4O_MINI),
                our_label="test",
            )
            verdicts = [
                json.loads(line)
                for line in (out / "autorater_verdicts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertTrue(second["regenerated_from_scratch"])
            self.assertEqual(second["n_new_ratings"], 1)
            self.assertEqual(second["n_new_judge_calls"], 0)
            self.assertEqual(len(verdicts), 1)
            self.assertEqual(verdicts[0]["label"], "WRONG")
            self.assertFalse(stale_plot.exists())

    def test_run_benchmark_rejects_cached_source_predictions(self):
        pred = _prediction("conv-test-q-0", predicted_answer="painting")
        pred["cached"] = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_path = root / "predictions.jsonl"
            pred_path.write_text(json.dumps(pred) + "\n", encoding="utf-8")
            out = root / "autorater"
            out.mkdir()
            stale = out / "autorater_metrics.json"
            stale.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cached prediction rows"):
                run_benchmark(
                    pred_path,
                    out_dir=out,
                    autorater=MockAutorater(),
                    our_label="test",
                )
            self.assertFalse(stale.exists())


class TestMem0LiteraturePins(unittest.TestCase):
    def test_mem0_overall_j_matches_paper_table_two(self):
        self.assertEqual(literature_overall_j("Mem0"), 66.88)

    def test_mem0_single_hop_j_matches_paper_table_one(self):
        self.assertEqual(TABLE1_BY_CATEGORY["Mem0"]["single_hop"]["j"], 67.13)


if __name__ == "__main__":
    unittest.main()
