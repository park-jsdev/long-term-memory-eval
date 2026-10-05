"""Year-family robustness insight tables (offline, no API)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.campaign_insights import (
    attach_cell_cost,
    category_holes,
    classify_delta,
    classify_gap_change,
    efficiency,
    family_gaps,
    j_f1_gap,
    takeaway_contrast,
    method_ranks,
    pin_gaps,
    rank_flips,
    render_insight,
    saturation,
    series_gaps,
    thinking_deltas,
    year_deltas,
)
from src.experiment_runner.analysis.load_design import load_design_yaml
from src.experiment_runner.analysis.report import render_design


class TestClassifyDelta(unittest.TestCase):
    def test_classify_delta_returns_improving_when_gain_exceeds_eps(self):
        self.assertEqual(classify_delta(0.05), "improving")

    def test_classify_delta_returns_declining_when_drop_exceeds_eps(self):
        self.assertEqual(classify_delta(-0.05), "declining")

    def test_classify_delta_returns_not_moving_inside_eps(self):
        self.assertEqual(classify_delta(0.01), "not_moving")

    def test_classify_delta_returns_missing_for_none(self):
        self.assertEqual(classify_delta(None), "missing")


class TestClassifyGapChange(unittest.TestCase):
    def test_classify_gap_change_returns_closing_when_abs_gap_shrinks(self):
        self.assertEqual(classify_gap_change(0.20, 0.05), "gap_closing")

    def test_classify_gap_change_returns_widening_when_abs_gap_grows(self):
        self.assertEqual(classify_gap_change(0.05, 0.20), "gap_widening")

    def test_classify_gap_change_returns_stable_when_abs_gap_holds(self):
        self.assertEqual(classify_gap_change(0.10, 0.11), "gap_stable")


class TestYearDeltas(unittest.TestCase):
    def test_year_deltas_pairs_same_result_source_and_skips_pin_vs_live(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2024",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "result_source": "paper",
                    "thinking": None,
                    "locomo_f1": 0.50,
                    "judge_score": 0.73,
                },
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "result_source": "live",
                    "thinking": "off",
                    "locomo_f1": 0.60,
                    "judge_score": 0.80,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "result_source": "live",
                    "thinking": "off",
                    "locomo_f1": 0.70,
                    "judge_score": 0.82,
                },
            ]
        )
        out = year_deltas(table, ["locomo_f1"])
        live = out[
            (out["result_source"] == "live") & (out["metric"] == "locomo_f1")
        ]
        self.assertEqual(len(live), 1)
        self.assertEqual(live.iloc[0]["from_generation"], "2025")
        self.assertEqual(live.iloc[0]["to_generation"], "2026")
        self.assertAlmostEqual(float(live.iloc[0]["delta"]), 0.10)
        self.assertEqual(live.iloc[0]["move"], "improving")
        paper = out[out["result_source"] == "paper"]
        self.assertTrue((paper["move"] == "missing").all())

    def test_year_deltas_treats_lower_latency_as_improving(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "agent_latency_seconds": 10.0,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "agent_latency_seconds": 6.0,
                },
            ]
        )
        out = year_deltas(table, ["agent_latency_seconds"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["move"], "improving")
        self.assertAlmostEqual(float(out.iloc[0]["delta"]), -4.0)


class TestFamilyGaps(unittest.TestCase):
    def test_family_gaps_marks_closing_when_openai_lead_shrinks(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "thinking": "off",
                    "locomo_f1": 0.80,
                },
                {
                    "generation": "2025",
                    "model_family": "DeepSeek",
                    "memory_method": "full_context",
                    "thinking": "off",
                    "locomo_f1": 0.50,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "thinking": "off",
                    "locomo_f1": 0.82,
                },
                {
                    "generation": "2026",
                    "model_family": "DeepSeek",
                    "memory_method": "full_context",
                    "thinking": "off",
                    "locomo_f1": 0.78,
                },
            ]
        )
        out = family_gaps(table, ["locomo_f1"])
        later = out[out["generation"] == "2026"].iloc[0]
        self.assertAlmostEqual(float(later["gap"]), 0.04, places=6)
        self.assertEqual(later["gap_move"], "gap_closing")


class TestMethodRanks(unittest.TestCase):
    def test_method_ranks_orders_full_context_ahead_of_rag(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "thinking": "off",
                    "memory_method": "rag",
                    "locomo_f1": 0.40,
                },
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "thinking": "off",
                    "memory_method": "full_context",
                    "locomo_f1": 0.70,
                },
            ]
        )
        out = method_ranks(table, "locomo_f1")
        fc = out[out["memory_method"] == "full_context"].iloc[0]
        rag = out[out["memory_method"] == "rag"].iloc[0]
        self.assertEqual(int(fc["rank"]), 1)
        self.assertEqual(int(rag["rank"]), 2)

    def test_rank_flips_reports_when_family_method_order_disagrees(self):
        ranks = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "thinking": "off",
                    "memory_method": "full_context",
                    "locomo_f1": 0.70,
                    "rank": 1,
                },
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "thinking": "off",
                    "memory_method": "rag",
                    "locomo_f1": 0.40,
                    "rank": 2,
                },
                {
                    "generation": "2025",
                    "model_family": "DeepSeek",
                    "thinking": "off",
                    "memory_method": "rag",
                    "locomo_f1": 0.60,
                    "rank": 1,
                },
                {
                    "generation": "2025",
                    "model_family": "DeepSeek",
                    "thinking": "off",
                    "memory_method": "full_context",
                    "locomo_f1": 0.50,
                    "rank": 2,
                },
            ]
        )
        flips = rank_flips(ranks)
        self.assertGreaterEqual(len(flips), 1)
        self.assertEqual(flips.iloc[0]["label"], "rank_flip")
        self.assertEqual(flips.iloc[0]["split"], "model_family")


class TestThinkingDeltas(unittest.TestCase):
    def test_thinking_deltas_labels_helps_when_on_beats_off(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "thinking": "off",
                    "locomo_f1": 0.50,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "thinking": "on",
                    "locomo_f1": 0.60,
                },
            ]
        )
        out = thinking_deltas(table, ["locomo_f1"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["label"], "thinking_helps")
        self.assertAlmostEqual(float(out.iloc[0]["delta"]), 0.10)


class TestCategoryHoles(unittest.TestCase):
    def test_category_holes_marks_persistent_when_every_year_is_below_max(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "question_category": "1 multi-hop",
                    "locomo_f1": 0.20,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "question_category": "1 multi-hop",
                    "locomo_f1": 0.22,
                },
            ]
        )
        out = category_holes(table, "locomo_f1", hole_max=0.40)
        self.assertEqual(out.iloc[0]["label"], "persistent_hole")
        self.assertEqual(out.iloc[0]["move"], "not_moving")

    def test_category_holes_marks_closing_when_last_year_leaves_the_band(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "question_category": "2 temporal",
                    "locomo_f1": 0.20,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "question_category": "2 temporal",
                    "locomo_f1": 0.55,
                },
            ]
        )
        out = category_holes(table, "locomo_f1", hole_max=0.40)
        self.assertEqual(out.iloc[0]["label"], "closing_hole")
        self.assertEqual(out.iloc[0]["move"], "improving")


class TestPinGaps(unittest.TestCase):
    def test_pin_gaps_compares_2024_paper_to_2025_openai_live_thinking_off(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2024",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "result_source": "paper",
                    "thinking": None,
                    "judge_score": 0.73,
                },
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "result_source": "live",
                    "thinking": "on",
                    "judge_score": 0.99,
                },
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "result_source": "live",
                    "thinking": "off",
                    "judge_score": 0.80,
                },
            ]
        )
        out = pin_gaps(table, "judge_score")
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["label"], "pin_vs_live_not_matched_stack")
        self.assertAlmostEqual(float(out.iloc[0]["live_value"]), 0.80)
        self.assertEqual(out.iloc[0]["move"], "improving")

    def test_pin_gaps_uses_compare_source_codex_as_live_side(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2024",
                    "model_family": "OpenAI",
                    "paper_method": "full_context",
                    "compare_source": "paper",
                    "judge_score": 0.729,
                },
                {
                    "generation": "2024",
                    "model_family": "OpenAI",
                    "paper_method": "full_context",
                    "compare_source": "gpt-4o-mini + Codex",
                    "judge_score": 0.678,
                },
            ]
        )
        out = pin_gaps(table, "judge_score", live_generation="2024")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["live_value"]), 0.678)
        self.assertAlmostEqual(float(out.iloc[0]["pin_value"]), 0.729)

    def test_pin_gaps_compares_2024_paper_to_2026_openai_when_live_generation_set(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2024",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "result_source": "paper",
                    "judge_score": 0.73,
                },
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "result_source": "live",
                    "thinking": "off",
                    "judge_score": 0.80,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "result_source": "live",
                    "thinking": "off",
                    "judge_score": 0.90,
                },
            ]
        )
        out = pin_gaps(table, "judge_score", live_generation="2026")
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["live_generation"], "2026")
        self.assertAlmostEqual(float(out.iloc[0]["live_value"]), 0.90)


class TestEfficiencyAndSaturation(unittest.TestCase):
    def test_efficiency_divides_score_by_latency_and_usd(self):
        table = pd.DataFrame(
            [
                {
                    "locomo_f1": 0.50,
                    "judge_score": 0.80,
                    "agent_latency_seconds": 2.0,
                    "usd_actual": 10.0,
                    "agent_reasoning_tokens": 2000.0,
                }
            ]
        )
        out = efficiency(table)
        self.assertAlmostEqual(float(out.iloc[0]["locomo_f1_per_second"]), 0.25)
        self.assertAlmostEqual(float(out.iloc[0]["judge_score_per_usd"]), 0.08)
        self.assertAlmostEqual(float(out.iloc[0]["locomo_f1_per_k_reasoning"]), 0.25)

    def test_saturation_labels_diminishing_returns_when_score_flat_and_latency_up(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "judge_score": 0.70,
                    "agent_latency_seconds": 1.0,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "judge_score": 0.71,
                    "agent_latency_seconds": 8.0,
                },
            ]
        )
        out = saturation(table, "judge_score")
        later = out[out["generation"] == "2026"].iloc[0]
        self.assertEqual(later["band"], "open_headroom")
        self.assertEqual(later["label"], "diminishing_returns")

    def test_saturation_labels_near_ceiling_as_saturated_flat_when_j_holds(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "judge_score": 0.91,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "judge_score": 0.92,
                },
            ]
        )
        out = saturation(table, "judge_score")
        later = out[out["generation"] == "2026"].iloc[0]
        self.assertEqual(later["band"], "near_ceiling")
        self.assertEqual(later["label"], "saturated_flat")

    def test_attach_cell_cost_joins_usd_by_year_family_method(self):
        quality = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "memory_method": "full_context",
                    "thinking": "off",
                    "locomo_f1": 0.50,
                }
            ]
        )
        cost = pd.DataFrame(
            [
                {
                    "experiment": "readers",
                    "memory_method": "full_context",
                    "cell_model": "gpt-5",
                    "thinking": "off",
                    "role": "reader",
                    "usd_actual": 8.0,
                },
                {
                    "experiment": "readers",
                    "memory_method": "full_context",
                    "cell_model": "gpt-5",
                    "thinking": "off",
                    "role": "judge",
                    "usd_actual": 2.0,
                },
            ]
        )
        out = attach_cell_cost(quality, cost)
        self.assertAlmostEqual(float(out.iloc[0]["usd_actual"]), 10.0)


class TestJF1Gap(unittest.TestCase):
    def test_j_f1_gap_subtracts_locomo_f1_from_judge_score(self):
        table = pd.DataFrame(
            [
                {
                    "memory_method": "workspace_files",
                    "judge_score": 0.68,
                    "locomo_f1": 0.25,
                },
                {
                    "memory_method": "full_context",
                    "judge_score": 0.74,
                    "locomo_f1": 0.54,
                },
            ]
        )
        out = j_f1_gap(table)
        self.assertAlmostEqual(float(out.iloc[0]["j_minus_f1"]), 0.43)
        self.assertAlmostEqual(float(out.iloc[1]["j_minus_f1"]), 0.20)
        via = render_insight("j_f1_gap", table)
        self.assertEqual(list(via["j_minus_f1"]), list(out["j_minus_f1"]))


class TestSeriesGaps(unittest.TestCase):
    def test_series_gaps_sorts_by_absolute_delta_and_keeps_sign(self):
        table = pd.DataFrame(
            {
                "question_category": ["1 multi-hop", "1 multi-hop", "2 temporal", "2 temporal"],
                "mini_side": ["4o-mini", "4o-mini + Codex", "4o-mini", "4o-mini + Codex"],
                "judge_score": [0.80, 0.50, 0.70, 0.68],
            }
        )
        gaps = series_gaps(
            table,
            metric="judge_score",
            series_col="mini_side",
            left="4o-mini",
            right="4o-mini + Codex",
        )
        self.assertEqual(list(gaps["question_category"]), ["1 multi-hop", "2 temporal"])
        self.assertAlmostEqual(float(gaps.iloc[0]["delta"]), 0.30)
        self.assertAlmostEqual(float(gaps.iloc[1]["delta"]), 0.02)


class TestTakeawayContrast(unittest.TestCase):
    def test_takeaway_contrast_is_left_minus_right(self):
        left = pd.DataFrame([{"memory_method": "workspace_files", "judge_score": 0.68, "n": 1540}])
        right = pd.DataFrame([{"memory_method": "full_context", "judge_score": 0.74, "n": 1540}])
        out = takeaway_contrast(
            left,
            right,
            ["judge_score"],
            takeaway_id="harness_reader_vs_chat_reader",
            title="Harness vs chat reader",
            claim="harness_reader_weaker_than_chat_reader",
            finding="Harness trails stuffed-context.",
            left_label="workspace_codex",
            right_label="chat_full_context",
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["delta"]), -0.06)
        self.assertIn("trails stuffed-context", str(out.iloc[0]["finding"]))


class TestRenderInsightDispatch(unittest.TestCase):
    def test_render_insight_year_deltas_matches_direct_call(self):
        table = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "locomo_f1": 0.50,
                },
                {
                    "generation": "2026",
                    "model_family": "OpenAI",
                    "locomo_f1": 0.60,
                },
            ]
        )
        a = render_insight("year_deltas", table, metrics=["locomo_f1"])
        b = year_deltas(table, ["locomo_f1"])
        self.assertEqual(list(a["move"]), list(b["move"]))


