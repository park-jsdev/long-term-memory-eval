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
    method_ranks,
    pin_gaps,
    rank_flips,
    render_insight,
    saturation,
    thinking_deltas,
    year_deltas,
)
from src.memorybench.analysis.load_campaign import load_campaign_yaml
from src.memorybench.analysis.report import render_campaign


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


class TestYearFamilyYamlInsights(unittest.TestCase):
    YAML = ROOT / "configs" / "analysis" / "campaign_year_family.yaml"

    def test_year_family_yaml_lists_robustness_analyses_and_insights(self):
        cfg = load_campaign_yaml(self.YAML)
        ids = [spec.id for spec in cfg.campaign_analyses]
        self.assertIn("reader_live_year_family", ids)
        self.assertIn("writer_live_year_family", ids)
        self.assertIn("reader_fc_category_year", ids)
        self.assertIn("reader_latency_year", ids)
        self.assertIn("reader_efficiency_year", ids)
        insight_ids = [spec.id for spec in cfg.insights]
        self.assertIn("reader_year_deltas", insight_ids)
        self.assertIn("reader_rank_flips", insight_ids)
        self.assertIn("writer_rank_flips", insight_ids)
        self.assertIn("pin_vs_live_2025", insight_ids)
        self.assertIn("reader_efficiency", insight_ids)
        self.assertIn("reader_saturation", insight_ids)
        self.assertIn("reader_roi_year_deltas", insight_ids)
        self.assertTrue(cfg.cost is not None)
        known = set(ids) | set(insight_ids)
        for spec in cfg.insights:
            self.assertIn(spec.source_analysis, known)
        self.assertEqual(
            cfg.notebook, "notebooks/15_year_family_robustness_analysis.ipynb"
        )
        self.assertEqual(cfg.experiments["readers"].pretest.n_cells, 8)
        self.assertEqual(cfg.experiments["readers_2026"].pretest.n_cells, 8)

    def test_year_family_notebook_is_a_thin_yaml_wrapper(self):
        notebook = ROOT / "notebooks" / "15_year_family_robustness_analysis.ipynb"
        text = notebook.read_text(encoding="utf-8")
        self.assertIn("campaign_year_family.yaml", text)
        self.assertIn("notebook_pretest", text)
        self.assertIn("notebook_posttest", text)
        self.assertIn("render_campaign(camp", text)
        self.assertNotIn("matplotlib", text)
        self.assertNotIn("groupby(", text)
        self.assertIn("hue=thinking", text.lower())
        self.assertIn("rank_flip", text)
        self.assertIn("diminishing_returns", text)
        self.assertIn("judge_score_per_usd", text)

    def test_render_campaign_writes_insight_csv_when_live_packs_exist(self):
        cfg = load_campaign_yaml(self.YAML)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for year, exp, model, provider in (
                ("2025", "locomo-2025-readers-openai-deepseek", "gpt-5", "openai"),
                ("2026", "locomo-2026-readers-openai-deepseek", "gpt-5.6-terra", "openai"),
            ):
                rows = []
                for method in ("full_context", "rag"):
                    for think in ("off", "on"):
                        fc = method == "full_context"
                        score = (0.50 if year == "2025" else 0.70) if fc else (
                            0.30 if year == "2025" else 0.35
                        )
                        for cat, used in ((1, score), (5, 0.0)):
                            rows.append(
                                {
                                    "reader_model": model,
                                    "reader_provider": provider,
                                    "reader_generation": year,
                                    "memory_method": method,
                                    "thinking": think,
                                    "question_category": cat,
                                    "locomo_f1": used,
                                    "judge_score": used,
                                    "num_examples": 1986,
                                }
                            )
                pack = root / "experiments" / exp / "aggregate"
                pack.mkdir(parents=True)
                frame = pd.DataFrame(rows)
                frame.to_parquet(pack / "examples.parquet", index=False)
                frame.head(2).assign(run_id=["a", "b"]).to_parquet(
                    pack / "runs.parquet", index=False
                )
            report = render_campaign(cfg, root=root)
            delta_path = report.out_dir / "tables" / "reader_year_deltas.csv"
            self.assertTrue(delta_path.is_file())
            deltas = pd.read_csv(delta_path)
            self.assertIn("improving", set(deltas["move"].astype(str)))
            self.assertIn("reader_live_year_family", [item.spec.id for item in report.results])


if __name__ == "__main__":
    unittest.main()
