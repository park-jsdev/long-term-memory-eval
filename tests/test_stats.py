"""Stats helpers: mean/CI, Wilcoxon, McNemar, seed aggregation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.stats import (
    bootstrap_ci95,
    mcnemar_test,
    mean_ci95,
    mean_std,
    summarize_seed_metrics,
    t_critical_975,
    wilcoxon_signed_rank,
)


class TestMeanStdAndCi(unittest.TestCase):
    def test_mean_std_returns_none_for_empty_input(self):
        mean, std = mean_std([])
        self.assertIsNone(mean)
        self.assertIsNone(std)

    def test_mean_ci95_for_ten_identical_values_has_zero_width(self):
        stats = mean_ci95([66.88] * 10)
        self.assertEqual(stats["n"], 10)
        self.assertEqual(stats["mean"], 66.88)
        self.assertEqual(stats["std"], 0.0)
        self.assertEqual(stats["ci95_low"], 66.88)
        self.assertEqual(stats["ci95_high"], 66.88)

    def test_t_critical_975_uses_paper_n10_df(self):
        self.assertAlmostEqual(t_critical_975(9), 2.262, places=3)

    def test_bootstrap_ci95_is_seeded_and_contains_the_mean(self):
        values = [50.0, 52.0, 51.0, 49.5, 50.5]
        a = bootstrap_ci95(values, seed=0)
        b = bootstrap_ci95(values, seed=0)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci95_low"], a["mean"])
        self.assertGreaterEqual(a["ci95_high"], a["mean"])


class TestWilcoxonAndMcnemar(unittest.TestCase):
    def test_wilcoxon_signed_rank_is_significant_when_all_deltas_are_positive(self):
        result = wilcoxon_signed_rank([0.2, 0.1, 0.3, 0.15, 0.25, 0.18])
        self.assertEqual(result["n"], 6)
        self.assertLess(result["p_two_sided"], 0.05)

    def test_wilcoxon_signed_rank_drops_zero_deltas(self):
        result = wilcoxon_signed_rank([0.0, 0.0, 0.1, -0.1])
        self.assertEqual(result["n"], 2)

    def test_mcnemar_test_counts_discordant_pairs_and_is_not_significant_when_balanced(self):
        a = [True, True, False, False, True, False]
        b = [True, False, True, False, True, False]
        result = mcnemar_test(a, b)
        self.assertEqual(result["n01"], 1)
        self.assertEqual(result["n10"], 1)
        self.assertGreater(result["p_two_sided"], 0.05)


class TestSummarizeSeedMetrics(unittest.TestCase):
    def test_summarize_seed_metrics_reports_mean_and_std_of_j(self):
        rows = [
            {"llm_judge_pct": 66.7},
            {"llm_judge_pct": 67.0},
            {"llm_judge_pct": 66.9},
        ]
        out = summarize_seed_metrics(rows, keys=("llm_judge_pct",))
        self.assertEqual(out["n_seeds"], 3)
        self.assertAlmostEqual(out["llm_judge_pct"]["mean"], 66.866666, places=3)
        self.assertIsNotNone(out["llm_judge_pct"]["std"])


if __name__ == "__main__":
    unittest.main()
