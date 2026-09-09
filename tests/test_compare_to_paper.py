"""Paper vs local LLM-as-a-Judge comparison (offline; no API)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.compare_to_paper import (
    best_local_score,
    compare_to_paper,
    infer_paper_method,
    merge_best_by_method,
)
from src.locomo_eval.mem0_baselines import (
    literature_overall_j,
    paper_method_from_label,
    rag_paper_method,
)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _autorater_metrics(
    *,
    j_pct: float,
    memory_type: str = "Mem0",
    provider: str = "openai",
    by_category: dict | None = None,
) -> dict:
    return {
        "number_of_questions": 10,
        "n_judged": 8,
        "n_skipped": 2,
        "skip_category": 5,
        "metrics": {
            "n": 8,
            "llm_judge_pct": j_pct,
            "mem0_f1_pct": 30.0,
            "mem0_bleu1_pct": 20.0,
        },
        "by_category": by_category
        or {
            "single_hop": {"n": 4, "llm_judge_pct": j_pct, "mem0_f1_pct": 31.0},
        },
        "memory_type": memory_type,
        "autorater_model": "gpt-4o-mini",
        "autorater_provider": provider,
        "score_kind": (
            "mock_sanity_not_llm_judge" if provider == "mock" else "llm_as_a_judge"
        ),
    }


class TestPaperMethodMapping(unittest.TestCase):
    def test_paper_method_from_label_maps_memory_type_ids(self):
        self.assertEqual(paper_method_from_label("full_context"), "Full-context")
        self.assertEqual(paper_method_from_label("openai_memory"), "OpenAI")
        self.assertEqual(paper_method_from_label("mem0"), "Mem0")
        self.assertEqual(paper_method_from_label("Mem0g"), "Mem0g")

    def test_paper_method_from_label_keeps_table_two_names(self):
        self.assertEqual(paper_method_from_label("RAG k=2 256"), "RAG k=2 256")
        self.assertEqual(paper_method_from_label("Full-context"), "Full-context")

    def test_rag_paper_method_defaults_to_best_paper_cell(self):
        self.assertEqual(rag_paper_method(None, None), "RAG k=2 256")
        self.assertEqual(rag_paper_method(1, 128), "RAG k=1 128")


class TestBestLocalJudgeScore(unittest.TestCase):
    def test_best_local_score_picks_max_j_and_ignores_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "mem0_qa"
            _write_json(run / "run_meta.json", {"run_id": "mem0_qa", "memory_type": "mem0"})
            _write_json(
                run / "autorater" / "autorater_metrics.json",
                _autorater_metrics(j_pct=48.0, memory_type="Mem0"),
            )
            _write_json(
                run / "autorater" / "run_meta.json",
                {"autorater_provider": "openai"},
            )
            _write_json(
                run / "autorater_seeds" / "seed_00" / "autorater_metrics.json",
                _autorater_metrics(j_pct=51.2, memory_type="Mem0"),
            )
            _write_json(
                run / "autorater_seeds" / "seed_00" / "run_meta.json",
                {"autorater_provider": "openai"},
            )
            _write_json(
                run / "autorater_seeds" / "seed_01" / "autorater_metrics.json",
                _autorater_metrics(j_pct=90.0, memory_type="Mem0", provider="mock"),
            )
            _write_json(
                run / "autorater_seeds" / "seed_01" / "run_meta.json",
                {"autorater_provider": "mock"},
            )
            score = best_local_score(run)
            self.assertIsNotNone(score)
            assert score is not None
            self.assertEqual(score["method"], "Mem0")
            self.assertEqual(score["local_j"], 51.2)
            self.assertEqual(score["n_judge_packs"], 2)
            self.assertIn("seed_00", score["selected_pack"])

    def test_infer_paper_method_builds_rag_label_from_k_and_index_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "rag_k2_256_qa"
            index = root / "rag_locomo10" / "rag_index"
            _write_json(
                run / "run_meta.json",
                {
                    "run_id": "rag_k2_256_qa",
                    "memory_type": "rag",
                    "rag_k": 2,
                    "rag_index_run_id": "rag_locomo10",
                },
            )
            _write_json(index / "run_meta.json", {"chunk_size": 256})
            self.assertEqual(infer_paper_method(run), "RAG k=2 256")

    def test_merge_best_by_method_keeps_higher_j(self):
        merged = merge_best_by_method(
            [
                {"method": "Mem0", "local_j": 40.0, "n_judge_packs": 1, "run_id": "a"},
                {"method": "Mem0", "local_j": 48.0, "n_judge_packs": 2, "run_id": "b"},
            ]
        )
        self.assertEqual(merged["Mem0"]["local_j"], 48.0)
        self.assertEqual(merged["Mem0"]["n_judge_packs"], 3)
        self.assertEqual(merged["Mem0"]["run_id"], "b")

    def test_compare_to_paper_writes_grouped_table_and_uses_paper_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fc = root / "full_context_qa"
            _write_json(
                fc / "run_meta.json",
                {"run_id": "full_context_qa", "memory_type": "full_context"},
            )
            _write_json(
                fc / "autorater" / "autorater_metrics.json",
                _autorater_metrics(j_pct=74.68, memory_type="Full-context"),
            )
            _write_json(
                fc / "autorater" / "run_meta.json",
                {"autorater_provider": "openai"},
            )
            out = root / "compare"
            result = compare_to_paper(
                [str(fc)],
                out_dir=out,
                methods=["Full-context", "Mem0", "Mem0g"],
            )
            by_method = {row["method"]: row for row in result["rows"]}
            self.assertEqual(by_method["Full-context"]["local_j"], 74.68)
            self.assertEqual(
                by_method["Full-context"]["paper_j"], literature_overall_j("Full-context")
            )
            self.assertIsNone(by_method["Mem0"]["local_j"])
            self.assertEqual(by_method["Mem0"]["paper_j"], 66.88)
            self.assertTrue((out / "tables" / "overall.csv").is_file())
            self.assertTrue((out / "SUMMARY.md").is_file())
            summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("best", summary.lower())
            self.assertIn("74.68", summary)


if __name__ == "__main__":
    unittest.main()
