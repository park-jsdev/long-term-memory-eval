"""Context-window utilization vs memory coverage (offline, no API)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.mem0_baselines import paper_full_context_memory_tokens
from src.experiment_runner.analysis.context_window import (
    coverage_kind,
    judged_mask,
    load_context_windows,
    render_context_window_report,
    resolve_window,
    utilization,
)


def _write_pack(root: Path, name: str, rows: list[dict]) -> None:
    agg = root / "experiments" / name / "aggregate"
    agg.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(agg / "examples.parquet", index=False)


def _example(
    *,
    memory: str,
    reader: str,
    tokens: int,
    f1: float,
    judge: float,
    category: int = 4,
    thinking: str = "off",
    latency: float = 1.0,
    writer: str | None = None,
    generation: str = "2025",
) -> dict:
    return {
        "run_id": f"{memory}-{reader}-{thinking}",
        "memory_method": memory,
        "reader_model": reader,
        "reader_generation": generation,
        "writer_model": writer,
        "thinking": thinking,
        "question_category": category,
        "agent_input_tokens": tokens,
        "agent_output_tokens": 10,
        "agent_latency_seconds": latency,
        "locomo_f1": f1,
        "judge_score": judge,
        "question_id": f"q-{category}-{tokens}",
    }


class TestPaperFullContextPin(unittest.TestCase):
    def test_paper_full_context_memory_tokens_is_table2_26031(self):
        self.assertEqual(paper_full_context_memory_tokens(), 26031)


class TestCoverageKind(unittest.TestCase):
    def test_full_context_covers_all_turns_rag_is_topk(self):
        self.assertEqual(coverage_kind("full_context"), "all_turns")
        self.assertEqual(coverage_kind("session_summaries"), "all_sessions_compressed")
        self.assertEqual(coverage_kind("rag"), "topk_chunks")
        self.assertEqual(coverage_kind("mem0g"), "retrieved_facts_plus_graph")


class TestWindowResolveAndUtilization(unittest.TestCase):
    def test_gpt4o_mini_window_is_128k_and_fc_fill_is_about_one_fifth(self):
        spec = resolve_window("gpt-4o-mini")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.context_window, 128000)
        rate = utilization(27848, spec.context_window)
        self.assertAlmostEqual(rate, 27848 / 128000, places=6)

    def test_deepseek_v3_alias_uses_chat_128k(self):
        spec = resolve_window("deepseek-v3")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.context_window, 128000)

    def test_gpt5_window_is_400k_with_272k_max_input(self):
        spec = resolve_window("gpt-5")
        self.assertEqual(spec.context_window, 400000)
        self.assertEqual(spec.max_input, 272000)

    def test_terra_window_is_about_one_million(self):
        spec = resolve_window("gpt-5.6-terra")
        self.assertGreaterEqual(spec.context_window, 1_000_000)

    def test_utilization_none_when_window_missing(self):
        self.assertIsNone(utilization(100, None))
        self.assertIsNone(utilization(None, 128000))

    def test_load_windows_yaml_pins_source_strings(self):
        table = load_context_windows()
        self.assertIn("gpt-4o-mini", table)
        self.assertTrue(table["gpt-4o-mini"].source)


class TestJudgedMask(unittest.TestCase):
    def test_judged_mask_drops_category_five(self):
        df = pd.DataFrame({"question_category": [1, 5, 4]})
        mask = judged_mask(df)
        self.assertEqual(mask.tolist(), [True, False, True])


class TestRenderContextWindowReport(unittest.TestCase):
    def test_report_writes_cells_and_fc_outcovers_rag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pack(
                root,
                "locomo-2025-readers-openai-deepseek",
                [
                    _example(
                        memory="full_context",
                        reader="gpt-5",
                        tokens=28000,
                        f1=0.45,
                        judge=0.80,
                        generation="2025",
                    ),
                    _example(
                        memory="full_context",
                        reader="gpt-5",
                        tokens=28000,
                        f1=0.45,
                        judge=0.0,
                        category=5,
                        generation="2025",
                    ),
                    _example(
                        memory="rag",
                        reader="gpt-5",
                        tokens=900,
                        f1=0.34,
                        judge=0.58,
                        generation="2025",
                    ),
                ],
            )
            out = root / "report"
            report = render_context_window_report(
                root=root,
                out_dir=out,
                packs=["locomo-2025-readers-openai-deepseek"],
                audit_run=None,
                preprocess=None,
                local_runs=(),
            )
            self.assertTrue((out / "SUMMARY.md").is_file())
            self.assertTrue((out / "tables" / "cells.csv").is_file())
            self.assertFalse(report.cell_table.empty)
            fc = report.cell_table[
                report.cell_table["memory_method"] == "full_context"
            ]
            fc = fc[fc["result_source"] != "paper"]
            rag = report.cell_table[report.cell_table["memory_method"] == "rag"]
            self.assertAlmostEqual(float(fc.iloc[0]["window_utilization"]), 28000 / 400000)
            self.assertAlmostEqual(float(fc.iloc[0]["judge_score"]), 0.80)
            self.assertEqual(int(fc.iloc[0]["n_judged"]), 1)
            self.assertGreater(
                float(fc.iloc[0]["agent_input_tokens_mean"]),
                float(rag.iloc[0]["agent_input_tokens_mean"]),
            )
            self.assertLess(float(rag.iloc[0]["coverage_vs_full_context"]), 0.1)
            summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("not the dataset", summary)
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                matplotlib = None
            if matplotlib is not None:
                self.assertTrue(any(p.suffix == ".png" for p in report.plot_paths))

    def test_conversation_audit_joins_sessions_and_billed_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "experiments" / "full_context_qa"
            mem = run / "memory"
            mem.mkdir(parents=True)
            (mem / "index.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "conv-26",
                        "memory_type": "full_context",
                        "n_chars": 81592,
                        "n_source_ids": 419,
                        "key_kind": "sample",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            preds = [
                {
                    "sample_id": "conv-26",
                    "question_id": "q1",
                    "category": 4,
                    "memory_type": "full_context",
                    "reader_model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 21205, "completion_tokens": 6},
                    "latency_s": 1.0,
                }
            ]
            with (run / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for rec in preds:
                    handle.write(json.dumps(rec) + "\n")
            (run / "metrics.json").write_text(
                json.dumps({"metrics": {"locomo_f1": 0.42}}), encoding="utf-8"
            )
            pre = root / "experiments" / "locomo_preprocess" / "preprocess" / "by_sample" / "conv-26"
            pre.mkdir(parents=True)
            (pre / "sessions.jsonl").write_text(
                json.dumps({"session_id": 1}) + "\n" + json.dumps({"session_id": 2}) + "\n",
                encoding="utf-8",
            )
            report = render_context_window_report(
                root=root,
                out_dir=root / "report",
                packs=[],
                audit_run="full_context_qa",
                preprocess="locomo_preprocess",
                local_runs=("full_context_qa",),
            )
            conv = report.conversation_table.iloc[0]
            self.assertEqual(int(conv["n_sessions"]), 2)
            self.assertEqual(int(conv["n_turns"]), 419)
            self.assertAlmostEqual(float(conv["agent_input_tokens_mean"]), 21205)
            clone = report.cell_table[report.cell_table["pack"] == "full_context_qa"]
            self.assertAlmostEqual(
                float(clone.iloc[0]["window_utilization"]), 21205 / 128000
            )


class TestContextWindowNotebookContract(unittest.TestCase):
    def test_notebook_16_calls_helper_and_avoids_matplotlib_groupby(self):
        notebook = ROOT / "notebooks" / "16_locomo_full_context_window_analysis.ipynb"
        self.assertTrue(notebook.is_file())
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source") or [])
            for cell in payload.get("cells") or []
            if cell.get("cell_type") == "code"
        )
        md = "\n".join(
            "".join(cell.get("source") or [])
            for cell in payload.get("cells") or []
            if cell.get("cell_type") == "markdown"
        )
        self.assertIn("render_context_window_report", code)
        self.assertIn("notebook_show_context_window", code)
        self.assertNotIn("matplotlib", code)
        self.assertNotIn("groupby(", code)
        self.assertIn("entire conversation", md.lower())
        self.assertIn("26,031", md.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
