"""Deterministic experiment-pack verifier (no API)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.experiment_pack.audit_layout import audit_layout_meta
from src.locomo_eval.experiment_pack.verify_graph_years import diagnose_graph_year_stagnation
from src.locomo_eval.experiment_pack.verify_pack import (
    QA_MEM0_V1,
    discover_run_dirs,
    verify_pack,
)
from src.locomo_eval.prompts import locate_prompt_file


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _graph_pack(
    run: Path,
    *,
    parse_fallback: int = 0,
    n_calls: int = 19,
    gold: str = "a unique gold phrase xyz",
    memory_text: str | None = None,
    n_valid_edges: int = 40,
    message_layout: str = "mem0_system_only",
    include_memory: bool = True,
) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run.name,
        "memory_type": "graph",
        "reader_model": "gpt-4o-mini",
        "teacher_model": "gpt-5",
        "prompt_path": QA_MEM0_V1,
        "message_layout": message_layout,
        "n_predictions": 1,
        "audit_layout": audit_layout_meta(),
    }
    _write_json(run / "run_meta.json", meta)
    _write_json(
        run / "metrics.json",
        {"metrics": {"locomo_f1": 0.3}, "memory_type": "graph"},
    )
    text = memory_text or (
        "Conversation between Caroline and Melanie.\n\n"
        "Graph relations:\n"
        "caroline -- attended -- lgbtq_support_group\n"
        "melanie -- enjoys -- painting\n"
    )
    pred = {
        "question_id": "q0",
        "sample_id": "conv-26",
        "reference_answer": gold,
        "memory_text": text,
        "predicted_answer": "painting",
        "category": 4,
    }
    _write_jsonl(run / "predictions.jsonl", [pred])
    (run / "TRACE.md").write_text("# TRACE\n", encoding="utf-8")
    (run / "config.source.yaml").write_text("pipeline: {memory: graph}\n", encoding="utf-8")
    (run / "config.resolved.yaml").write_text("yaml: {}\n", encoding="utf-8")
    _write_json(run / "cost.json", {"reader": {"prompt_tokens": 10}})
    qa_src = locate_prompt_file(QA_MEM0_V1)
    graph_src = locate_prompt_file("prompts/writers/graph_v1.txt")
    qa_dest = run / QA_MEM0_V1
    graph_dest = run / "prompts/writers/graph_v1.txt"
    qa_dest.parent.mkdir(parents=True, exist_ok=True)
    graph_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(qa_src, qa_dest)
    shutil.copy2(graph_src, graph_dest)
    qa_hash = _sha(qa_dest)
    graph_hash = _sha(graph_dest)
    _write_json(
        run / "prompts" / "index.json",
        {
            "prompts": [
                {
                    "role": "reader",
                    "config_key": "pipeline.prompt_path",
                    "source_path": QA_MEM0_V1,
                    "snapshot_path": QA_MEM0_V1,
                    "sha256": qa_hash,
                },
                {
                    "role": "teacher",
                    "config_key": "teacher.graph_prompt_path",
                    "source_path": "prompts/writers/graph_v1.txt",
                    "snapshot_path": "prompts/writers/graph_v1.txt",
                    "sha256": graph_hash,
                },
            ]
        },
    )
    if include_memory:
        _write_jsonl(
            run / "memory" / "index.jsonl",
            [
                {
                    "sample_id": "conv-26",
                    "memory_type": "graph",
                    "n_chars": len(text),
                    "n_source_ids": n_valid_edges,
                    "key_kind": "sample",
                }
            ],
        )
        (run / "memory" / "by_sample").mkdir(parents=True, exist_ok=True)
        (run / "memory" / "by_sample" / "conv-26.txt").write_text(text, encoding="utf-8")
        _write_jsonl(
            run / "memory" / "graph" / "index.jsonl",
            [
                {
                    "sample_id": "conv-26",
                    "n_nodes": 8,
                    "n_edges": n_valid_edges,
                    "n_valid_edges": n_valid_edges,
                }
            ],
        )
        _write_json(
            run / "memory" / "writer" / "quality.json",
            {
                "by_writer": {
                    "openai": {
                        "n_calls": n_calls,
                        "n_parse_ok": n_calls - parse_fallback,
                        "n_parse_fallback": parse_fallback,
                        "n_relations": n_valid_edges,
                    }
                },
            },
        )
    return run


def _sha(path: Path) -> str:
    import hashlib

    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestVerifyPackLayout(unittest.TestCase):
    def test_missing_run_meta_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "empty"
            run.mkdir()
            report = verify_pack(run)
            self.assertEqual(report.verdict, "invalid")
            self.assertEqual(report.pack_kind, "incomplete")
            ids = {c.id: c.status for c in report.checks}
            self.assertEqual(ids["layout.run_meta"], "fail")

    def test_prediction_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(Path(tmp) / "locomo-mem0-reader-x")
            meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
            meta["n_predictions"] = 99
            _write_json(run / "run_meta.json", meta)
            report = verify_pack(run)
            check = next(c for c in report.checks if c.id == "layout.predictions")
            self.assertEqual(check.status, "fail")
            self.assertEqual(report.verdict, "invalid")


class TestVerifyPackGraph(unittest.TestCase):
    def test_healthy_graph_smoke_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(Path(tmp) / "locomo-mem0-reader-ok")
            report = verify_pack(run)
            self.assertEqual(report.verdict, "valid")
            by_id = {c.id: c for c in report.checks}
            self.assertEqual(by_id["graph.parse_rate"].status, "pass")
            self.assertEqual(by_id["memory.graph_grammar"].status, "pass")
            self.assertEqual(by_id["gold.not_in_memory"].status, "pass")
            self.assertEqual(by_id["prompt.graph_timeless"].status, "pass")
            self.assertEqual(by_id["config.sandwich_reader"].status, "pass")

    def test_parse_fallback_mock_rate_invalidates_graph_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(
                Path(tmp) / "locomo-mem0-reader-badparse",
                parse_fallback=14,
                n_calls=19,
            )
            report = verify_pack(run)
            self.assertEqual(report.verdict, "invalid")
            check = next(c for c in report.checks if c.id == "graph.parse_rate")
            self.assertEqual(check.status, "fail")
            self.assertIn("fallback_mock", check.detail)

    def test_gold_in_memory_text_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold = "Caroline went to the unique support group"
            run = _graph_pack(
                Path(tmp) / "locomo-mem0-reader-leak",
                gold=gold,
                memory_text=(
                    "Conversation between Caroline and Melanie.\n\n"
                    "Graph relations:\n"
                    f"note -- quotes -- {gold}\n"
                ),
            )
            report = verify_pack(run)
            check = next(c for c in report.checks if c.id == "gold.not_in_memory")
            self.assertEqual(check.status, "fail")

    def test_default_system_user_layout_is_warning_not_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(
                Path(tmp) / "locomo-mem0-reader-layout",
                message_layout="default_system_user",
            )
            report = verify_pack(run)
            check = next(c for c in report.checks if c.id == "config.message_layout")
            self.assertEqual(check.status, "fail")
            self.assertEqual(check.severity, "warning")
            self.assertEqual(report.verdict, "valid_with_warnings")

    def test_thin_catalog_skips_graph_dump_as_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(
                Path(tmp) / "locomo-mem0-reader-thin",
                include_memory=False,
            )
            report = verify_pack(run)
            self.assertEqual(report.pack_kind, "thin_catalog")
            dump = next(c for c in report.checks if c.id == "graph.dump")
            self.assertEqual(dump.status, "fail")
            self.assertEqual(dump.severity, "warning")
            self.assertNotEqual(report.verdict, "invalid")


class TestDiscoverRunDirsPrefersFullAudit(unittest.TestCase):
    def test_full_memory_pack_wins_over_thin_catalog_same_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            thin = root / "aggregate" / "by_run" / "cell-aaaa"
            full = root / "collected" / "runs" / "cell-aaaa"
            _graph_pack(thin, include_memory=False)
            _graph_pack(full, include_memory=True)
            found = discover_run_dirs(root)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0], full)


class TestGraphYearDiagnosis(unittest.TestCase):
    def test_flat_scores_and_similar_tokens_are_scientific_not_technical(self):
        cells = [
            {
                "family": "openai",
                "generation": "2025",
                "thinking": "off",
                "locomo_f1": 0.309,
                "judge_score": 0.475,
                "agent_input_tokens_mean": 4200.0,
            },
            {
                "family": "openai",
                "generation": "2026",
                "thinking": "off",
                "locomo_f1": 0.305,
                "judge_score": 0.470,
                "agent_input_tokens_mean": 4689.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(Path(tmp) / "locomo-mem0-reader-ok")
            report = verify_pack(run)
            diag = diagnose_graph_year_stagnation(
                [],
                pack_reports=[report],
                cells=cells,
            )
        self.assertTrue(any("not_moving" in item or "frozen" in item.lower() for item in diag.scientific))
        labels = {p.label for p in diag.pairs if p.metric == "locomo_f1"}
        self.assertIn("not_moving", labels)
        self.assertFalse(diag.technical)

    def test_declining_scores_count_as_did_not_improve(self):
        cells = [
            {
                "family": "openai",
                "generation": "2025",
                "thinking": "off",
                "locomo_f1": 0.309,
                "judge_score": 0.475,
                "agent_input_tokens_mean": 4210.0,
            },
            {
                "family": "openai",
                "generation": "2026",
                "thinking": "off",
                "locomo_f1": 0.287,
                "judge_score": 0.449,
                "agent_input_tokens_mean": 4685.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(Path(tmp) / "locomo-mem0-reader-ok")
            report = verify_pack(run)
            diag = diagnose_graph_year_stagnation(
                [],
                pack_reports=[report],
                cells=cells,
            )
        labels = {p.label for p in diag.pairs if p.metric == "locomo_f1"}
        self.assertIn("declining", labels)
        self.assertIn("did not improve", diag.headline.lower())
        self.assertTrue(any("did not improve" in item for item in diag.scientific))

    def test_parse_fallback_pack_is_labeled_technical(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(
                Path(tmp) / "locomo-mem0-reader-badparse",
                parse_fallback=14,
                n_calls=19,
            )
            report = verify_pack(run)
            diag = diagnose_graph_year_stagnation(
                [],
                pack_reports=[report],
                cells=[],
            )
        self.assertTrue(diag.technical)
        self.assertIn("invalid", diag.headline.lower() + " " + " ".join(diag.technical).lower())


class TestVerifyExperimentsCli(unittest.TestCase):
    def test_cli_returns_nonzero_on_invalid_pack(self):
        from scripts.analysis.verify_experiments import main

        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(
                Path(tmp) / "locomo-mem0-reader-badparse",
                parse_fallback=14,
                n_calls=19,
            )
            code = main([str(run)])
            self.assertEqual(code, 1)

    def test_cli_returns_zero_on_valid_pack(self):
        from scripts.analysis.verify_experiments import main

        with tempfile.TemporaryDirectory() as tmp:
            run = _graph_pack(Path(tmp) / "locomo-mem0-reader-ok")
            code = main([str(run)])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
