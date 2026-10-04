"""Dump and load tests for a sandwich-run audit (no API)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.experiment_pack.audit_layout import AuditPaths, audit_layout_meta
from src.locomo_eval.experiment_pack.audit_loader import (
    SandwichAudit,
    load_json,
    load_jsonl,
    load_qa_pack,
    load_sandwich_audit,
    predictions_jsonl,
    resolve_predictions_jsonl,
)
from src.locomo_eval.experiment_pack.audit_writer import (
    write_frozen_config,
    write_graph_ingest,
    write_reader_module,
    write_writer_module,
    write_writer_sessions,
)


class TestPredictionsJsonlPrefersRootThenReader(unittest.TestCase):
    def test_predictions_jsonl_returns_root_file_when_both_exist(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "predictions.jsonl").write_text(
                json.dumps({"question_id": "root"}) + "\n", encoding="utf-8"
            )
            reader = run / "reader"
            reader.mkdir()
            (reader / "predictions.jsonl").write_text(
                json.dumps({"question_id": "reader"}) + "\n", encoding="utf-8"
            )
            path = predictions_jsonl(run)
            self.assertEqual(path, run / "predictions.jsonl")

    def test_predictions_jsonl_falls_back_to_reader_when_root_missing(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            reader = run / "reader"
            reader.mkdir(parents=True)
            (reader / "predictions.jsonl").write_text(
                json.dumps({"question_id": "reader"}) + "\n", encoding="utf-8"
            )
            path = predictions_jsonl(run)
            self.assertEqual(path, reader / "predictions.jsonl")
            resolved = resolve_predictions_jsonl(run)
            self.assertEqual(resolved, path)


class TestLoadQaPackAndSandwichAudit(unittest.TestCase):
    def test_load_qa_pack_returns_compare_full_runs_keys(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "metrics.json").write_text(
                json.dumps({"metrics": {"locomo_f1": 0.5}}), encoding="utf-8"
            )
            (run / "run_meta.json").write_text(
                json.dumps({"run_id": "r1", "audit_layout": audit_layout_meta()}),
                encoding="utf-8",
            )
            (run / "predictions.jsonl").write_text(
                json.dumps({"question_id": "q0", "predicted_answer": "a"}) + "\n",
                encoding="utf-8",
            )
            pack = load_qa_pack(run)
            self.assertEqual(
                set(pack),
                {"dir", "run_id", "metrics", "meta", "predictions", "by_qid"},
            )
            self.assertEqual(pack["run_id"], "r1")
            self.assertEqual(pack["by_qid"]["q0"]["predicted_answer"], "a")

    def test_load_sandwich_audit_indexes_writer_calls(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "metrics.json").write_text("{}", encoding="utf-8")
            (run / "run_meta.json").write_text("{}", encoding="utf-8")
            write_reader_module(
                run,
                prediction_rows=[{"question_id": "q0", "sample_id": "s1"}],
                traces=[{"question_id": "q0", "role": "reader", "reasoning": ""}],
            )
            write_writer_module(
                run,
                calls=[
                    {
                        "sample_id": "s1",
                        "session_id": 1,
                        "writer_id": "openai",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "reasoning": "because",
                        "relations": [
                            {"source": "alice", "relationship": "likes", "target": "pizza"}
                        ],
                    }
                ],
            )
            pack = load_sandwich_audit(run)
            self.assertEqual(len(pack.writer_calls_for("openai")), 1)
            self.assertEqual(pack.writer_calls_for("openai")[0]["reasoning"], "because")
            self.assertTrue(AuditPaths.from_run_dir(run).writer_calls.is_file())


class TestOptionalAuditLayersLoadEmptyInsteadOfFailing(unittest.TestCase):
    def test_load_json_and_load_jsonl_return_empty_when_optional_files_are_missing(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            self.assertEqual(load_json(missing), {})
            self.assertEqual(load_jsonl(missing.with_suffix(".jsonl")), [])

    def test_predictions_jsonl_returns_none_when_root_and_reader_copies_are_omitted(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            self.assertIsNone(predictions_jsonl(run))

    def test_load_sandwich_audit_returns_empty_optional_layers_when_only_metrics_exist(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "metrics.json").write_text("{}", encoding="utf-8")
            pack = load_sandwich_audit(run)
            self.assertEqual(pack.predictions, [])
            self.assertEqual(pack.writer_calls, [])
            self.assertEqual(pack.lineage, [])
            self.assertEqual(pack.retrieve_ranks, [])
            self.assertEqual(pack.graph_ingest, [])
            self.assertEqual(pack.attribution, [])
            self.assertEqual(pack.autorater_verdicts, [])
            self.assertEqual(pack.writer_quality, {})
            self.assertEqual(pack.cost, {})

    def test_load_sandwich_audit_falls_back_to_compat_writer_calls_when_writer_dir_is_omitted(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            mem = run / "memory"
            mem.mkdir(parents=True)
            (run / "metrics.json").write_text("{}", encoding="utf-8")
            (mem / "writer_calls.jsonl").write_text(
                json.dumps({"writer_id": "openai", "sample_id": "s1"}) + "\n",
                encoding="utf-8",
            )
            pack = load_sandwich_audit(run)
            self.assertEqual(len(pack.writer_calls), 1)
            self.assertEqual(pack.writer_calls[0]["writer_id"], "openai")

    def test_load_qa_pack_uses_directory_name_when_run_meta_run_id_is_omitted(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "qa_optional_run"
            run.mkdir()
            (run / "metrics.json").write_text("{}", encoding="utf-8")
            pack = load_qa_pack(run)
            self.assertEqual(pack["run_id"], "qa_optional_run")
            self.assertEqual(pack["predictions"], [])
            self.assertEqual(pack["by_qid"], {})

    def test_writer_calls_for_falls_back_to_by_writer_file_when_combined_list_is_empty(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            paths = AuditPaths.from_run_dir(run)
            dest = paths.writer_calls_path("openai")
            dest.parent.mkdir(parents=True)
            dest.write_text(
                json.dumps({"writer_id": "openai", "session_id": 1}) + "\n",
                encoding="utf-8",
            )
            audit = SandwichAudit(paths=paths, meta={}, metrics={}, predictions=[], writer_calls=[])
            self.assertEqual(len(audit.writer_calls_for("openai")), 1)
            self.assertEqual(audit.writer_calls_for("anthropic"), [])

    def test_write_writer_module_returns_none_when_calls_and_sessions_are_omitted(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            self.assertIsNone(write_writer_module(run, calls=[], session_texts=None))
            self.assertIsNone(write_writer_sessions(run, []))
            self.assertIsNone(write_graph_ingest(run, []))


class TestOptionalLoaderFiltersDoNotLeakWhenSetAndReturnAllWhenOmitted(unittest.TestCase):
    def _audit(self) -> SandwichAudit:
        return SandwichAudit(
            paths=AuditPaths.from_run_dir("unused"),
            meta={},
            metrics={},
            predictions=[],
            lineage=[
                {"question_id": "q0", "sample_id": "s1", "item_id": "e-s1"},
                {"question_id": "q0", "sample_id": "s2", "item_id": "e-s2"},
                {"question_id": "q1", "sample_id": "s1", "item_id": "e-s1-q1"},
            ],
            retrieve_ranks=[
                {"question_id": "q0", "sample_id": "s1", "candidates": [{"item_id": "s1-win"}]},
                {"question_id": "q0", "sample_id": "s2", "candidates": [{"item_id": "s2-win"}]},
            ],
            attribution=[
                {
                    "role": "graph",
                    "writer_id": "openai",
                    "sample_id": "s1",
                    "question_id": None,
                    "claims": [{"injected_question_ids": ["q0"]}],
                },
                {
                    "role": "graph",
                    "writer_id": "anthropic",
                    "sample_id": "s2",
                    "question_id": None,
                    "claims": [{"injected_question_ids": ["q0"]}],
                },
                {
                    "role": "reader",
                    "writer_id": None,
                    "sample_id": "s1",
                    "question_id": "q0",
                    "claims": [],
                },
            ],
        )

    def test_lineage_for_returns_all_rows_when_question_and_sample_are_omitted(self):
        self.assertEqual(len(self._audit().lineage_for()), 3)

    def test_lineage_for_question_only_does_not_drop_the_matching_sample_rows(self):
        rows = self._audit().lineage_for(question_id="q0")
        self.assertEqual({row["sample_id"] for row in rows}, {"s1", "s2"})

    def test_lineage_for_sample_only_does_not_include_the_other_sample(self):
        rows = self._audit().lineage_for(sample_id="s1")
        self.assertEqual({row["item_id"] for row in rows}, {"e-s1", "e-s1-q1"})

    def test_lineage_for_question_and_sample_together_does_not_leak_the_sibling_sample(self):
        rows = self._audit().lineage_for(question_id="q0", sample_id="s1")
        self.assertEqual([row["item_id"] for row in rows], ["e-s1"])

    def test_retrieve_ranks_for_omitted_sample_id_returns_the_first_matching_question(self):
        row = self._audit().retrieve_ranks_for(question_id="q0")
        self.assertEqual(row["sample_id"], "s1")

    def test_retrieve_ranks_for_with_sample_id_does_not_return_the_sibling_samples_ranks(self):
        row = self._audit().retrieve_ranks_for(question_id="q0", sample_id="s2")
        self.assertEqual(row["candidates"][0]["item_id"], "s2-win")

    def test_retrieve_ranks_for_returns_none_when_the_optional_sample_has_no_row(self):
        self.assertIsNone(self._audit().retrieve_ranks_for(question_id="q0", sample_id="s-missing"))

    def test_attribution_for_returns_all_calls_when_optional_filters_are_omitted(self):
        self.assertEqual(len(self._audit().attribution_for()), 3)

    def test_attribution_for_role_only_does_not_require_the_other_optional_filters(self):
        rows = self._audit().attribution_for(role="reader")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question_id"], "q0")

    def test_attribution_for_question_and_sample_together_does_not_include_the_other_writer(self):
        rows = self._audit().attribution_for(question_id="q0", sample_id="s1")
        ids = {(row["role"], row.get("writer_id"), row["sample_id"]) for row in rows}
        self.assertEqual(ids, {("graph", "openai", "s1"), ("reader", None, "s1")})
        self.assertNotIn("anthropic", [row.get("writer_id") for row in rows])


class TestSessionTextDedupAndPredictionsSearchIsolation(unittest.TestCase):
    def test_write_writer_sessions_keeps_first_text_and_drops_later_duplicate_sample_session(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            write_writer_sessions(
                run,
                [
                    {"sample_id": "s1", "session_id": 1, "session_index": 0, "text": "first"},
                    {"sample_id": "s1", "session_id": 1, "session_index": 0, "text": "second-should-not-win"},
                    {"sample_id": "s2", "session_id": 1, "session_index": 0, "text": "other-sample"},
                ],
            )
            paths = AuditPaths.from_run_dir(run)
            self.assertEqual(paths.writer_session_text_path("s1", 1).read_text(encoding="utf-8"), "first")
            self.assertEqual(paths.writer_session_text_path("s2", 1).read_text(encoding="utf-8"), "other-sample")
            index = load_jsonl(paths.writer_sessions_index)
            self.assertEqual(len(index), 2)
            self.assertEqual([row["sample_id"] for row in index], ["s1", "s2"])

    def test_resolve_predictions_jsonl_skips_empty_decoy_dir_and_finds_experiments_run(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "iso_run_a").mkdir()
            real = root / "experiments" / "iso_run_a"
            real.mkdir(parents=True)
            (real / "predictions.jsonl").write_text(
                json.dumps({"question_id": "wanted"}) + "\n", encoding="utf-8"
            )
            found = resolve_predictions_jsonl("iso_run_a", repo_root=root)
            self.assertEqual(found, real / "predictions.jsonl")

    def test_resolve_predictions_jsonl_does_not_select_a_sibling_run_with_a_longer_name(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "experiments" / "iso_run_a"
            extra = root / "experiments" / "iso_run_a_extra"
            run_a.mkdir(parents=True)
            extra.mkdir(parents=True)
            (run_a / "predictions.jsonl").write_text(
                json.dumps({"question_id": "a"}) + "\n", encoding="utf-8"
            )
            (extra / "predictions.jsonl").write_text(
                json.dumps({"question_id": "extra"}) + "\n", encoding="utf-8"
            )
            found = resolve_predictions_jsonl("iso_run_a", repo_root=root)
            rows = load_jsonl(found)
            self.assertEqual(rows[0]["question_id"], "a")

    def test_resolve_predictions_jsonl_prefers_an_explicit_jsonl_file_over_a_same_stem_directory(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "iso_explicit.jsonl"
            jsonl.write_text(json.dumps({"question_id": "file"}) + "\n", encoding="utf-8")
            decoy = root / "iso_explicit"
            decoy.mkdir()
            (decoy / "predictions.jsonl").write_text(
                json.dumps({"question_id": "dir"}) + "\n", encoding="utf-8"
            )
            found = resolve_predictions_jsonl(jsonl, repo_root=root)
            self.assertEqual(found, jsonl.resolve())

    def test_write_frozen_config_omits_none_cli_overrides(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            overrides = __import__("types").SimpleNamespace(
                config=None, reader="mock", max_questions=None
            )
            path = write_frozen_config(
                run, cfg={"pipeline": {"memory": "raw_chunks"}}, overrides=overrides
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("reader: mock", text)
            self.assertNotIn("max_questions", text)
            self.assertFalse((run / "config.source.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
