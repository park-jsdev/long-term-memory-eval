"""Eval pipeline orchestration: index + QA + mock autorater (no live API)."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.eval_pipeline import run_eval_pipeline
from src.locomo_eval.models import resolve_model
from src.locomo_eval.readers import get_reader

GOLD_LOCK = "UNIQ_GOLD_REF_ZZZ"

MINI = {
    "sample_id": "conv-eval",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice!"},
        ],
    },
    "session_summary": {"session_1_summary": "Alice paints."},
    "qa": [
        {
            "question": "What hobby did Alice begin?",
            "answer": GOLD_LOCK,
            "category": 4,
            "evidence": ["D1:1"],
        }
    ],
}


def _args(tmp: Path, **kwargs) -> Namespace:
    data = Path(tmp) / "locomo.json"
    data.write_text(json.dumps([MINI]), encoding="utf-8")
    defaults = dict(
        method="full_context",
        config=str(ROOT / "configs" / "writers" / "full_context.yaml"),
        data=str(data),
        memory=None,
        reader="mock",
        model="gpt-4o-mini",
        temperature=None,
        max_tokens=None,
        message_layout=None,
        writer=None,
        writer_model=None,
        writer_max_tokens=None,
        prompt=str(ROOT / "prompts" / "readers" / "qa_mem0_v1.txt"),
        output_dir=str(Path(tmp) / "experiments"),
        run_id="smoke_eval",
        max_questions=1,
        sample_id=None,
        max_samples=1,
        index_run_id=None,
        skip_index=False,
        force_index=False,
        skip_qa=False,
        extractor="mock",
        embedder="mock",
        chunk_size=16,
        k=1,
        autorater="mock",
        n_judge_runs=2,
        offline_evaluate=False,
        label="full_context",
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestEvalPipelineFullContextMock(unittest.TestCase):
    def test_full_context_pipeline_writes_qa_pack_and_seed_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                run_dir = run_eval_pipeline(_args(tmp))
            self.assertTrue((run_dir / "predictions.jsonl").is_file())
            self.assertTrue((run_dir / "memory" / "schema.json").is_file())
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["memory_type"], "full_context")
            row = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn(GOLD_LOCK, row["memory_text"])
            self.assertTrue((run_dir / "autorater" / "seed_aggregate.json").is_file())
            agg = json.loads((run_dir / "autorater" / "seed_aggregate.json").read_text(encoding="utf-8"))
            self.assertEqual(agg["n_seeds"], 2)

    def test_rag_pipeline_indexes_then_retrieves_without_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(
                tmp,
                method="rag",
                config=str(ROOT / "configs" / "writers" / "rag.yaml"),
                run_id="smoke_rag",
                index_run_id="smoke_rag_idx",
                n_judge_runs=1,
                label="rag",
            )
            with redirect_stdout(io.StringIO()):
                run_dir = run_eval_pipeline(args)
            self.assertTrue(
                (Path(tmp) / "experiments" / "smoke_rag_idx" / "rag_index" / "run_meta.json").is_file()
            )
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["memory_type"], "rag")
            self.assertTrue((run_dir / "memory" / "by_question").is_dir())
            row = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn(GOLD_LOCK, row["memory_text"])
            self.assertIn("search_latency_s", row)


class TestEvalIndexSampleScope(unittest.TestCase):
    def test_subset_smoke_uses_run_id_index_not_yaml_full_dump_id(self):
        args = Namespace(
            index_run_id=None,
            max_samples=1,
            sample_id=None,
            run_id="smoke_eval_rag",
        )
        cfg = {"rag": {"index_run_id": "rag_locomo10"}}
        from src.locomo_eval.eval_pipeline import resolve_eval_index_run_id

        self.assertEqual(
            resolve_eval_index_run_id("rag", cfg, args), "smoke_eval_rag_index"
        )

    def test_explicit_index_run_id_wins_even_with_max_samples(self):
        args = Namespace(
            index_run_id="rag_locomo10",
            max_samples=1,
            sample_id=None,
            run_id="smoke_eval_rag",
        )
        cfg = {"rag": {"index_run_id": "rag_locomo10"}}
        from src.locomo_eval.eval_pipeline import resolve_eval_index_run_id

        self.assertEqual(resolve_eval_index_run_id("rag", cfg, args), "rag_locomo10")

    def test_rag_max_samples_keeps_capped_qa_inside_indexed_conversation(self):
        """A capped QA run must not read beyond the indexed conversation."""

        def _sample(sid: str, n_q: int) -> dict:
            return {
                "sample_id": sid,
                "conversation": MINI["conversation"],
                "session_summary": MINI["session_summary"],
                "qa": [
                    {
                        "question": f"{sid} q{i}",
                        "answer": GOLD_LOCK,
                        "category": 4,
                        "evidence": ["D1:1"],
                    }
                    for i in range(n_q)
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            args = _args(
                tmp,
                method="rag",
                config=str(ROOT / "configs" / "writers" / "rag.yaml"),
                run_id="smoke_rag",
                index_run_id=None,
                max_samples=1,
                max_questions=5,
                n_judge_runs=1,
                label="rag",
            )
            Path(args.data).write_text(
                json.dumps([_sample("conv-a", 5), _sample("conv-b", 5)]),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                run_dir = run_eval_pipeline(args)
            rows = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 5)
            self.assertEqual({r["sample_id"] for r in rows}, {"conv-a"})
            out = Path(tmp) / "experiments"
            self.assertTrue(
                (out / "smoke_rag_index" / "rag_index" / "run_meta.json").is_file()
            )
            self.assertFalse(
                (out / "rag_locomo10" / "rag_index" / "run_meta.json").is_file()
            )


class TestReaderCatalogInfra(unittest.TestCase):
    def test_get_reader_mock_resolves_deepseek_and_claude_ids(self):
        deepseek = get_reader("mock", model="deepseek-chat")
        claude = get_reader("mock", model="claude-3-5-haiku-latest")
        self.assertEqual(deepseek.model_name, "deepseek-chat")
        self.assertEqual(claude.model_name, "claude-3-5-haiku-latest")
        self.assertEqual(resolve_model("deepseek-chat").family, "deepseek")
        self.assertEqual(resolve_model("claude-sonnet-4-5").family, "claude")

    def test_get_reader_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            get_reader("not-a-provider", model="gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
