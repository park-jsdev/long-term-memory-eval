"""Claim-audit helpers: lineage, ranks, ingest, cost, frozen config (no API)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.experiments.audit_layout import AUDIT_LAYOUT_VERSION, AuditPaths
from src.locomo_eval.experiments.audit_loader import load_sandwich_audit
from src.locomo_eval.experiments.audit_writer import (
    write_claim_audit,
    write_frozen_config,
    write_graph_ingest,
    write_teacher_module,
)
from src.locomo_eval.experiments.claim_audit import (
    cost_rollup,
    lineage_rows,
    ranked_candidates,
    teacher_quality_stats,
)
from src.locomo_eval.mem0.embeddings import MockEmbedder
from src.locomo_eval.mem0.graph_memory import Mem0GraphMemory
from src.locomo_eval.mem0.vector_store import VectorMemoryStore
from src.locomo_eval.mem0.schemas import Fact
from src.locomo_eval.mem0.retrieve import rank_speaker_facts
from src.locomo_eval.rag.chunk import chunk_transcript
from src.locomo_eval.rag.dump import write_sample_dump
from src.locomo_eval.rag.retrieve import retrieve_rag_with_ranks
from src.locomo_eval.schemas import Memory


class TestRetrieveRanksIncludeLosers(unittest.TestCase):
    def test_rank_speaker_facts_marks_non_top_k_unselected(self):
        embedder = MockEmbedder()
        store = VectorMemoryStore(sample_id="s", speaker_index="a", speaker_name="Alice")
        painting = Fact(
            fact_id="a:0000",
            text="Alice started painting yesterday",
            timestamp="1 Jan 2023",
            embedding=embedder.embed_one("Alice started painting yesterday"),
            speaker_index="a",
        )
        pizza = Fact(
            fact_id="a:0001",
            text="Bob likes pizza",
            timestamp="1 Jan 2023",
            embedding=embedder.embed_one("Bob likes pizza"),
            speaker_index="a",
        )
        store.add(painting)
        store.add(pizza)
        rows = rank_speaker_facts(
            store, embedder.embed_one("What did Alice start painting"), top_k=1
        )
        self.assertGreaterEqual(len(rows), 2)
        selected = [r for r in rows if r["selected"]]
        losers = [r for r in rows if not r["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["item_id"], "a:0000")
        self.assertTrue(losers)
        self.assertEqual(rows[0]["rank"], 1)

    def test_rag_ranks_keep_unselected_chunks(self):
        transcript = "Alice started painting in Boston. Alice got a nursing job in Seattle."
        chunks = chunk_transcript(transcript, 4, sample_id="conv-rag")
        embedder = MockEmbedder()
        for chunk, emb in zip(chunks, embedder.embed([c.text for c in chunks])):
            chunk.embedding = emb
        with tempfile.TemporaryDirectory() as tmp:
            index_root = Path(tmp)
            write_sample_dump(
                index_root,
                sample_id="conv-rag",
                transcript=transcript,
                chunks=chunks,
                chunk_size=4,
                encoding="cl100k_base",
            )
            _text, source_ids, _s, ranks = retrieve_rag_with_ranks(
                index_root, "conv-rag", "nursing job", embedder, k=1
            )
            self.assertEqual(len(source_ids), 1)
            if len(chunks) > 1:
                self.assertGreater(len(ranks), 1)
                self.assertEqual(sum(1 for r in ranks if r["selected"]), 1)
                self.assertTrue(any(not r["selected"] for r in ranks))


class TestGraphIngestAfterFusion(unittest.TestCase):
    def test_second_identical_triple_logs_skip_dup(self):
        graph = Mem0GraphMemory(MockEmbedder(), threshold=0.7)
        rels = [{"source": "alice", "relationship": "started", "target": "painting"}]
        ents = [{"entity": "alice", "entity_type": "person"}]
        first = graph.ingest_triples(
            entities=ents, relations=rels, user_id="Alice", timestamp="1 Jan"
        )
        second = graph.ingest_triples(
            entities=ents, relations=rels, user_id="Alice", timestamp="2 Jan"
        )
        self.assertTrue(any(op["op"] == "add_edge" for op in first))
        self.assertTrue(any(op["op"] == "skip_dup" for op in second))
        self.assertTrue(any(op["op"] == "new_node" for op in first))


class TestLineageJoinsQuestionToTeacher(unittest.TestCase):
    def test_graph_lineage_uses_ingest_edge_id_and_fusion_proposed_by(self):
        memory = Memory(
            memory_type="fused_teacher_graph",
            text="alice -- started -- painting",
            source_ids=["e0000"],
        )
        rows = lineage_rows(
            prediction_rows=[
                {"question_id": "q0", "sample_id": "s1", "predicted_answer": "painting"}
            ],
            memories_by_sample={"s1": memory},
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
                }
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
                            "proposed_by": ["openai", "anthropic"],
                            "votes": 2,
                            "kept": True,
                        }
                    ],
                }
            ],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question_id"], "q0")
        self.assertEqual(rows[0]["item_id"], "e0000")
        self.assertEqual(rows[0]["proposed_by"], ["openai", "anthropic"])


class TestTeacherQualityAndCost(unittest.TestCase):
    def test_quality_counts_parse_ok_and_keep_rate(self):
        stats = teacher_quality_stats(
            calls=[
                {
                    "teacher_id": "openai",
                    "model": "gpt-4o-mini",
                    "parse": "ok",
                    "n_entities": 2,
                    "n_relations": 2,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "latency_s": 0.2,
                }
            ],
            fusion_rows=[
                {
                    "teacher_ids": ["openai"],
                    "relations": [
                        {"proposed_by": ["openai"], "kept": True},
                        {"proposed_by": ["openai"], "kept": False},
                    ],
                }
            ],
        )
        rec = stats["by_teacher"]["openai"]
        self.assertEqual(rec["n_parse_ok"], 1)
        self.assertEqual(rec["n_proposed_kept"], 1)
        self.assertEqual(stats["fusion"]["n_triples_kept"], 1)
        self.assertEqual(stats["fusion"]["n_triples_discarded"], 1)

    def test_cost_rollup_prices_known_openai_model(self):
        cost = cost_rollup(
            reader_traces=[
                {
                    "model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000},
                }
            ],
            teacher_calls=[],
        )
        self.assertEqual(cost["reader"]["usd"], 0.15)
        self.assertEqual(cost["total"]["usd"], 0.15)

    def test_ranked_candidates_sets_rank_and_selected(self):
        rows = ranked_candidates(
            [{"item_id": "a", "score": 0.9}, {"item_id": "b", "score": 0.1}],
            selected_ids={"a"},
        )
        self.assertEqual(rows[0]["rank"], 1)
        self.assertTrue(rows[0]["selected"])
        self.assertFalse(rows[1]["selected"])


class TestClaimAuditDump(unittest.TestCase):
    def test_write_claim_audit_writes_summary_cost_lineage_and_frozen_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            write_frozen_config(
                run,
                cfg={"pipeline": {"memory": "teacher_graph"}},
                overrides=type("NS", (), {"config": None, "reader": "mock", "max_questions": 1})(),
            )
            write_teacher_module(
                run,
                calls=[
                    {
                        "sample_id": "s1",
                        "session_id": 1,
                        "teacher_id": "openai",
                        "model": "gpt-4o-mini",
                        "parse": "ok",
                        "n_entities": 1,
                        "n_relations": 1,
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                    }
                ],
                fusion_rows=[
                    {
                        "sample_id": "s1",
                        "session_id": 1,
                        "teacher_ids": ["openai"],
                        "relations": [
                            {
                                "source": "alice",
                                "relationship": "started",
                                "target": "painting",
                                "proposed_by": ["openai"],
                                "votes": 1,
                                "kept": True,
                            }
                        ],
                    }
                ],
                session_texts=[
                    {
                        "sample_id": "s1",
                        "session_id": 1,
                        "session_index": 0,
                        "text": "DATE: 1 Jan 2023\nSESSION 1:\nAlice: I started painting.",
                    }
                ],
            )
            write_graph_ingest(
                run,
                [
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
                    }
                ],
            )
            memory = Memory(
                memory_type="teacher_graph",
                text="alice -- started -- painting",
                source_ids=["e0000"],
            )
            write_claim_audit(
                run,
                prediction_rows=[
                    {
                        "question_id": "q0",
                        "sample_id": "s1",
                        "predicted_answer": "painting",
                    }
                ],
                reader_traces=[
                    {
                        "question_id": "q0",
                        "model": "mock",
                        "usage": {},
                    }
                ],
                teacher_calls=[
                    {
                        "teacher_id": "openai",
                        "model": "gpt-4o-mini",
                        "parse": "ok",
                        "n_entities": 1,
                        "n_relations": 1,
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                    }
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
                                "votes": 1,
                                "kept": True,
                            }
                        ],
                    }
                ],
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
                    }
                ],
                memories_by_sample={"s1": memory},
                metrics={"metrics": {"locomo_f1": 0.5, "token_f1": 0.4, "exact_match": 0.0}},
                meta={"run_id": "r1", "memory_type": "teacher_graph", "reader_model": "mock"},
            )
            paths = AuditPaths.from_run_dir(run)
            self.assertEqual(AUDIT_LAYOUT_VERSION, "audit_pack.v2")
            self.assertTrue(paths.summary.is_file())
            self.assertTrue(paths.cost.is_file())
            self.assertTrue(paths.lineage.is_file())
            self.assertTrue(paths.config_resolved.is_file())
            self.assertTrue(paths.teacher_sessions_index.is_file())
            session_txt = paths.teacher_session_text_path("s1", 1)
            self.assertTrue(session_txt.is_file())
            self.assertIn("painting", session_txt.read_text(encoding="utf-8"))
            pack = load_sandwich_audit(run)
            lineage = pack.lineage_for(question_id="q0")
            self.assertEqual(len(lineage), 1)
            self.assertEqual(lineage[0]["proposed_by"], ["openai"])
            summary = paths.summary.read_text(encoding="utf-8")
            self.assertIn("audit of claims", summary)
            self.assertIn("How to audit one claim", summary)
            cost = json.loads(paths.cost.read_text(encoding="utf-8"))
            self.assertIn("total", cost)


if __name__ == "__main__":
    unittest.main()
