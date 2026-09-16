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
    attribution_call_rows,
    cost_rollup,
    graph_edge_provenance,
    infer_llm_role,
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

    def test_cost_rollup_sums_reasoning_tokens_from_usage(self):
        cost = cost_rollup(
            reader_traces=[
                {
                    "model": "gpt-5",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                        "reasoning_tokens": 8,
                    },
                }
            ],
            teacher_calls=[
                {
                    "model": "gpt-5",
                    "reasoning_tokens": 40,
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 50,
                        "total_tokens": 55,
                    },
                }
            ],
        )
        self.assertEqual(cost["reader"]["reasoning_tokens"], 8)
        self.assertEqual(cost["teacher"]["reasoning_tokens"], 40)
        self.assertEqual(cost["total"]["reasoning_tokens"], 48)

    def test_ranked_candidates_sets_rank_and_selected(self):
        rows = ranked_candidates(
            [{"item_id": "a", "score": 0.9}, {"item_id": "b", "score": 0.1}],
            selected_ids={"a"},
        )
        self.assertEqual(rows[0]["rank"], 1)
        self.assertTrue(rows[0]["selected"])
        self.assertFalse(rows[1]["selected"])


class TestAttributionLinksCallRoleAndClaims(unittest.TestCase):
    def test_teacher_graph_call_lists_kept_triple_and_injected_question(self):
        calls = attribution_call_rows(
            reader_traces=[
                {
                    "sample_id": "s1",
                    "question_id": "q0",
                    "role": "reader",
                    "model": "mock",
                    "predicted_answer": "painting",
                }
            ],
            teacher_calls=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "teacher_id": "openai",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "role": "teacher_graph",
                    "relations": [
                        {"source": "alice", "relationship": "started", "target": "painting"}
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
                            "proposed_by": ["openai"],
                            "votes": 1,
                            "kept": True,
                        }
                    ],
                }
            ],
            lineage=[
                {
                    "question_id": "q0",
                    "sample_id": "s1",
                    "item_id": "e0000",
                    "item_kind": "graph_edge",
                    "session_id": 1,
                    "source": "alice",
                    "relationship": "started",
                    "target": "painting",
                    "proposed_by": ["openai"],
                    "text_preview": "alice -- started -- painting",
                }
            ],
        )
        teachers = [c for c in calls if c["role"] == "teacher_graph"]
        readers = [c for c in calls if c["role"] == "reader"]
        self.assertEqual(len(teachers), 1)
        self.assertEqual(teachers[0]["layer"], "write")
        self.assertEqual(len(teachers[0]["claims"]), 1)
        claim = teachers[0]["claims"][0]
        self.assertEqual(claim["claim_kind"], "graph_triple")
        self.assertTrue(claim["kept"])
        self.assertEqual(claim["injected_question_ids"], ["q0"])
        self.assertEqual(len(readers), 1)
        self.assertEqual(readers[0]["layer"], "read")
        self.assertEqual(readers[0]["claims"][0]["claim_kind"], "predicted_answer")
        self.assertEqual(readers[0]["used_memory_items"][0]["item_id"], "e0000")
        self.assertEqual(readers[0]["used_memory_items"][0]["proposed_by"], ["openai"])

    def test_session_summary_call_lists_summary_text_as_claim(self):
        calls = attribution_call_rows(
            reader_traces=[],
            teacher_calls=[
                {
                    "sample_id": "s1",
                    "session_id": 2,
                    "teacher_id": "openai",
                    "model": "gpt-4o-mini",
                    "role": "teacher",
                    "output_text": "Alice started painting.",
                }
            ],
            lineage=[
                {
                    "question_id": "q1",
                    "sample_id": "s1",
                    "item_id": "session_2_teacher",
                    "item_kind": "teacher_summary",
                    "session_id": 2,
                    "proposed_by": ["openai"],
                }
            ],
        )
        self.assertEqual(calls[0]["role"], "teacher")
        self.assertEqual(calls[0]["claims"][0]["claim_kind"], "session_summary")
        self.assertIn("painting", calls[0]["claims"][0]["text_preview"])
        self.assertEqual(calls[0]["claims"][0]["injected_question_ids"], ["q1"])


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
                        "sample_id": "s1",
                        "question_id": "q0",
                        "role": "reader",
                        "model": "mock",
                        "predicted_answer": "painting",
                        "usage": {},
                    }
                ],
                teacher_calls=[
                    {
                        "sample_id": "s1",
                        "session_id": 1,
                        "teacher_id": "openai",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "role": "teacher_graph",
                        "parse": "ok",
                        "n_entities": 1,
                        "n_relations": 1,
                        "relations": [
                            {
                                "source": "alice",
                                "relationship": "started",
                                "target": "painting",
                            }
                        ],
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
            self.assertTrue(paths.attribution.is_file())
            self.assertTrue(paths.attribution_md.is_file())
            self.assertTrue(paths.config_resolved.is_file())
            self.assertTrue(paths.teacher_sessions_index.is_file())
            session_txt = paths.teacher_session_text_path("s1", 1)
            self.assertTrue(session_txt.is_file())
            self.assertIn("painting", session_txt.read_text(encoding="utf-8"))
            pack = load_sandwich_audit(run)
            lineage = pack.lineage_for(question_id="q0")
            self.assertEqual(len(lineage), 1)
            self.assertEqual(lineage[0]["proposed_by"], ["openai"])
            teacher_attr = pack.attribution_for(role="teacher_graph")
            self.assertEqual(len(teacher_attr), 1)
            self.assertEqual(teacher_attr[0]["claims"][0]["claim_kind"], "graph_triple")
            self.assertTrue(teacher_attr[0]["claims"][0]["kept"])
            reader_attr = pack.attribution_for(role="reader", question_id="q0")
            self.assertEqual(len(reader_attr), 1)
            self.assertEqual(reader_attr[0]["used_memory_items"][0]["proposed_by"], ["openai"])
            summary = paths.summary.read_text(encoding="utf-8")
            self.assertIn("audit of claims", summary)
            self.assertIn("LLM roles", summary)
            attr_md = paths.attribution_md.read_text(encoding="utf-8")
            self.assertIn("teacher_graph", attr_md)
            self.assertIn("predicted:", attr_md)
            cost = json.loads(paths.cost.read_text(encoding="utf-8"))
            self.assertIn("total", cost)
            self.assertFalse(paths.retrieve_ranks.is_file())


class TestGraphProvenanceDoesNotLeakAcrossSamples(unittest.TestCase):
    def test_graph_edge_provenance_keys_by_sample_and_edge_so_shared_ids_do_not_overwrite(self):
        prov = graph_edge_provenance(
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
                },
                {
                    "sample_id": "s2",
                    "session_id": 1,
                    "ops": [
                        {
                            "op": "add_edge",
                            "edge_id": "e0000",
                            "source": "bob",
                            "relationship": "likes",
                            "target": "pizza",
                        }
                    ],
                },
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
                },
                {
                    "sample_id": "s2",
                    "session_id": 1,
                    "relations": [
                        {
                            "source": "bob",
                            "relationship": "likes",
                            "target": "pizza",
                            "proposed_by": ["anthropic"],
                            "votes": 1,
                            "kept": True,
                        }
                    ],
                },
            ],
        )
        self.assertNotIn("e0000", prov)
        self.assertEqual(prov[("s1", "e0000")]["proposed_by"], ["openai"])
        self.assertEqual(prov[("s2", "e0000")]["proposed_by"], ["anthropic"])
        self.assertEqual(prov[("s1", "e0000")]["target"], "painting")
        self.assertEqual(prov[("s2", "e0000")]["target"], "pizza")

    def test_lineage_rows_does_not_copy_proposed_by_from_another_sample_with_the_same_edge_id(self):
        rows = lineage_rows(
            prediction_rows=[
                {"question_id": "q0", "sample_id": "s1"},
                {"question_id": "q1", "sample_id": "s2"},
            ],
            memories_by_sample={
                "s1": Memory(memory_type="fused_teacher_graph", text="p", source_ids=["e0000"]),
                "s2": Memory(memory_type="fused_teacher_graph", text="z", source_ids=["e0000"]),
            },
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
                },
                {
                    "sample_id": "s2",
                    "session_id": 1,
                    "ops": [
                        {
                            "op": "add_edge",
                            "edge_id": "e0000",
                            "source": "bob",
                            "relationship": "likes",
                            "target": "pizza",
                        }
                    ],
                },
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
                            "kept": True,
                        }
                    ],
                },
                {
                    "sample_id": "s2",
                    "session_id": 1,
                    "relations": [
                        {
                            "source": "bob",
                            "relationship": "likes",
                            "target": "pizza",
                            "proposed_by": ["anthropic"],
                            "kept": True,
                        }
                    ],
                },
            ],
        )
        by_q = {row["question_id"]: row for row in rows}
        self.assertEqual(by_q["q0"]["proposed_by"], ["openai"])
        self.assertEqual(by_q["q1"]["proposed_by"], ["anthropic"])
        self.assertEqual(by_q["q0"]["sample_id"], "s1")
        self.assertEqual(by_q["q1"]["sample_id"], "s2")

    def test_lineage_rows_joins_fusion_when_session_id_is_int_on_ingest_and_str_on_fusion(self):
        rows = lineage_rows(
            prediction_rows=[{"question_id": "q0", "sample_id": "s1"}],
            memories_by_sample={
                "s1": Memory(memory_type="teacher_graph", text="p", source_ids=["e0000"]),
            },
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
                    "session_id": "1",
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
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["proposed_by"], ["openai"])


class TestRetrieveLineageExcludesLosersAndSiblingQuestions(unittest.TestCase):
    def test_lineage_rows_omits_unselected_retrieve_candidates(self):
        rows = lineage_rows(
            prediction_rows=[{"question_id": "q0", "sample_id": "s1"}],
            retrieve_ranks=[
                {
                    "sample_id": "s1",
                    "question_id": "q0",
                    "retriever": "rag",
                    "candidates": [
                        {"item_id": "win", "rank": 1, "score": 0.9, "selected": True, "text": "painting"},
                        {"item_id": "lose", "rank": 2, "score": 0.1, "selected": False, "text": "pizza"},
                    ],
                }
            ],
        )
        self.assertEqual([r["item_id"] for r in rows], ["win"])
        self.assertNotIn("lose", [r["item_id"] for r in rows])

    def test_lineage_rows_does_not_use_another_samples_ranks_for_the_same_question_id(self):
        rows = lineage_rows(
            prediction_rows=[
                {"question_id": "q0", "sample_id": "s1"},
                {"question_id": "q0", "sample_id": "s2"},
            ],
            retrieve_ranks=[
                {
                    "sample_id": "s1",
                    "question_id": "q0",
                    "retriever": "rag",
                    "candidates": [
                        {"item_id": "s1-win", "rank": 1, "selected": True, "text": "painting"},
                    ],
                },
                {
                    "sample_id": "s2",
                    "question_id": "q0",
                    "retriever": "rag",
                    "candidates": [
                        {"item_id": "s2-win", "rank": 1, "selected": True, "text": "pizza"},
                    ],
                },
            ],
        )
        by_sample = {row["sample_id"]: row["item_id"] for row in rows}
        self.assertEqual(by_sample["s1"], "s1-win")
        self.assertEqual(by_sample["s2"], "s2-win")

    def test_ranked_candidates_preserves_input_order_when_scores_tie(self):
        rows = ranked_candidates(
            [{"item_id": "first", "score": 0.5}, {"item_id": "second", "score": 0.5}],
            selected_ids={"first"},
        )
        self.assertEqual([r["item_id"] for r in rows], ["first", "second"])
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[1]["rank"], 2)
        self.assertTrue(rows[0]["selected"])
        self.assertFalse(rows[1]["selected"])

    def test_lineage_rows_is_deterministic_for_the_same_inputs(self):
        kwargs = dict(
            prediction_rows=[{"question_id": "q0", "sample_id": "s1"}],
            retrieve_ranks=[
                {
                    "sample_id": "s1",
                    "question_id": "q0",
                    "retriever": "rag",
                    "candidates": [
                        {"item_id": "win", "rank": 1, "selected": True},
                        {"item_id": "lose", "rank": 2, "selected": False},
                    ],
                }
            ],
        )
        self.assertEqual(lineage_rows(**kwargs), lineage_rows(**kwargs))


class TestAttributionDoesNotLeakTeachersSessionsOrQuestions(unittest.TestCase):
    def test_teacher_graph_call_does_not_attach_another_teachers_triple(self):
        calls = attribution_call_rows(
            reader_traces=[],
            teacher_calls=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "teacher_id": "openai",
                    "role": "teacher_graph",
                    "relations": [
                        {"source": "alice", "relationship": "started", "target": "painting"}
                    ],
                },
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "teacher_id": "anthropic",
                    "role": "teacher_graph",
                    "relations": [
                        {"source": "bob", "relationship": "likes", "target": "pizza"}
                    ],
                },
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
                            "kept": True,
                        },
                        {
                            "source": "bob",
                            "relationship": "likes",
                            "target": "pizza",
                            "proposed_by": ["anthropic"],
                            "kept": True,
                        },
                    ],
                }
            ],
            lineage=[
                {
                    "question_id": "q0",
                    "sample_id": "s1",
                    "session_id": 1,
                    "source": "alice",
                    "relationship": "started",
                    "target": "painting",
                    "proposed_by": ["openai"],
                },
                {
                    "question_id": "q1",
                    "sample_id": "s1",
                    "session_id": 1,
                    "source": "bob",
                    "relationship": "likes",
                    "target": "pizza",
                    "proposed_by": ["anthropic"],
                },
            ],
        )
        openai = next(c for c in calls if c["teacher_id"] == "openai")
        anthropic = next(c for c in calls if c["teacher_id"] == "anthropic")
        self.assertEqual([c["target"] for c in openai["claims"]], ["painting"])
        self.assertEqual([c["target"] for c in anthropic["claims"]], ["pizza"])
        self.assertEqual(openai["claims"][0]["injected_question_ids"], ["q0"])
        self.assertEqual(anthropic["claims"][0]["injected_question_ids"], ["q1"])
        self.assertNotIn("q1", openai["claims"][0]["injected_question_ids"])

    def test_teacher_graph_call_does_not_attach_the_same_triple_from_another_session(self):
        calls = attribution_call_rows(
            reader_traces=[],
            teacher_calls=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "teacher_id": "openai",
                    "role": "teacher_graph",
                    "relations": [
                        {"source": "alice", "relationship": "started", "target": "painting"}
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
                            "proposed_by": ["openai"],
                            "kept": True,
                            "votes": 1,
                        }
                    ],
                },
                {
                    "sample_id": "s1",
                    "session_id": 2,
                    "relations": [
                        {
                            "source": "alice",
                            "relationship": "started",
                            "target": "painting",
                            "proposed_by": ["openai"],
                            "kept": False,
                            "votes": 0,
                        }
                    ],
                },
            ],
        )
        claim = calls[0]["claims"][0]
        self.assertTrue(claim["kept"])
        self.assertEqual(claim["votes"], 1)

    def test_canonical_fusion_lookup_joins_despite_casing_and_does_not_use_a_raw_mismatch(self):
        calls = attribution_call_rows(
            reader_traces=[],
            teacher_calls=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "teacher_id": "openai",
                    "role": "teacher_graph",
                    "relations": [
                        {"source": "Alice", "relationship": "Started", "target": "Painting"}
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
                            "proposed_by": ["openai"],
                            "kept": True,
                            "votes": 2,
                        }
                    ],
                }
            ],
        )
        claim = calls[0]["claims"][0]
        self.assertTrue(claim["kept"])
        self.assertEqual(claim["votes"], 2)
        self.assertEqual(claim["source"], "alice")

    def test_reader_used_memory_items_do_not_include_another_questions_lineage(self):
        calls = attribution_call_rows(
            reader_traces=[
                {
                    "sample_id": "s1",
                    "question_id": "q0",
                    "role": "reader",
                    "predicted_answer": "painting",
                }
            ],
            teacher_calls=[],
            lineage=[
                {
                    "question_id": "q0",
                    "sample_id": "s1",
                    "item_id": "e0000",
                    "proposed_by": ["openai"],
                },
                {
                    "question_id": "q1",
                    "sample_id": "s1",
                    "item_id": "e0001",
                    "proposed_by": ["anthropic"],
                },
            ],
        )
        used = calls[0]["used_memory_items"]
        self.assertEqual([u["item_id"] for u in used], ["e0000"])
        self.assertEqual(used[0]["proposed_by"], ["openai"])

    def test_reader_used_memory_items_do_not_include_another_sample_with_the_same_question_id(self):
        calls = attribution_call_rows(
            reader_traces=[
                {
                    "sample_id": "s1",
                    "question_id": "q0",
                    "role": "reader",
                    "predicted_answer": "painting",
                }
            ],
            teacher_calls=[],
            lineage=[
                {
                    "question_id": "q0",
                    "sample_id": "s1",
                    "item_id": "s1-edge",
                    "proposed_by": ["openai"],
                },
                {
                    "question_id": "q0",
                    "sample_id": "s2",
                    "item_id": "s2-edge",
                    "proposed_by": ["anthropic"],
                },
            ],
        )
        used = calls[0]["used_memory_items"]
        self.assertEqual([u["item_id"] for u in used], ["s1-edge"])

    def test_attribution_call_ids_are_deterministic_for_the_same_inputs(self):
        kwargs = dict(
            reader_traces=[
                {"sample_id": "s1", "question_id": "q0", "role": "reader", "predicted_answer": "a"}
            ],
            teacher_calls=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "teacher_id": "openai",
                    "role": "teacher_graph",
                    "relations": [
                        {"source": "alice", "relationship": "started", "target": "painting"}
                    ],
                }
            ],
        )
        first = attribution_call_rows(**kwargs)
        second = attribution_call_rows(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["call_id"], "teacher_graph:openai:s1:1:0")
        self.assertEqual(first[1]["call_id"], "reader:q0:0")


class TestCostAndQualityBucketsDoNotMixRoles(unittest.TestCase):
    def test_cost_rollup_does_not_add_teacher_tokens_to_the_reader_bucket(self):
        cost = cost_rollup(
            reader_traces=[
                {
                    "model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                }
            ],
            teacher_calls=[
                {
                    "model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050},
                }
            ],
        )
        self.assertEqual(cost["reader"]["total_tokens"], 110)
        self.assertEqual(cost["teacher"]["total_tokens"], 1050)
        self.assertEqual(cost["total"]["total_tokens"], 1160)
        self.assertNotEqual(cost["reader"]["usd"], cost["teacher"]["usd"])

    def test_quality_keep_counts_do_not_credit_a_teacher_who_did_not_propose_the_triple(self):
        stats = teacher_quality_stats(
            calls=[
                {"teacher_id": "openai", "parse": "ok", "n_relations": 1},
                {"teacher_id": "anthropic", "parse": "ok", "n_relations": 1},
            ],
            fusion_rows=[
                {
                    "teacher_ids": ["openai", "anthropic"],
                    "relations": [
                        {"proposed_by": ["openai"], "kept": True},
                        {"proposed_by": ["anthropic"], "kept": False},
                    ],
                }
            ],
        )
        self.assertEqual(stats["by_teacher"]["openai"]["n_proposed_kept"], 1)
        self.assertEqual(stats["by_teacher"]["anthropic"]["n_proposed_kept"], 0)
        self.assertEqual(stats["fusion"]["n_triples_kept"], 1)
        self.assertEqual(stats["fusion"]["n_triples_discarded"], 1)


class TestOptionalLineageInputsDoNotInventOrLeak(unittest.TestCase):
    def test_lineage_rows_returns_empty_when_optional_memory_ranks_and_ingest_are_omitted(self):
        rows = lineage_rows(prediction_rows=[{"question_id": "q0", "sample_id": "s1"}])
        self.assertEqual(rows, [])

    def test_lineage_rows_prefers_per_question_memory_when_sample_memory_is_also_present(self):
        rows = lineage_rows(
            prediction_rows=[{"question_id": "q0", "sample_id": "s1"}],
            memories_by_question={
                "q0": Memory(memory_type="rag", text="q", source_ids=["from-q"]),
            },
            memories_by_sample={
                "s1": Memory(memory_type="rag", text="s", source_ids=["from-s"]),
            },
        )
        self.assertEqual([r["item_id"] for r in rows], ["from-q"])

    def test_lineage_rows_joins_rank_row_when_sample_id_is_omitted_on_the_rank_dump(self):
        rows = lineage_rows(
            prediction_rows=[{"question_id": "q0", "sample_id": "s1"}],
            retrieve_ranks=[
                {
                    "question_id": "q0",
                    "retriever": "rag",
                    "candidates": [{"item_id": "win", "selected": True}],
                }
            ],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["item_id"], "win")
        self.assertEqual(rows[0]["sample_id"], "s1")

    def test_lineage_rows_unlabeled_rank_row_does_not_override_a_sample_keyed_row(self):
        rows = lineage_rows(
            prediction_rows=[
                {"question_id": "q0", "sample_id": "s1"},
                {"question_id": "q0", "sample_id": "s2"},
            ],
            retrieve_ranks=[
                {
                    "question_id": "q0",
                    "candidates": [{"item_id": "unlabeled", "selected": True}],
                },
                {
                    "sample_id": "s1",
                    "question_id": "q0",
                    "candidates": [{"item_id": "s1-win", "selected": True}],
                },
            ],
        )
        by_sample = {row["sample_id"]: row["item_id"] for row in rows}
        self.assertEqual(by_sample["s1"], "s1-win")
        self.assertEqual(by_sample["s2"], "unlabeled")

    def test_lineage_rows_leaves_proposed_by_empty_when_optional_fusion_rows_are_omitted(self):
        rows = lineage_rows(
            prediction_rows=[{"question_id": "q0", "sample_id": "s1"}],
            memories_by_sample={
                "s1": Memory(memory_type="teacher_graph", text="p", source_ids=["e0000"]),
            },
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
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["proposed_by"], [])
        self.assertEqual(rows[0]["item_id"], "e0000")

    def test_graph_edge_provenance_skips_ops_without_edge_id_and_non_add_edge_ops(self):
        prov = graph_edge_provenance(
            ingest_rows=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "ops": [
                        {"op": "skip_dup", "edge_id": "e0000", "source": "alice", "relationship": "started", "target": "painting"},
                        {"op": "add_edge", "edge_id": "", "source": "bob", "relationship": "likes", "target": "pizza"},
                        {
                            "op": "add_edge",
                            "edge_id": "e0001",
                            "source": "alice",
                            "relationship": "started",
                            "target": "painting",
                        },
                    ],
                }
            ],
            fusion_rows=[],
        )
        self.assertEqual(set(prov), {("s1", "e0001")})

    def test_graph_edge_provenance_does_not_treat_omitted_kept_on_a_fusion_row_as_true(self):
        prov = graph_edge_provenance(
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
                            "proposed_by": ["openai"],
                        }
                    ],
                }
            ],
        )
        self.assertIsNone(prov[("s1", "e0000")]["kept"])
        self.assertEqual(prov[("s1", "e0000")]["proposed_by"], ["openai"])


class TestOptionalAttributionFieldsDoNotCrashOrLeak(unittest.TestCase):
    def test_attribution_call_rows_accepts_omitted_fusion_and_lineage(self):
        calls = attribution_call_rows(
            reader_traces=[{"question_id": "q0", "predicted_answer": "a", "sample_id": "s1"}],
            teacher_calls=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "teacher_id": "openai",
                    "role": "teacher_graph",
                    "relations": [
                        {"source": "alice", "relationship": "started", "target": "painting"}
                    ],
                }
            ],
        )
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["claims"][0]["kept"])
        self.assertEqual(calls[0]["claims"][0]["injected_question_ids"], [])
        self.assertEqual(calls[1]["used_memory_items"], [])

    def test_teacher_graph_call_falls_back_to_fusion_when_relations_list_is_omitted(self):
        calls = attribution_call_rows(
            reader_traces=[],
            teacher_calls=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "teacher_id": "openai",
                    "role": "teacher_graph",
                    "n_relations": 1,
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
                            "kept": True,
                        },
                        {
                            "source": "bob",
                            "relationship": "likes",
                            "target": "pizza",
                            "proposed_by": ["anthropic"],
                            "kept": True,
                        },
                    ],
                }
            ],
        )
        self.assertEqual(len(calls[0]["claims"]), 1)
        self.assertEqual(calls[0]["claims"][0]["target"], "painting")
        self.assertNotIn("pizza", [c["target"] for c in calls[0]["claims"]])

    def test_session_summary_call_omits_claim_when_output_text_is_empty(self):
        calls = attribution_call_rows(
            reader_traces=[],
            teacher_calls=[
                {
                    "sample_id": "s1",
                    "session_id": 1,
                    "teacher_id": "openai",
                    "role": "teacher",
                    "output_text": "  ",
                }
            ],
        )
        self.assertEqual(calls[0]["claims"], [])
        self.assertEqual(calls[0]["n_claims"], 0)

    def test_reader_still_joins_lineage_when_sample_id_is_omitted_on_the_lineage_row(self):
        calls = attribution_call_rows(
            reader_traces=[
                {"sample_id": "s1", "question_id": "q0", "role": "reader", "predicted_answer": "a"}
            ],
            teacher_calls=[],
            lineage=[
                {"question_id": "q0", "item_id": "legacy", "proposed_by": ["openai"]},
            ],
        )
        self.assertEqual(calls[0]["used_memory_items"][0]["item_id"], "legacy")

    def test_infer_llm_role_prefers_logged_role_when_present(self):
        self.assertEqual(infer_llm_role({"role": "reader", "relations": [{}]}), "reader")

    def test_infer_llm_role_treats_relations_as_teacher_graph_when_role_is_omitted(self):
        self.assertEqual(
            infer_llm_role({"relations": [{"source": "a", "relationship": "r", "target": "b"}]}),
            "teacher_graph",
        )

    def test_infer_llm_role_treats_question_id_as_reader_when_role_is_omitted(self):
        self.assertEqual(infer_llm_role({"question_id": "q0"}), "reader")

    def test_infer_llm_role_defaults_to_teacher_when_no_role_fields_are_present(self):
        self.assertEqual(infer_llm_role({}), "teacher")

    def test_cost_rollup_treats_omitted_usage_as_zero_and_unknown_model_tokens_as_unpriced(self):
        cost = cost_rollup(
            reader_traces=[{"model": "mock"}],
            teacher_calls=[
                {
                    "model": "claude-haiku-4-5",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                }
            ],
        )
        self.assertEqual(cost["reader"]["total_tokens"], 0)
        self.assertEqual(cost["reader"]["usd"], 0.0)
        self.assertEqual(cost["teacher"]["total_tokens"], 11)
        self.assertIsNone(cost["teacher"]["usd"])

    def test_cost_rollup_reads_anthropic_token_keys_when_openai_keys_are_omitted(self):
        cost = cost_rollup(
            reader_traces=[
                {
                    "model": "gpt-4o-mini",
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
                }
            ],
            teacher_calls=[],
        )
        self.assertEqual(cost["reader"]["prompt_tokens"], 1_000_000)
        self.assertEqual(cost["reader"]["usd"], 0.15)


if __name__ == "__main__":
    unittest.main()
