"""Session-document join + naive retrieval sanity (HLD i preprocess).

SessionBlock stays gold-free. Gold is only on the *joined* QA table for audit
and oracle checks. Retrieval queries are the question text only.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.locomo import CATEGORY_NAMES as DATA_CATEGORY_NAMES
from src.locomo_eval.preprocess import PreprocessingPipeline
from src.locomo_eval.preprocess.data_ingestor import DataIngestor
from src.locomo_eval.preprocess.session_documents import (
    block_matches_raw_session,
    build_session_documents,
    count_raw_sessions,
    evidence_tokens,
    join_qa_to_documents,
    naive_rank_session_ids,
    parse_dia_id,
    recall_evidence_sessions,
)
from src.metrics.locomo_qa import CATEGORY_NAMES as EVAL_CATEGORY_NAMES
from src.metrics.locomo_qa import score_prediction
from src.locomo_eval.preprocess.dataset_stats import write_dataset_analysis
from src.locomo_eval.preprocess.export_session_tables import export_tables

GOLD_LOCK = "UNIQ_GOLD_REF_ZZZ"
GOLD_JOB = "UNIQ_GOLD_JOB_ZZZ"

RICH = {
    "sample_id": "conv-docs",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {
                "dia_id": "D1:1",
                "speaker": "Alice",
                "text": "I started painting landscapes.",
                "blip_caption": "a paintbrush",
                "img_url": "http://example.test/brush.png",
            },
            {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice hobby!"},
        ],
        "session_2_date_time": "2 Jan 2023",
        "session_2": [],
        "session_3_date_time": "1:56 pm on 8 May, 2023",
        "session_3": [
            {"dia_id": "D3:1", "speaker": "Alice", "text": "I got a nursing job at Northside."},
        ],
    },
    "session_summary": {
        "session_1_summary": "Alice said she started painting landscapes.",
        "session_3_summary": "Alice got a nursing job at Northside.",
    },
    "observation": {
        "session_1_observation": {
            "Alice": [["Alice started painting landscapes.", "D1:1"]],
            "Bob": [],
        },
        "session_3_observation": {
            "Alice": [["Alice got a nursing job.", "D3:1"]],
            "Bob": [],
        },
    },
    "event_summary": {
        "events_session_1": {
            "Alice": ["Alice starts painting."],
            "Bob": [],
            "date": "1 Jan 2023",
        },
        "events_session_3": {
            "Alice": ["Alice starts a nursing job."],
            "Bob": [],
            "date": "8 May, 2023",
        },
    },
    "qa": [
        {
            "question": "What hobby did Alice begin painting?",
            "answer": GOLD_LOCK,
            "category": 4,
            "evidence": ["D1:1"],
        },
        {
            "question": "What hobby and nursing job did Alice mention?",
            "answer": GOLD_JOB,
            "category": 1,
            "evidence": ["D1:1", "D3:1"],
        },
    ],
}


def _docs():
    return build_session_documents(RICH)


class TestOfficialCategoryIdsMatchTaskEvalNotPaperProse(unittest.TestCase):
    def test_data_and_scorer_category_names_use_the_same_official_ids(self):
        self.assertEqual(DATA_CATEGORY_NAMES, EVAL_CATEGORY_NAMES)
        self.assertEqual(DATA_CATEGORY_NAMES[1], "multi_hop")
        self.assertEqual(DATA_CATEGORY_NAMES[4], "single_hop")
        self.assertEqual(DATA_CATEGORY_NAMES[5], "adversarial")

    def test_score_prediction_uses_adversarial_rule_only_for_category_five(self):
        self.assertEqual(score_prediction("not mentioned", "anything", 5), 1.0)
        self.assertLess(score_prediction("not mentioned", GOLD_LOCK, 4), 1.0)


class TestDatasetIsProcessedIntoSessionDocuments(unittest.TestCase):
    def test_build_session_documents_emits_one_doc_per_nonempty_raw_session(self):
        docs = _docs()
        self.assertEqual(count_raw_sessions(RICH), 3)
        self.assertEqual([d.session_id for d in docs], [1, 3])
        self.assertEqual(len(docs), 2)


class TestEachBlockJoinsTheCorrectSupplementFields(unittest.TestCase):
    def test_session_one_joins_summary_observation_event_and_image_from_raw(self):
        doc = _docs()[0]
        self.assertIn("painting landscapes", doc.session_summary)
        self.assertEqual(doc.observations[0]["source_dia_id"], "D1:1")
        self.assertEqual(doc.events_a, ["Alice starts painting."])
        self.assertEqual(doc.n_images, 1)
        self.assertEqual(doc.event_date, "1 Jan 2023")


class TestEachBlockContainsTheCorrectRawTurns(unittest.TestCase):
    def test_processed_block_turns_match_raw_session_list(self):
        conv = DataIngestor().ingest_sample(RICH)
        processed = PreprocessingPipeline().process(conv)
        raw = RICH["conversation"]["session_1"]
        self.assertTrue(block_matches_raw_session(processed.session_blocks[0], raw))
        self.assertEqual(processed.session_blocks[0].turns[0].text, raw[0]["text"])


class TestSingleHopRetrievalOnJoinedGold(unittest.TestCase):
    def test_oracle_join_finds_single_hop_evidence_turn_in_its_session(self):
        docs = _docs()
        hops = [r for r in join_qa_to_documents(RICH, docs) if r.category == 4]
        self.assertEqual(len(hops), 1)
        row = hops[0]
        self.assertEqual(row.answer, GOLD_LOCK)
        self.assertEqual(row.evidence_session_ids, [1])
        self.assertTrue(row.all_evidence_in_blocks)
        self.assertIn("D1:1", docs[0].dia_ids)

    def test_naive_turn_retrieval_ranks_single_hop_evidence_session_first(self):
        docs = _docs()
        row = next(r for r in join_qa_to_documents(RICH, docs) if r.category == 4)
        ranked = naive_rank_session_ids(row.question, docs, "turns")
        self.assertEqual(ranked[0], 1)
        self.assertEqual(recall_evidence_sessions(row.evidence_session_ids, ranked, k=1), 1.0)
        self.assertNotIn(GOLD_LOCK, row.question)


class TestMultiHopRetrievalOnJoinedGold(unittest.TestCase):
    def test_oracle_join_maps_multi_hop_evidence_to_two_session_documents(self):
        docs = _docs()
        row = next(r for r in join_qa_to_documents(RICH, docs) if r.category == 1)
        self.assertEqual(row.evidence_session_ids, [1, 3])
        self.assertTrue(row.all_evidence_in_blocks)
        self.assertEqual(row.n_evidence_sessions, 2)

    def test_naive_turn_retrieval_recalls_both_multi_hop_sessions_at_k_two(self):
        docs = _docs()
        row = next(r for r in join_qa_to_documents(RICH, docs) if r.category == 1)
        ranked = naive_rank_session_ids(row.question, docs, "turns")
        self.assertEqual(recall_evidence_sessions(row.evidence_session_ids, ranked, k=2), 1.0)


class TestNaiveAgentAndMemoryRetrievalOnBlocks(unittest.TestCase):
    def test_naive_agent_query_is_the_question_and_blocks_do_not_contain_gold(self):
        docs = _docs()
        blob = " ".join(d.turns_text() + d.session_summary for d in docs)
        self.assertNotIn(GOLD_LOCK, blob)
        row = next(r for r in join_qa_to_documents(RICH, docs) if r.category == 4)
        ranked = naive_rank_session_ids(row.question, docs, "turns")
        self.assertEqual(ranked[0], 1)
        self.assertNotIn(GOLD_LOCK, row.question)

    def test_naive_memory_summary_field_ranks_session_one_for_painting_question(self):
        docs = _docs()
        ranked = naive_rank_session_ids("What hobby did Alice begin painting?", docs, "summary")
        self.assertEqual(ranked[0], 1)


class TestExportAndVisualizationReports(unittest.TestCase):
    def test_export_tables_and_plots_write_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            counts = export_tables([RICH], out)
            self.assertEqual(counts["n_sessions"], 2)
            self.assertEqual(counts["n_qa"], 2)
            self.assertEqual(counts["n_images"], 1)
            qa_csv = (out / "qa_joined.csv").read_text(encoding="utf-8")
            self.assertIn(GOLD_LOCK, qa_csv)
            session_csv = (out / "session_documents.csv").read_text(encoding="utf-8")
            self.assertNotIn(GOLD_LOCK, session_csv)
            plots = write_dataset_analysis([RICH], out)
            self.assertTrue((plots / "turns_per_session.png").is_file())
            self.assertTrue((plots / "naive_retrieval_recall.png").is_file())
            stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["naive_agent_turns"]["field"], "turns")
            self.assertEqual(stats["naive_memory_summary"]["field"], "summary")


class TestParseDiaId(unittest.TestCase):
    def test_parse_dia_id_returns_session_and_turn_token(self):
        self.assertEqual(parse_dia_id("D3:1"), (3, "1"))
        self.assertEqual(parse_dia_id("D30:05"), (30, "5"))
        self.assertEqual(parse_dia_id("D:11:26"), (11, "26"))
        self.assertIsNone(parse_dia_id("not-an-id"))

    def test_evidence_tokens_splits_semicolon_and_space_packed_ids(self):
        self.assertEqual(evidence_tokens(["D8:6; D9:17"]), ["D8:6", "D9:17"])
        self.assertEqual(evidence_tokens(["D9:1 D4:4"]), ["D9:1", "D4:4"])


class TestRealLocomoSessionDocumentsIfPresent(unittest.TestCase):
    def setUp(self):
        path = ROOT / "data" / "raw" / "locomo10.json"
        if not path.is_file():
            self.skipTest("locomo10.json not fetched")
        self.samples = json.loads(path.read_text(encoding="utf-8"))

    def test_every_nonempty_raw_session_becomes_a_session_document(self):
        n_docs = 0
        n_nonempty = 0
        for sample in self.samples:
            conv = sample.get("conversation") or {}
            n_nonempty += sum(
                1
                for k, v in conv.items()
                if k.startswith("session_")
                and "date_time" not in k
                and isinstance(v, list)
                and v
            )
            n_docs += len(build_session_documents(sample))
        self.assertEqual(n_docs, n_nonempty)
        self.assertGreater(n_docs, 100)

    def test_blocks_contain_every_evidence_turn_that_exists_in_raw_dialog(self):
        from src.locomo_eval.preprocess.session_documents import (
            canonical_dia_id,
            evidence_tokens,
        )

        n_checked = 0
        missing_in_blocks: list[str] = []
        dangling: list[str] = []
        for sample in self.samples:
            conv = sample.get("conversation") or {}
            raw_dias = set()
            for key, turns in conv.items():
                if not (key.startswith("session_") and "date_time" not in key):
                    continue
                if not isinstance(turns, list):
                    continue
                for t in turns:
                    if isinstance(t, dict) and t.get("dia_id"):
                        raw_dias.add(str(t["dia_id"]))
                        canon = canonical_dia_id(str(t["dia_id"]))
                        if canon:
                            raw_dias.add(canon)
            docs = build_session_documents(sample)
            block_dias = set()
            for doc in docs:
                block_dias.update(doc.dia_ids)
                block_dias.update(filter(None, (canonical_dia_id(x) for x in doc.dia_ids)))
            for row in join_qa_to_documents(sample, docs):
                for tok in evidence_tokens(row.evidence):
                    canon = canonical_dia_id(tok) or tok
                    if tok in raw_dias or canon in raw_dias:
                        n_checked += 1
                        if tok not in block_dias and canon not in block_dias:
                            missing_in_blocks.append(f"{row.question_id}:{tok}")
                    else:
                        dangling.append(f"{row.question_id}:{tok}")
        self.assertEqual(missing_in_blocks, [])
        self.assertGreater(n_checked, 1000)
        # locomo10 has a couple of evidence ids with no matching turn (annotation noise).
        self.assertLessEqual(len(dangling), 5)
