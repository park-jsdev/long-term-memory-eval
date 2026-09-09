"""Evaluation-pipeline tests (HLD component: evaluation).

HLD components:
  i)   pre-processing — tests/test_preprocessing_pipeline.py (not wired into run.py)
  ii)  teacher orchestrator — pooling/fusion in teacher_orchestrator.py (passthrough if no teachers)
  iii) post-processing — not this file
  iv)  evaluation  ← this file (string metrics + LoCoMo F1)

Online / LLM autoraters are out of scope until that split is designed.

Parse and memory-builder checks remain in this module as the frozen *inputs*
to the scorer, including a raw_chunks vs session_summaries prompt-text
check (different memories must fill different reader prompts).
Dedicated pre-processing / teacher-memory test modules wait on HLD lock-in —
do not rename production classes to match an unfinished LLD.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.dataset import iter_questions, parse_sample, select_question_pairs
from src.locomo_eval.schemas import Conversation, Question
from src.locomo_eval.memory import (
    RawConversationMemoryBuilder,
    SessionSummaryMemoryBuilder,
    get_memory_builder,
    resolve_memory_name,
)
from src.locomo_eval.metrics import exact_match, normalize_answer, token_f1
from src.locomo_eval.prompts import load_prompt_template, render_qa_prompt
from src.metrics.locomo_qa import score_prediction


MINI = {
    "sample_id": "conv-test",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice!"},
        ],
        "session_2_date_time": "2 Jan 2023",
        "session_2": [
            {"dia_id": "D2:1", "speaker": "Alice", "text": "I got a nursing job."},
        ],
    },
    "session_summary": {
        "session_1_summary": "Alice said she started painting.",
        "session_2_summary": "Alice got a nursing job.",
    },
    "observation": {},
    "qa": [
        {
            "question": "What hobby did Alice begin?",
            "answer": "painting",
            "category": 4,
            "evidence": ["D1:1"],
        },
        {
            "question": "When did this happen?",
            "answer": "last June",
            "category": 2,
            "evidence": [],
        },
    ],
}


class TestParseSample(unittest.TestCase):
    def test_parse_sample_sets_sample_id_and_speakers_from_json(self):
        conv = parse_sample(MINI)
        self.assertEqual(conv.sample_id, "conv-test")
        self.assertEqual(conv.speaker_a, "Alice")
        self.assertEqual(conv.speaker_b, "Bob")

    def test_parse_sample_creates_sessions_in_id_order_with_turns(self):
        conv = parse_sample(MINI)
        self.assertEqual(len(conv.sessions), 2)
        self.assertEqual(conv.sessions[0].session_id, 1)
        self.assertEqual(len(conv.sessions[0].turns), 2)
        self.assertEqual(conv.sessions[1].session_id, 2)

    def test_parse_sample_assigns_question_ids_from_sample_id_and_index(self):
        conv = parse_sample(MINI)
        self.assertEqual(len(conv.questions), 2)
        self.assertEqual(conv.questions[0].question_id, "conv-test-q-0")
        self.assertEqual(conv.questions[1].question_id, "conv-test-q-1")

    def test_parse_sample_indexes_session_summaries_by_session_id(self):
        conv = parse_sample(MINI)
        self.assertIn(1, conv.session_summaries)
        self.assertIn("painting", conv.session_summaries[1])


class TestSelectQuestionPairs(unittest.TestCase):
    def _pairs(self):
        convs = []
        for sid, n_q in (("c1", 3), ("c2", 3)):
            questions = [
                Question(
                    sample_id=sid,
                    question_id=f"{sid}-q-{i}",
                    question=f"q{i}",
                    answer="a",
                    category=4,
                )
                for i in range(n_q)
            ]
            convs.append(
                Conversation(
                    sample_id=sid,
                    speaker_a="A",
                    speaker_b="B",
                    sessions=[],
                    session_summaries={},
                    observations={},
                    questions=questions,
                )
            )
        return list(iter_questions(convs))

    def test_round_robin_takes_one_question_from_each_conversation_before_a_second(self):
        pairs = self._pairs()
        got = select_question_pairs(pairs, 2, mode="round_robin")
        self.assertEqual(
            [q.question_id for _, q in got],
            ["c1-q-0", "c2-q-0"],
        )
        got4 = select_question_pairs(pairs, 4, mode="round_robin")
        self.assertEqual(
            [q.question_id for _, q in got4],
            ["c1-q-0", "c2-q-0", "c1-q-1", "c2-q-1"],
        )

    def test_prefix_takes_the_first_n_pairs_in_file_order(self):
        pairs = self._pairs()
        got = select_question_pairs(pairs, 2, mode="prefix")
        self.assertEqual(
            [q.question_id for _, q in got],
            ["c1-q-0", "c1-q-1"],
        )


class TestSessionSummaryMemoryBuilder(unittest.TestCase):
    def test_build_sets_memory_type_to_session_summaries(self):
        conv = parse_sample(MINI)
        mem = SessionSummaryMemoryBuilder().build(conv, conv.questions[0])
        self.assertEqual(mem.memory_type, "session_summaries")

    def test_build_concatenates_summaries_in_session_order(self):
        conv = parse_sample(MINI)
        mem = SessionSummaryMemoryBuilder().build(conv, conv.questions[0])
        self.assertIn("Session 1", mem.text)
        self.assertIn("painting", mem.text)
        self.assertIn("nursing", mem.text)
        self.assertLess(mem.text.index("painting"), mem.text.index("nursing"))

    def test_build_records_summary_source_ids(self):
        conv = parse_sample(MINI)
        mem = SessionSummaryMemoryBuilder().build(conv, conv.questions[0])
        self.assertEqual(mem.source_ids, ["session_1_summary", "session_2_summary"])


class TestRawConversationMemoryBuilder(unittest.TestCase):
    def test_build_sets_memory_type_to_raw_chunks(self):
        conv = parse_sample(MINI)
        mem = RawConversationMemoryBuilder().build(conv, conv.questions[0])
        self.assertEqual(mem.memory_type, "raw_chunks")

    def test_build_includes_turn_text_session_headers_and_dia_ids(self):
        conv = parse_sample(MINI)
        mem = RawConversationMemoryBuilder().build(conv, conv.questions[0])
        self.assertIn("I started painting", mem.text)
        self.assertIn("SESSION 1", mem.text)
        self.assertIn("D1:1", mem.text)

    def test_build_truncates_from_the_start_and_marks_truncated(self):
        conv = parse_sample(MINI)
        mem = RawConversationMemoryBuilder(max_chars=40).build(conv, conv.questions[0])
        self.assertIn("truncated", mem.text.lower())
        self.assertLessEqual(len(mem.text), 40 + len("[... earlier turns truncated to max_chars ...]\n"))


class TestRawChunksAndSessionSummariesFillDifferentReaderPrompts(unittest.TestCase):
    """Different conditions → different memory → different filled reader prompts."""

    def setUp(self):
        conv = parse_sample(MINI)
        self.question = conv.questions[0]
        self.mem_raw = RawConversationMemoryBuilder().build(conv, self.question)
        self.mem_summaries = SessionSummaryMemoryBuilder().build(conv, self.question)
        _, self.template = load_prompt_template(ROOT / "prompts" / "qa_v1.txt")

    def test_raw_chunks_and_session_summaries_produce_different_memory_text_for_the_same_question(self):
        self.assertNotEqual(self.mem_raw.text, self.mem_summaries.text)
        self.assertIn("I started painting", self.mem_raw.text)
        self.assertIn("Alice said she started painting", self.mem_summaries.text)

    def test_raw_chunks_and_session_summaries_render_different_reader_prompts(self):
        prompt_raw = render_qa_prompt(self.template, self.mem_raw.text, self.question.question)
        prompt_summaries = render_qa_prompt(
            self.template, self.mem_summaries.text, self.question.question
        )
        self.assertNotEqual(prompt_raw, prompt_summaries)


class TestResolveMemoryName(unittest.TestCase):
    def test_resolve_memory_name_maps_legacy_c0_alias_to_raw_chunks(self):
        self.assertEqual(resolve_memory_name("c0"), "raw_chunks")

    def test_resolve_memory_name_maps_session_summary_alias_to_session_summaries(self):
        self.assertEqual(resolve_memory_name("session_summary"), "session_summaries")

    def test_resolve_memory_name_raises_for_unknown_builder(self):
        with self.assertRaises(ValueError):
            resolve_memory_name("not_a_builder")


class TestGetMemoryBuilder(unittest.TestCase):
    def test_get_memory_builder_session_summary_alias_returns_session_summaries_builder(self):
        builder = get_memory_builder("session_summary")
        self.assertEqual(builder.name, "session_summaries")

    def test_get_memory_builder_legacy_c0_alias_returns_raw_chunks_builder(self):
        builder = get_memory_builder("c0")
        self.assertEqual(builder.name, "raw_chunks")


class TestNormalizeAnswer(unittest.TestCase):
    def test_normalize_answer_lowercases_and_strips_articles_and_punctuation(self):
        self.assertEqual(normalize_answer("The Cat!"), "cat")


class TestExactMatch(unittest.TestCase):
    def test_exact_match_returns_one_when_answers_match_after_normalization(self):
        self.assertEqual(exact_match("The cat", "cat"), 1.0)

    def test_exact_match_returns_zero_when_normalized_answers_differ(self):
        self.assertEqual(exact_match("painting", "nursing"), 0.0)


class TestTokenF1(unittest.TestCase):
    def test_token_f1_is_greater_than_half_when_most_tokens_overlap(self):
        self.assertGreater(token_f1("red blue cat", "blue cat"), 0.5)

    def test_token_f1_returns_one_when_normalized_token_bags_match(self):
        self.assertEqual(token_f1("The cat", "cat"), 1.0)

    def test_token_f1_returns_zero_when_tokens_do_not_overlap(self):
        self.assertEqual(token_f1("red", "blue"), 0.0)


class TestScorePrediction(unittest.TestCase):
    def test_score_prediction_returns_one_for_equivalent_temporal_dates(self):
        self.assertEqual(score_prediction("May 7 2023", "7 May 2023", 2), 1.0)

    def test_score_prediction_returns_one_for_adversarial_not_mentioned(self):
        self.assertEqual(score_prediction("not mentioned", "anything", 5), 1.0)


class TestPairPredictions(unittest.TestCase):
    def test_pair_predictions_joins_on_question_id_and_attaches_locomo_f1_delta(self):
        from scripts.analysis.compare_predictions import pair_predictions

        rows_a = [
            {
                "question_id": "q0",
                "category": 4,
                "question": "hobby?",
                "predicted_answer": "painting",
                "reference_answer": "painting",
            },
            {
                "question_id": "q_only_a",
                "category": 4,
                "question": "skip",
                "predicted_answer": "x",
                "reference_answer": "x",
            },
        ]
        rows_b = [
            {
                "question_id": "q0",
                "category": 4,
                "question": "hobby?",
                "predicted_answer": "nursing",
                "reference_answer": "painting",
            }
        ]
        paired = pair_predictions(rows_a, rows_b)
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["question_id"], "q0")
        self.assertEqual(paired[0]["locomo_f1_a"], 1.0)
        self.assertLess(paired[0]["locomo_f1_b"], 1.0)
        self.assertEqual(
            paired[0]["delta_locomo_f1_b_minus_a"],
            round(paired[0]["locomo_f1_b"] - paired[0]["locomo_f1_a"], 6),
        )

    def test_pair_predictions_returns_empty_when_question_ids_do_not_overlap(self):
        from scripts.analysis.compare_predictions import pair_predictions

        paired = pair_predictions(
            [{"question_id": "a", "predicted_answer": "x", "reference_answer": "x", "category": 4}],
            [{"question_id": "b", "predicted_answer": "x", "reference_answer": "x", "category": 4}],
        )
        self.assertEqual(paired, [])


class TestResolvePredictionsJsonl(unittest.TestCase):
    def test_load_prediction_rows_reads_jsonl_from_a_run_directory(self):
        from scripts.analysis.compare_predictions import load_prediction_rows

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "cmp_raw_chunks"
            run.mkdir()
            row = {
                "question_id": "q0",
                "predicted_answer": "painting",
                "reference_answer": "painting",
                "category": 4,
            }
            (run / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = load_prediction_rows(run)
            self.assertEqual(loaded[0]["question_id"], "q0")

    def test_resolve_predictions_jsonl_raises_when_run_id_does_not_exist(self):
        from scripts.analysis.compare_predictions import resolve_predictions_jsonl

        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_predictions_jsonl("definitely_missing_run_xyz")
        msg = str(ctx.exception)
        self.assertIn("No predictions found", msg)
        self.assertIn("--run-id", msg)


class TestWriteLocomoF1ComparePlots(unittest.TestCase):
    def test_write_compare_prediction_plots_writes_boxplot_and_histogram_pngs(self):
        from scripts.analysis.compare_predictions import write_compare_prediction_plots

        paired = [
            {"locomo_f1_a": 1.0, "locomo_f1_b": 0.0},
            {"locomo_f1_a": 0.5, "locomo_f1_b": 0.5},
            {"locomo_f1_a": 0.0, "locomo_f1_b": 1.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            written = write_compare_prediction_plots(
                paired, label_a="raw_chunks", label_b="session_summaries", out_dir=Path(tmp)
            )
            if not written:
                self.skipTest("matplotlib not installed")
            names = {p.name for p in written}
            self.assertIn("locomo_f1_overall.png", names)
            self.assertIn("locomo_f1_boxplot.png", names)
            self.assertIn("locomo_f1_histograms.png", names)
            for p in written:
                self.assertGreater(p.stat().st_size, 0)


class TestRealLocomoIfPresent(unittest.TestCase):
    def setUp(self):
        path = ROOT / "data" / "raw" / "locomo10.json"
        if not path.exists():
            self.skipTest("locomo10.json not fetched")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.conv = parse_sample(data[0])

    def test_parse_sample_loads_at_least_one_question_from_real_locomo(self):
        self.assertTrue(self.conv.questions)

    def test_session_summaries_build_produces_memory_longer_than_100_chars_on_real_locomo(self):
        mem = SessionSummaryMemoryBuilder().build(self.conv, self.conv.questions[0])
        self.assertGreater(len(mem.text), 100)

    def test_raw_chunks_build_produces_memory_longer_than_100_chars_on_real_locomo(self):
        mem = RawConversationMemoryBuilder().build(self.conv, self.conv.questions[0])
        self.assertGreater(len(mem.text), 100)


if __name__ == "__main__":
    unittest.main()
