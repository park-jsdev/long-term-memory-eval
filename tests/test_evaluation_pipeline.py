"""Evaluation-pipeline tests (HLD component: evaluation).

HLD components (not yet separate packages — do not invent new ones here):
  i)   pre-processing
  ii)  teacher memory / teacher orchestration
  iii) post-processing
  iv)  evaluation  ← this file (string metrics + LoCoMo F1)

Online / LLM autoraters are out of scope until that split is designed.

Parse and C0/C1 memory checks remain in this module as the frozen *inputs*
to the scorer, including a C0 vs C1 cache-key sanity check (different memories
must not hash to the same answer-reader payload). Dedicated pre-processing /
teacher-memory test modules wait on HLD lock-in — do not rename production
classes to match an unfinished LLD.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.dataset import parse_sample
from src.locomo_eval.memory import (
    RawConversationMemoryBuilder,
    SessionSummaryMemoryBuilder,
    get_memory_builder,
    resolve_memory_name,
)
from src.locomo_eval.metrics import exact_match, normalize_answer, token_f1
from src.locomo_eval.prompts import load_prompt_template, render_qa_prompt
from src.locomo_eval.utils.llm_response_cache import (
    PIPELINE_STAGE_ANSWER_READER,
    LlmResponseCache,
)
from src.metrics.locomo_qa import score_prediction

# Must match OpenAIReader.answer cache_payload + configs/baseline.yaml reader knobs.
_ANSWER_READER_CACHE_MODEL = "gpt-4.1-mini"
_ANSWER_READER_CACHE_TEMPERATURE = 0.0
_ANSWER_READER_CACHE_MAX_TOKENS = 64


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


class TestSessionSummaryMemoryBuilder(unittest.TestCase):
    def test_build_sets_memory_type_to_c1_session_summary(self):
        conv = parse_sample(MINI)
        mem = SessionSummaryMemoryBuilder().build(conv, conv.questions[0])
        self.assertEqual(mem.memory_type, "c1_session_summary")

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
    def test_build_sets_memory_type_to_c0_raw(self):
        conv = parse_sample(MINI)
        mem = RawConversationMemoryBuilder().build(conv, conv.questions[0])
        self.assertEqual(mem.memory_type, "c0_raw")

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


def _answer_reader_cache_key(memory_text: str, question: str, template: str) -> str:
    """Hash the payload OpenAIReader would use if LlmResponseCache were wired."""
    prompt = render_qa_prompt(template, memory=memory_text, question=question)
    return LlmResponseCache.make_key(
        {
            "pipeline_stage": PIPELINE_STAGE_ANSWER_READER,
            "provider": "openai",
            "model": _ANSWER_READER_CACHE_MODEL,
            "temperature": _ANSWER_READER_CACHE_TEMPERATURE,
            "max_tokens": _ANSWER_READER_CACHE_MAX_TOKENS,
            "prompt": prompt,
        }
    )


class TestC0AndC1WouldNotShareAnswerReaderCacheKeys(unittest.TestCase):
    """Different conditions → different memory → different prompts → different keys.

    LlmResponseCache is unwired; this is the confound lock for when it returns.
    Keep here until a memory-component test module exists.
    """

    def setUp(self):
        conv = parse_sample(MINI)
        self.question = conv.questions[0]
        self.mem_c0 = RawConversationMemoryBuilder().build(conv, self.question)
        self.mem_c1 = SessionSummaryMemoryBuilder().build(conv, self.question)
        _, self.template = load_prompt_template(ROOT / "prompts" / "qa_v1.txt")

    def test_c0_and_c1_builders_produce_different_memory_text_for_the_same_question(self):
        self.assertNotEqual(self.mem_c0.text, self.mem_c1.text)
        self.assertIn("I started painting", self.mem_c0.text)
        self.assertIn("Alice said she started painting", self.mem_c1.text)

    def test_same_c1_memory_and_question_hash_to_the_same_answer_reader_cache_key(self):
        key_a = _answer_reader_cache_key(self.mem_c1.text, self.question.question, self.template)
        key_b = _answer_reader_cache_key(self.mem_c1.text, self.question.question, self.template)
        self.assertEqual(key_a, key_b)

    def test_c0_and_c1_rendered_prompts_hash_to_different_answer_reader_cache_keys(self):
        prompt_c0 = render_qa_prompt(self.template, self.mem_c0.text, self.question.question)
        prompt_c1 = render_qa_prompt(self.template, self.mem_c1.text, self.question.question)
        self.assertNotEqual(prompt_c0, prompt_c1)
        key_c0 = _answer_reader_cache_key(self.mem_c0.text, self.question.question, self.template)
        key_c1 = _answer_reader_cache_key(self.mem_c1.text, self.question.question, self.template)
        self.assertNotEqual(key_c0, key_c1)


class TestResolveMemoryName(unittest.TestCase):
    def test_resolve_memory_name_maps_c0_alias_to_c0_raw(self):
        self.assertEqual(resolve_memory_name("c0"), "c0_raw")

    def test_resolve_memory_name_maps_session_summary_alias_to_c1(self):
        self.assertEqual(resolve_memory_name("session_summary"), "c1_session_summary")

    def test_resolve_memory_name_raises_for_unknown_builder(self):
        with self.assertRaises(ValueError):
            resolve_memory_name("not_a_builder")


class TestGetMemoryBuilder(unittest.TestCase):
    def test_get_memory_builder_session_summary_alias_returns_c1_builder(self):
        builder = get_memory_builder("session_summary")
        self.assertEqual(builder.name, "c1_session_summary")

    def test_get_memory_builder_c0_returns_raw_builder(self):
        builder = get_memory_builder("c0")
        self.assertEqual(builder.name, "c0_raw")


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


class TestRealLocomoIfPresent(unittest.TestCase):
    def setUp(self):
        path = ROOT / "data" / "raw" / "locomo10.json"
        if not path.exists():
            self.skipTest("locomo10.json not fetched")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.conv = parse_sample(data[0])

    def test_parse_sample_loads_at_least_one_question_from_real_locomo(self):
        self.assertTrue(self.conv.questions)

    def test_c1_build_produces_memory_longer_than_100_chars_on_real_locomo(self):
        mem = SessionSummaryMemoryBuilder().build(self.conv, self.conv.questions[0])
        self.assertGreater(len(mem.text), 100)

    def test_c0_build_produces_memory_longer_than_100_chars_on_real_locomo(self):
        mem = RawConversationMemoryBuilder().build(self.conv, self.conv.questions[0])
        self.assertGreater(len(mem.text), 100)


if __name__ == "__main__":
    unittest.main()
