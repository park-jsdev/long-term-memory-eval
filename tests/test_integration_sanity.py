"""Model-integration sanity: reader swap, teacher swap, evaluator disagreement.

Offline only (mock / deterministic doubles). Live API is a separate CLI run.
Covers the two new seams: (1) answer/eval model swap, (2) teacher model swap
within a family. Also one LoCoMo-vs-SPEC scorer check.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.dataset import parse_sample
from src.locomo_eval.memory import (
    SessionSummaryMemoryBuilder,
    TeacherSessionMemoryBuilder,
    get_memory_builder,
)
from src.locomo_eval.memory_log import write_memory_run_log
from src.locomo_eval.metrics import exact_match, score_row, summarize_predictions, token_f1
from src.locomo_eval.models import (
    BASELINE_READER_MODEL,
    FAMILY_GPT41,
    FAMILY_GPT56,
    GPT56_LUNA,
    GPT56_TERRA,
    chat_create_kwargs,
    models_in_family,
    resolve_model,
    same_family,
)
from src.locomo_eval.prompts import load_prompt_template
from src.locomo_eval.readers import MockReader, Reader, get_reader
from src.locomo_eval.schemas import Prediction
from src.locomo_eval.teachers import MockTeacher, get_teacher
from src.metrics.locomo_qa import score_prediction
from scripts.compare_cross_model import cross_model_analysis, theoretical_cache_key
from scripts.compare_full_runs import load_pack


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
    "qa": [
        {
            "question": "What hobby did Alice begin?",
            "answer": "painting",
            "category": 4,
            "evidence": ["D1:1"],
        },
    ],
}

# Verbose baseline vs short Luna: same gold, different EM / LoCoMo F1.
READER_ANSWERS = {
    BASELINE_READER_MODEL: "Alice started painting as a hobby.",
    GPT56_LUNA: "painting",
}


class DeterministicReader(Reader):
    """Test double: predicted answer is a function of model_id (not the API)."""

    def __init__(self, model_name: str, answers: dict[str, str] | None = None):
        self.model_name = resolve_model(model_name).model_id
        self.answers = answers or {}

    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        text = self.answers.get(self.model_name, f"Unknown ({self.model_name}).")
        return text, {
            "cached": False,
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
        }


class EchoMemoryReader(Reader):
    """Test double: answer is a clip of Memory.text so teacher swaps propagate."""

    def __init__(self, model_name: str = BASELINE_READER_MODEL):
        self.model_name = resolve_model(model_name).model_id

    def answer(self, memory: str, question: str, prompt_template: str) -> tuple[str, dict[str, Any]]:
        return memory[:120], {
            "cached": False,
            "latency_s": 0.0,
            "usage": {},
            "model": self.model_name,
        }


def _c1_memory_text() -> str:
    conv = parse_sample(MINI)
    return SessionSummaryMemoryBuilder().build(conv, conv.questions[0]).text


def _qa_template() -> str:
    _, text = load_prompt_template(ROOT / "prompts" / "qa_v1.txt")
    return text


def _score_reader(model: str) -> dict[str, float]:
    conv = parse_sample(MINI)
    q = conv.questions[0]
    memory = _c1_memory_text()
    pred, _meta = DeterministicReader(model, READER_ANSWERS).answer(
        memory, q.question, _qa_template()
    )
    return score_row(pred, q.answer, q.category)


def _prediction_row(*, model: str, answer: str, memory: str, teacher_model: str | None = None) -> dict:
    conv = parse_sample(MINI)
    q = conv.questions[0]
    pred = Prediction(
        sample_id=q.sample_id,
        question_id=q.question_id,
        question=q.question,
        reference_answer=q.answer,
        predicted_answer=answer,
        category=q.category,
        memory_type="c1_session_summary" if not teacher_model else "c1_teacher",
        memory_text=memory,
        reader_model=resolve_model(model).model_id,
        prompt_version="qa_v1",
        run_id="test",
        teacher_model=teacher_model,
    )
    return pred.to_dict()


class TestModelCatalog(unittest.TestCase):
    def test_resolve_model_maps_luna_alias_to_gpt56_luna_id(self):
        self.assertEqual(resolve_model("luna").model_id, GPT56_LUNA)

    def test_resolve_model_maps_baseline_alias_to_gpt41_mini(self):
        self.assertEqual(resolve_model("baseline").model_id, BASELINE_READER_MODEL)

    def test_gpt41_mini_and_luna_are_different_families(self):
        self.assertFalse(same_family(BASELINE_READER_MODEL, GPT56_LUNA))
        self.assertEqual(resolve_model(BASELINE_READER_MODEL).family, FAMILY_GPT41)
        self.assertEqual(resolve_model(GPT56_LUNA).family, FAMILY_GPT56)

    def test_luna_and_terra_share_gpt56_family(self):
        self.assertTrue(same_family(GPT56_LUNA, GPT56_TERRA))
        self.assertIn(GPT56_LUNA, models_in_family(FAMILY_GPT56))
        self.assertIn(GPT56_TERRA, models_in_family(FAMILY_GPT56))

    def test_chat_create_kwargs_for_luna_uses_max_completion_tokens_not_temperature(self):
        spec = resolve_model(GPT56_LUNA)
        kwargs = chat_create_kwargs(
            spec,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=64,
        )
        self.assertEqual(kwargs["model"], GPT56_LUNA)
        self.assertIn("max_completion_tokens", kwargs)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs.get("reasoning_effort"), "none")

    def test_chat_create_kwargs_for_gpt41_mini_keeps_max_tokens_and_temperature(self):
        spec = resolve_model(BASELINE_READER_MODEL)
        kwargs = chat_create_kwargs(
            spec,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=64,
        )
        self.assertIn("max_tokens", kwargs)
        self.assertIn("temperature", kwargs)
        self.assertNotIn("max_completion_tokens", kwargs)


class TestReaderModelSwap(unittest.TestCase):
    """Feature 1: swapping the answer/eval model changes outputs, scores, logs."""

    def test_get_reader_mock_resolves_luna_alias_on_model_name(self):
        reader = get_reader("mock", model="luna")
        self.assertIsInstance(reader, MockReader)
        self.assertEqual(reader.model_name, GPT56_LUNA)

    def test_reader_model_swap_changes_predicted_answer(self):
        memory = _c1_memory_text()
        q = "What hobby did Alice begin?"
        tmpl = _qa_template()
        a, _ = DeterministicReader(BASELINE_READER_MODEL, READER_ANSWERS).answer(memory, q, tmpl)
        b, _ = DeterministicReader(GPT56_LUNA, READER_ANSWERS).answer(memory, q, tmpl)
        self.assertNotEqual(a, b)

    def test_reader_model_swap_changes_locomo_f1_and_exact_match(self):
        scores_mini = _score_reader(BASELINE_READER_MODEL)
        scores_luna = _score_reader(GPT56_LUNA)
        self.assertNotEqual(scores_mini["locomo_f1"], scores_luna["locomo_f1"])
        self.assertNotEqual(scores_mini["exact_match"], scores_luna["exact_match"])
        self.assertEqual(scores_luna["exact_match"], 1.0)
        self.assertEqual(scores_mini["exact_match"], 0.0)

    def test_reader_model_swap_is_logged_on_prediction_rows(self):
        memory = _c1_memory_text()
        row_mini = _prediction_row(
            model=BASELINE_READER_MODEL,
            answer=READER_ANSWERS[BASELINE_READER_MODEL],
            memory=memory,
        )
        row_luna = _prediction_row(
            model=GPT56_LUNA,
            answer=READER_ANSWERS[GPT56_LUNA],
            memory=memory,
        )
        self.assertEqual(row_mini["reader_model"], BASELINE_READER_MODEL)
        self.assertEqual(row_luna["reader_model"], GPT56_LUNA)
        self.assertNotEqual(row_mini["reader_model"], row_luna["reader_model"])

    def test_reader_model_swap_produces_different_reader_cache_keys(self):
        memory = _c1_memory_text()
        q = parse_sample(MINI).questions[0]
        tmpl = _qa_template()
        meta = {"temperature": 0.0, "max_tokens": 64, "reader_model": BASELINE_READER_MODEL}
        row_a = {
            "reader_model": BASELINE_READER_MODEL,
            "memory_text": memory,
            "question": q.question,
        }
        row_b = {
            "reader_model": GPT56_LUNA,
            "memory_text": memory,
            "question": q.question,
        }
        key_a = theoretical_cache_key(row_a, meta, tmpl)
        key_b = theoretical_cache_key(row_b, {**meta, "reader_model": GPT56_LUNA}, tmpl)
        self.assertNotEqual(key_a, key_b)

    def test_summarize_predictions_overall_metrics_differ_after_reader_swap(self):
        memory = _c1_memory_text()
        rows_mini = [
            _prediction_row(
                model=BASELINE_READER_MODEL,
                answer=READER_ANSWERS[BASELINE_READER_MODEL],
                memory=memory,
            )
        ]
        rows_luna = [
            _prediction_row(
                model=GPT56_LUNA,
                answer=READER_ANSWERS[GPT56_LUNA],
                memory=memory,
            )
        ]
        sum_mini = summarize_predictions(rows_mini)
        sum_luna = summarize_predictions(rows_luna)
        self.assertNotEqual(sum_mini["metrics"]["locomo_f1"], sum_luna["metrics"]["locomo_f1"])
        self.assertNotEqual(sum_mini["metrics"]["exact_match"], sum_luna["metrics"]["exact_match"])


class TestLocomoEvaluatorSanity(unittest.TestCase):
    """Scorer is still LoCoMo F1; this only checks it can disagree with SPEC EM."""

    def test_locomo_f1_is_one_when_spec_exact_match_is_zero_for_reordered_dates(self):
        pred = "May 7 2023"
        gold = "7 May 2023"
        self.assertEqual(score_prediction(pred, gold, 2), 1.0)
        self.assertEqual(exact_match(pred, gold), 0.0)

    def test_token_f1_and_locomo_f1_disagree_on_adversarial_not_mentioned(self):
        pred = "not mentioned"
        gold = "painting"
        self.assertEqual(score_prediction(pred, gold, 5), 1.0)
        self.assertEqual(token_f1(pred, gold), 0.0)


class TestTeacherModelSwap(unittest.TestCase):
    """Feature 2: swapping teacher model within a family changes memory and logs."""

    def test_get_teacher_mock_resolves_luna_alias_on_model_name(self):
        teacher = get_teacher("mock", model="luna")
        self.assertIsInstance(teacher, MockTeacher)
        self.assertEqual(teacher.model_name, GPT56_LUNA)

    def test_same_family_teacher_models_produce_different_memory_text(self):
        conv = parse_sample(MINI)
        mem_luna = TeacherSessionMemoryBuilder(MockTeacher(GPT56_LUNA)).build(
            conv, conv.questions[0]
        )
        mem_terra = TeacherSessionMemoryBuilder(MockTeacher(GPT56_TERRA)).build(
            conv, conv.questions[0]
        )
        self.assertTrue(same_family(GPT56_LUNA, GPT56_TERRA))
        self.assertNotEqual(mem_luna.text, mem_terra.text)
        self.assertIn(GPT56_LUNA, mem_luna.text)
        self.assertIn(GPT56_TERRA, mem_terra.text)

    def test_teacher_model_swap_is_logged_on_memory_object(self):
        conv = parse_sample(MINI)
        mem = TeacherSessionMemoryBuilder(MockTeacher(GPT56_LUNA)).build(
            conv, conv.questions[0]
        )
        self.assertEqual(mem.memory_type, "c1_teacher")
        self.assertEqual(mem.teacher_model, GPT56_LUNA)
        self.assertEqual(mem.teacher_provider, "mock")

    def test_get_memory_builder_c1_teacher_records_teacher_model(self):
        builder = get_memory_builder("c1_teacher", teacher=MockTeacher("terra"))
        conv = parse_sample(MINI)
        mem = builder.build(conv, conv.questions[0])
        self.assertEqual(builder.name, "c1_teacher")
        self.assertEqual(mem.teacher_model, GPT56_TERRA)

    def test_teacher_model_swap_changes_downstream_reader_output(self):
        conv = parse_sample(MINI)
        q = conv.questions[0]
        tmpl = _qa_template()
        reader = EchoMemoryReader(BASELINE_READER_MODEL)
        mem_a = TeacherSessionMemoryBuilder(MockTeacher(GPT56_LUNA)).build(conv, q)
        mem_b = TeacherSessionMemoryBuilder(MockTeacher(GPT56_TERRA)).build(conv, q)
        ans_a, _ = reader.answer(mem_a.text, q.question, tmpl)
        ans_b, _ = reader.answer(mem_b.text, q.question, tmpl)
        self.assertNotEqual(ans_a, ans_b)

    def test_teacher_model_swap_is_written_to_memory_audit_logs(self):
        conv = parse_sample(MINI)
        mem = TeacherSessionMemoryBuilder(MockTeacher(GPT56_LUNA)).build(
            conv, conv.questions[0]
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_memory_run_log(
                run_dir,
                {conv.sample_id: mem},
                max_chars_cfg=None,
                prompt_template=_qa_template(),
                example_question=conv.questions[0].question,
            )
            schema = json.loads((run_dir / "memory" / "schema.json").read_text(encoding="utf-8"))
            index_line = (run_dir / "memory" / "index.jsonl").read_text(encoding="utf-8").splitlines()[0]
            index_row = json.loads(index_line)
            self.assertEqual(schema["teacher_model"], GPT56_LUNA)
            self.assertEqual(index_row["teacher_model"], GPT56_LUNA)
            self.assertEqual(schema["memory_type"], "c1_teacher")

    def test_teacher_and_dataset_c1_memories_are_different_texts(self):
        conv = parse_sample(MINI)
        q = conv.questions[0]
        dataset_c1 = SessionSummaryMemoryBuilder().build(conv, q)
        teacher_c1 = TeacherSessionMemoryBuilder(MockTeacher(GPT56_LUNA)).build(conv, q)
        self.assertNotEqual(dataset_c1.text, teacher_c1.text)
        self.assertEqual(dataset_c1.memory_type, "c1_session_summary")
        self.assertEqual(teacher_c1.memory_type, "c1_teacher")


class TestCrossModelCompareScript(unittest.TestCase):
    def _write_pack(
        self,
        root: Path,
        run_id: str,
        *,
        reader_model: str,
        answer: str,
        memory: str,
        teacher_model: str | None = None,
    ) -> Path:
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        row = _prediction_row(
            model=reader_model,
            answer=answer,
            memory=memory,
            teacher_model=teacher_model,
        )
        (run_dir / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        summary = summarize_predictions([row])
        summary["memory_type"] = row["memory_type"]
        summary["reader_model"] = row["reader_model"]
        if teacher_model:
            summary["teacher_model"] = teacher_model
        (run_dir / "metrics.json").write_text(json.dumps(summary), encoding="utf-8")
        meta = {
            "run_id": run_id,
            "reader_model": row["reader_model"],
            "teacher_model": teacher_model,
            "temperature": 0.0,
            "max_tokens": 64,
            "memory_type": row["memory_type"],
            "prompt_version": "qa_v1",
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return run_dir

    def test_compare_reader_axis_reports_answer_and_score_disagreement(self):
        memory = _c1_memory_text()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dir_a = self._write_pack(
                root,
                "mini",
                reader_model=BASELINE_READER_MODEL,
                answer=READER_ANSWERS[BASELINE_READER_MODEL],
                memory=memory,
            )
            dir_b = self._write_pack(
                root,
                "luna",
                reader_model=GPT56_LUNA,
                answer=READER_ANSWERS[GPT56_LUNA],
                memory=memory,
            )
            packs = [load_pack(dir_a), load_pack(dir_b)]
            paired = cross_model_analysis(packs, ROOT / "prompts" / "qa_v1.txt", "reader")
        self.assertEqual(paired["n_answer_disagreements"], 1)
        self.assertEqual(paired["fraction_same_answer"], 0.0)
        self.assertEqual(paired["fraction_same_memory_text"], 1.0)
        self.assertTrue(paired["sanity"]["reader_models_differ"])
        self.assertTrue(paired["sanity"]["memory_held_fixed"])
        self.assertNotEqual(paired["mean_delta_locomo_f1_b_minus_a"], 0)

    def test_compare_teacher_axis_reports_memory_text_difference(self):
        conv = parse_sample(MINI)
        q = conv.questions[0]
        mem_luna = TeacherSessionMemoryBuilder(MockTeacher(GPT56_LUNA)).build(conv, q)
        mem_terra = TeacherSessionMemoryBuilder(MockTeacher(GPT56_TERRA)).build(conv, q)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dir_a = self._write_pack(
                root,
                "teacher_luna",
                reader_model=BASELINE_READER_MODEL,
                answer=mem_luna.text[:80],
                memory=mem_luna.text,
                teacher_model=GPT56_LUNA,
            )
            dir_b = self._write_pack(
                root,
                "teacher_terra",
                reader_model=BASELINE_READER_MODEL,
                answer=mem_terra.text[:80],
                memory=mem_terra.text,
                teacher_model=GPT56_TERRA,
            )
            packs = [load_pack(dir_a), load_pack(dir_b)]
            paired = cross_model_analysis(packs, ROOT / "prompts" / "qa_v1.txt", "teacher")
        self.assertEqual(paired["fraction_same_memory_text"], 0.0)
        self.assertTrue(paired["sanity"]["teacher_models_differ"])
        self.assertTrue(paired["sanity"]["same_teacher_family"])
        self.assertTrue(paired["sanity"]["reader_held_fixed"])
        self.assertTrue(paired["sanity"]["memory_texts_differ"])


if __name__ == "__main__":
    unittest.main()
