"""Unit locks: no LLM cache/resume machinery; each run is self-contained.

Offline / mock only. Behavioral replace-not-append contracts also live in
``tests/test_regressions.py``.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.autorater import get_autorater
from src.locomo_eval.mem0.embeddings import get_embedder
from src.locomo_eval.mem0.extract import get_fact_extractor
from src.locomo_eval.mem0.update import get_memory_updater
from src.locomo_eval.readers import OpenAIChatCaller, get_reader
from src.locomo_eval.report import write_predictions_csv
from src.locomo_eval.schemas import Prediction
from src.locomo_eval.teachers import get_teacher
from tests.test_regressions import _load_json, _load_jsonl, _run_one

FORBIDDEN_CONFIG_TOKENS = (
    "cache_dir",
    "llm_response_hash",
    "llm_response_cache",
    "llm_request_hash",
)
FORBIDDEN_PARAM = "llm_response_hash"
FACTORIES = (
    get_reader,
    get_teacher,
    get_autorater,
    get_embedder,
    get_fact_extractor,
    get_memory_updater,
)


class TestHashModulesAreAbsent(unittest.TestCase):
    def test_utils_package_does_not_exist(self):
        self.assertFalse((ROOT / "src" / "locomo_eval" / "utils").exists())

    def test_llm_request_hash_module_does_not_exist(self):
        self.assertFalse((ROOT / "src" / "locomo_eval" / "utils" / "llm_request_hash.py").is_file())

    def test_llm_response_hash_module_does_not_exist(self):
        self.assertFalse((ROOT / "src" / "locomo_eval" / "utils" / "llm_response_hash.py").is_file())

    def test_llm_request_hash_is_not_importable(self):
        with self.assertRaises(ModuleNotFoundError):
            __import__("src.locomo_eval.utils.llm_request_hash")

    def test_llm_response_hash_is_not_importable(self):
        with self.assertRaises(ModuleNotFoundError):
            __import__("src.locomo_eval.utils.llm_response_hash")


class TestConfigsDoNotDeclareCaches(unittest.TestCase):
    def test_pipeline_yaml_files_do_not_declare_cache_or_hash_dirs(self):
        yaml_files = sorted((ROOT / "configs").glob("*.yaml"))
        self.assertTrue(yaml_files)
        for path in yaml_files:
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_CONFIG_TOKENS:
                self.assertNotIn(token, text, f"{path.name} still mentions {token}")


class TestCallSitesHaveNoCacheHooks(unittest.TestCase):
    def test_openai_chat_caller_init_has_no_response_hash_parameter(self):
        self.assertNotIn(FORBIDDEN_PARAM, inspect.signature(OpenAIChatCaller.__init__).parameters)

    def test_openai_chat_caller_complete_has_no_request_extra_or_hash_store(self):
        params = inspect.signature(OpenAIChatCaller.complete).parameters
        self.assertNotIn("request_extra", params)
        self.assertNotIn(FORBIDDEN_PARAM, params)
        self.assertIn("create_extra", params)

    def test_reader_teacher_autorater_and_mem0_factories_have_no_response_hash_parameter(self):
        for factory in FACTORIES:
            params = inspect.signature(factory).parameters
            self.assertNotIn(
                FORBIDDEN_PARAM,
                params,
                f"{factory.__name__} still accepts {FORBIDDEN_PARAM}",
            )


class TestPredictionSchemaHasNoCacheFields(unittest.TestCase):
    def test_prediction_dataclass_has_no_cached_or_request_hash_field(self):
        names = {field.name for field in dataclasses.fields(Prediction)}
        self.assertNotIn("cached", names)
        self.assertNotIn("llm_request_hash", names)

    def test_predictions_csv_header_omits_cached_and_request_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            write_predictions_csv(
                path,
                [
                    {
                        "sample_id": "s",
                        "question_id": "q",
                        "category": 4,
                        "question": "Q?",
                        "reference_answer": "A",
                        "predicted_answer": "A",
                        "exact_match": 1.0,
                        "token_f1": 1.0,
                        "locomo_f1": 1.0,
                        "memory_type": "session_summaries",
                        "reader_model": "mock",
                        "teacher_model": None,
                        "prompt_version": "qa_v1",
                        "memory_text": "memory",
                    }
                ],
            )
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertNotIn("cached", header)
            self.assertNotIn("llm_request_hash", header)


class TestMockRunDoesNotRecordCacheMetadata(unittest.TestCase):
    def test_mock_prediction_call_meta_omits_cached_and_request_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_one(Path(tmp), run_id="iso_meta", memory="session_summaries")
            row = _load_jsonl(run_dir / "predictions.jsonl")[0]
            meta = _load_json(run_dir / "run_meta.json")
            self.assertNotIn("cached", row)
            self.assertNotIn("llm_request_hash", row)
            self.assertNotIn("pipeline_stage", row)
            self.assertEqual(meta["n_new_api_calls"], meta["n_predictions"])
            self.assertNotIn("n_resumed", meta)


if __name__ == "__main__":
    unittest.main()
