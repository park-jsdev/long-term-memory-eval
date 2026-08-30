"""Preprocessing-pipeline tests (HLD component: pre-processing).

HLD:
  i)   pre-processing  ← this file (ingest, session blocks, persist)
  ii)  teacher orchestrator — passthrough seam only (no LLM)
  iii) post-processing — not this file
  iv)  evaluation — tests/test_evaluation_pipeline.py

run.py is not wired to this package yet.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.preprocess import (
    DataIngestor,
    PreprocessingPipeline,
    assign_turn_ids,
    normalize_speakers_and_times,
    parse_session_datetime,
    segment_sessions,
    write_conversation_run_log,
)
from src.locomo_eval.teacher_orchestrator import (
    TeacherOrchestrator,
    iter_session_blocks,
    process_session_block,
)

GOLD_LOCK = "UNIQ_GOLD_REF_ZZZ"

MINI = {
    "sample_id": "conv-pre",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
            {"dia_id": "D1:2", "speaker": " bob ", "text": "Nice!"},
        ],
        "session_2_date_time": "not a real date",
        "session_2": [],
        "session_3_date_time": "1:56 pm on 8 May, 2023",
        "session_3": [
            {"dia_id": "D3:1", "speaker": "Alice", "text": "I got a nursing job."},
        ],
    },
    "session_summary": {
        "session_1_summary": "Alice said she started painting.",
        "session_3_summary": "Alice got a nursing job.",
    },
    "observation": {},
    "qa": [
        {
            "question": "What hobby did Alice begin?",
            "answer": GOLD_LOCK,
            "category": 4,
            "evidence": ["D1:1"],
        }
    ],
}


def _ingest_mini():
    return DataIngestor().ingest_sample(MINI)


class TestDataIngestorDoesNotRewriteSourceJson(unittest.TestCase):
    def test_load_leaves_source_json_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "locomo.json"
            original = json.dumps([MINI], ensure_ascii=False)
            path.write_text(original, encoding="utf-8")
            DataIngestor().load(path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)


class TestSegmentSessionsDropsEmptyAndKeepsOrder(unittest.TestCase):
    def test_segment_sessions_skips_empty_session_and_orders_remaining_by_id(self):
        conv = _ingest_mini()
        blocks = segment_sessions(conv)
        self.assertEqual([b.session_id for b in blocks], [1, 3])
        self.assertEqual([b.session_index for b in blocks], [0, 1])
        self.assertEqual([b.source_key for b in blocks], ["session_1", "session_3"])


class TestAssignTurnIdsIsStableAndPreservesDiaId(unittest.TestCase):
    def test_assign_turn_ids_sets_padded_turn_id_and_keeps_source_dia_id(self):
        conv = _ingest_mini()
        blocks = assign_turn_ids(segment_sessions(conv))
        first = blocks[0].turns
        self.assertEqual(first[0].turn_id, "conv-pre:s1:t000")
        self.assertEqual(first[1].turn_id, "conv-pre:s1:t001")
        self.assertEqual(first[0].source_dia_id, "D1:1")
        self.assertEqual(first[1].source_dia_id, "D1:2")
        self.assertEqual(first[0].turn_index, 0)
        self.assertEqual(first[1].turn_index, 1)


class TestParseSessionDatetimeFormats(unittest.TestCase):
    def test_parse_session_datetime_returns_iso_for_locomo_release_string(self):
        self.assertEqual(
            parse_session_datetime("1:56 pm on 8 May, 2023"),
            "2023-05-08T13:56:00",
        )

    def test_parse_session_datetime_returns_iso_date_for_day_month_year(self):
        self.assertEqual(parse_session_datetime("1 Jan 2023"), "2023-01-01")

    def test_parse_session_datetime_returns_none_when_unparseable(self):
        self.assertIsNone(parse_session_datetime("not a real date"))


class TestNormalizeSpeakersAndTimes(unittest.TestCase):
    def test_normalize_speakers_and_times_maps_roles_and_iso_dates(self):
        conv = _ingest_mini()
        blocks = normalize_speakers_and_times(assign_turn_ids(segment_sessions(conv)))
        session1, session3 = blocks
        self.assertEqual(session1.turns[0].speaker_role, "a")
        self.assertEqual(session1.turns[1].speaker_role, "b")
        self.assertEqual(session1.turns[1].speaker_raw, " bob ")
        self.assertEqual(session1.date_time_raw, "1 Jan 2023")
        self.assertEqual(session1.date_time_normalized, "2023-01-01")
        self.assertEqual(session3.date_time_raw, "1:56 pm on 8 May, 2023")
        self.assertEqual(session3.date_time_normalized, "2023-05-08T13:56:00")

    def test_normalize_speakers_and_times_sets_normalized_null_when_date_unparseable(self):
        conv = DataIngestor().ingest_sample(
            {
                **MINI,
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1_date_time": "not a real date",
                    "session_1": [
                        {"dia_id": "D1:1", "speaker": "Carol", "text": "hi"},
                    ],
                },
            }
        )
        blocks = normalize_speakers_and_times(assign_turn_ids(segment_sessions(conv)))
        self.assertEqual(blocks[0].date_time_raw, "not a real date")
        self.assertIsNone(blocks[0].date_time_normalized)
        self.assertEqual(blocks[0].turns[0].speaker_role, "other")


class TestPreprocessingPipelineOmitsGold(unittest.TestCase):
    def test_process_does_not_copy_gold_answer_into_session_blocks(self):
        processed = PreprocessingPipeline().process(_ingest_mini())
        dumped = json.dumps(processed.to_dict(), ensure_ascii=False)
        self.assertNotIn(GOLD_LOCK, dumped)
        for block in processed.session_blocks:
            self.assertNotIn("answer", block.to_dict())
        self.assertEqual(processed.question_ids, ["conv-pre-q-0"])


class TestTeacherOrchestratorWalksOneBlockAtATime(unittest.TestCase):
    def test_iter_session_blocks_yields_first_session_then_second(self):
        processed = PreprocessingPipeline().process(_ingest_mini())
        blocks = list(iter_session_blocks(processed))
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].session_id, 1)
        self.assertEqual(blocks[1].session_id, 3)

    def test_process_session_block_returns_passthrough_status_and_turn_ids(self):
        processed = PreprocessingPipeline().process(_ingest_mini())
        record = process_session_block(processed.session_blocks[0])
        self.assertEqual(record["status"], "passthrough")
        self.assertEqual(record["session_id"], 1)
        self.assertEqual(record["n_turns"], 2)
        self.assertEqual(
            record["turn_ids"],
            ["conv-pre:s1:t000", "conv-pre:s1:t001"],
        )

    def test_teacher_orchestrator_class_matches_module_functions(self):
        processed = PreprocessingPipeline().process(_ingest_mini())
        orch = TeacherOrchestrator()
        first = next(orch.iter_session_blocks(processed))
        self.assertEqual(
            orch.process_session_block(first),
            process_session_block(first),
        )


class TestWriteConversationRunLogRoundTrip(unittest.TestCase):
    def test_write_conversation_run_log_round_trips_session_and_turn_ids(self):
        processed = PreprocessingPipeline().process(_ingest_mini())
        with tempfile.TemporaryDirectory() as tmp:
            out = write_conversation_run_log(tmp, [processed])
            index_path = out / "index.jsonl"
            sessions_path = out / "by_sample" / "conv-pre" / "sessions.jsonl"
            self.assertTrue((out / "schema.json").is_file())
            index = json.loads(index_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(index["sample_id"], "conv-pre")
            self.assertEqual(index["n_session_blocks"], 2)
            self.assertEqual(index["n_turns"], 3)
            rows = [
                json.loads(line)
                for line in sessions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([r["session_id"] for r in rows], [1, 3])
            self.assertEqual(rows[0]["turns"][0]["turn_id"], "conv-pre:s1:t000")
            blob = sessions_path.read_text(encoding="utf-8")
            self.assertNotIn(GOLD_LOCK, blob)


if __name__ == "__main__":
    unittest.main()
