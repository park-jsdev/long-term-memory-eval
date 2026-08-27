"""Mem0 / Mem0g write-index tests (mock only, no API).

One TestCase per function/class. Names = behavior + expected outcome.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.dataset import parse_sample
from src.locomo_eval.memory import get_memory_builder, is_question_independent
from src.locomo_eval.mem0.embeddings import MockEmbedder, cosine_similarity, get_embedder
from src.locomo_eval.mem0.extract import MockFactExtractor, get_fact_extractor
from src.locomo_eval.mem0.graph_memory import Mem0GraphMemory
from src.locomo_eval.mem0.ingest import iter_speaker_pairs, user_messages_text
from src.locomo_eval.mem0.indexer import Mem0Indexer
from src.locomo_eval.mem0.dump import write_sample_dump
from src.locomo_eval.mem0.retrieve import (
    MissingMem0IndexError,
    format_mem0_text,
    retrieve_speaker_facts,
)
from src.locomo_eval.mem0.schemas import ADD, DELETE, NONE, UPDATE, Fact, UpdateEvent
from src.locomo_eval.mem0.update import (
    MockMemoryUpdater,
    apply_update_events,
    get_memory_updater,
    parse_update_events,
)
from src.locomo_eval.mem0.vector_store import VectorMemoryStore
from src.locomo_eval.preprocess import DataIngestor, PreprocessingPipeline

GOLD_LOCK = "UNIQ_GOLD_REF_ZZZ"

MINI = {
    "sample_id": "conv-mem0",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1 Jan 2023",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice!"},
            {"dia_id": "D1:3", "speaker": "Alice", "text": "I also live in Boston."},
        ],
        "session_2_date_time": "2 Jan 2023",
        "session_2": [
            {"dia_id": "D2:1", "speaker": "Alice", "text": "I moved to Seattle."},
        ],
    },
    "session_summary": {"session_1_summary": "Alice paints."},
    "qa": [
        {
            "question": "Where does Alice live?",
            "answer": GOLD_LOCK,
            "category": 4,
            "evidence": ["D1:3"],
        }
    ],
}


def _processed(sample=None):
    conv = DataIngestor().ingest_sample(sample or MINI)
    return PreprocessingPipeline().process(conv)


class TestIterSpeakerPairsBatchesAndRoleFlips(unittest.TestCase):
    def test_three_turns_yield_two_pairs_per_speaker_with_remainder_singleton(self):
        processed = _processed()
        pairs = list(iter_speaker_pairs(processed, batch_size=2))
        a_pairs = [p for p in pairs if p.speaker_index == "a"]
        b_pairs = [p for p in pairs if p.speaker_index == "b"]
        self.assertEqual(len(a_pairs), 3)  # sess1: 2+1, sess2: 1
        self.assertEqual(len(b_pairs), 3)
        self.assertEqual(len(a_pairs[0].messages), 2)
        self.assertEqual(len(a_pairs[1].messages), 1)

    def test_role_flip_marks_index_speaker_as_user_and_other_as_assistant(self):
        processed = _processed()
        pairs = list(iter_speaker_pairs(processed, batch_size=2))
        a0 = next(p for p in pairs if p.speaker_index == "a")
        self.assertEqual(a0.messages[0].role, "user")
        self.assertEqual(a0.messages[1].role, "assistant")
        b0 = next(p for p in pairs if p.speaker_index == "b")
        self.assertEqual(b0.messages[0].role, "assistant")
        self.assertEqual(b0.messages[1].role, "user")

    def test_pair_content_prefixes_speaker_name_and_keeps_session_timestamp(self):
        processed = _processed()
        pair = next(iter_speaker_pairs(processed, batch_size=2))
        self.assertTrue(pair.messages[0].content.startswith("Alice:"))
        self.assertEqual(pair.timestamp, "1 Jan 2023")


class TestUserMessagesTextExcludesAssistant(unittest.TestCase):
    def test_user_messages_text_omits_assistant_role_lines(self):
        processed = _processed()
        pair = next(p for p in iter_speaker_pairs(processed, batch_size=2) if p.speaker_index == "a")
        text = user_messages_text(pair)
        self.assertIn("painting", text)
        self.assertNotIn("Nice!", text)


class TestMockFactExtractorUserOnly(unittest.TestCase):
    def test_extract_returns_user_lines_and_does_not_include_assistant_text(self):
        processed = _processed()
        pair = next(p for p in iter_speaker_pairs(processed, batch_size=2) if p.speaker_index == "a")
        extractor = MockFactExtractor(model_name="gpt-4o-mini")
        facts, _meta = extractor.extract(pair)
        blob = " ".join(facts)
        self.assertIn("painting", blob)
        self.assertNotIn("Nice!", blob)
        self.assertNotIn("Nice!", extractor.last_user_text)


class TestApplyUpdateEventsOps(unittest.TestCase):
    def setUp(self):
        self.embedder = MockEmbedder()
        self.store = VectorMemoryStore(
            sample_id="conv-mem0", speaker_index="a", speaker_name="Alice"
        )
        existing = Fact(
            fact_id=self.store.next_fact_id(),
            text="Alice lives in Boston",
            timestamp="1 Jan 2023",
            embedding=self.embedder.embed_one("Alice lives in Boston"),
            speaker_index="a",
        )
        self.store.add(existing)
        self.candidates = [existing]

    def test_add_event_appends_a_new_fact_to_the_store(self):
        apply_update_events(
            self.store,
            [UpdateEvent(event=ADD, text="Alice started painting")],
            candidates=self.candidates,
            embedder=self.embedder,
            timestamp="1 Jan 2023",
            source_turn_ids=["t0"],
            speaker_index="a",
        )
        self.assertEqual(len(self.store.facts), 2)

    def test_update_event_replaces_candidate_text_and_keeps_fact_id(self):
        apply_update_events(
            self.store,
            [
                UpdateEvent(
                    event=UPDATE,
                    text="Alice lives in Seattle",
                    local_id="0",
                    old_text="Alice lives in Boston",
                )
            ],
            candidates=self.candidates,
            embedder=self.embedder,
            timestamp="2 Jan 2023",
            source_turn_ids=["t1"],
            speaker_index="a",
        )
        self.assertEqual(len(self.store.facts), 1)
        self.assertEqual(self.store.facts[0].text, "Alice lives in Seattle")
        self.assertEqual(self.store.facts[0].fact_id, "a:0000")

    def test_delete_event_removes_the_candidate_fact(self):
        apply_update_events(
            self.store,
            [UpdateEvent(event=DELETE, text="Alice lives in Boston", local_id="0")],
            candidates=self.candidates,
            embedder=self.embedder,
            timestamp="1 Jan 2023",
            source_turn_ids=["t0"],
            speaker_index="a",
        )
        self.assertEqual(self.store.facts, [])

    def test_none_event_leaves_the_store_unchanged(self):
        apply_update_events(
            self.store,
            [UpdateEvent(event=NONE, text="Alice lives in Boston", local_id="0")],
            candidates=self.candidates,
            embedder=self.embedder,
            timestamp="1 Jan 2023",
            source_turn_ids=["t0"],
            speaker_index="a",
        )
        self.assertEqual(len(self.store.facts), 1)
        self.assertEqual(self.store.facts[0].text, "Alice lives in Boston")


class TestParseUpdateEvents(unittest.TestCase):
    def test_parse_update_events_reads_add_update_delete_none_from_json(self):
        events = parse_update_events(
            '{"memory": ['
            '{"id": "0", "text": "x", "event": "NONE"},'
            '{"id": "1", "text": "y", "event": "ADD"}'
            "]}"
        )
        self.assertEqual([e.event for e in events], [NONE, ADD])


class TestCosineRetrieveReturnsTopKInScoreOrder(unittest.TestCase):
    def test_search_returns_the_most_similar_facts_first(self):
        embedder = MockEmbedder()
        store = VectorMemoryStore(
            sample_id="conv-mem0", speaker_index="a", speaker_name="Alice"
        )
        painting = Fact(
            fact_id="a:0000",
            text="Alice started painting yesterday",
            timestamp="1 Jan 2023",
            embedding=embedder.embed_one("Alice started painting yesterday"),
        )
        pizza = Fact(
            fact_id="a:0001",
            text="Bob likes pizza",
            timestamp="1 Jan 2023",
            embedding=embedder.embed_one("Bob likes pizza"),
        )
        store.add(painting)
        store.add(pizza)
        q = embedder.embed_one("What did Alice start painting")
        hits = retrieve_speaker_facts(store, q, top_k=1)
        self.assertEqual(hits[0].fact_id, "a:0000")
        self.assertGreater(
            cosine_similarity(q, painting.embedding),
            cosine_similarity(q, pizza.embedding),
        )


class TestMem0GraphConflictMarksExclusiveEdgeInvalid(unittest.TestCase):
    def test_lives_in_conflict_sets_old_edge_valid_false_and_adds_new_edge(self):
        graph = Mem0GraphMemory(MockEmbedder(), threshold=0.7)
        graph.ingest("Alice lives in Boston", user_id="Alice", timestamp="1 Jan 2023")
        graph.ingest("Alice moved to Seattle", user_id="Alice", timestamp="2 Jan 2023")
        lives = [
            e for e in graph.edges if e.relationship == "lives_in"
        ]
        boston = [e for e in lives if e.target == "boston"]
        seattle = [e for e in lives if e.target == "seattle"]
        self.assertEqual(len(boston), 1)
        self.assertFalse(boston[0].valid)
        self.assertEqual(len(seattle), 1)
        self.assertTrue(seattle[0].valid)

    def test_loves_to_eat_does_not_invalidate_a_different_destination(self):
        graph = Mem0GraphMemory(MockEmbedder(), threshold=0.7)
        graph.ingest("Alice loves pizza", user_id="Alice", timestamp="1 Jan 2023")
        graph.ingest("Alice loves burger", user_id="Alice", timestamp="2 Jan 2023")
        eats = [e for e in graph.edges if e.relationship == "loves_to_eat" and e.valid]
        targets = {e.target for e in eats}
        self.assertEqual(targets, {"pizza", "burger"})


class TestDumpRoundTripBuilderDoesNotExtract(unittest.TestCase):
    def test_builder_loads_dump_and_does_not_call_extract(self):
        processed = _processed()
        extractor = MockFactExtractor(model_name="gpt-4o-mini")
        indexer = Mem0Indexer(
            extractor=extractor,
            updater=MockMemoryUpdater(model_name="gpt-4o-mini"),
            embedder=MockEmbedder(),
            graph=Mem0GraphMemory(MockEmbedder()),
            batch_size=2,
            similar_s=10,
        )
        store_a, store_b, log = indexer.index_conversation(processed)
        n_extract = extractor.extract
        calls = {"n": 0}

        def _boom(pair):
            calls["n"] += 1
            raise AssertionError("extract must not run on the read seam")

        extractor.extract = _boom  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mem0_index"
            write_sample_dump(
                root,
                sample_id=processed.sample_id,
                store_a=store_a,
                store_b=store_b,
                ingest_log=log,
                graph=indexer.graph,
            )
            conv = parse_sample(MINI)
            q = conv.questions[0]
            builder = get_memory_builder(
                "mem0",
                mem0_index_dir=root,
                mem0_top_k=30,
                mem0_embedder=MockEmbedder(),
            )
            memory = builder.build(conv, q)
            self.assertEqual(memory.memory_type, "mem0")
            self.assertIn("painting", memory.text)
            self.assertEqual(calls["n"], 0)
            gbuilder = get_memory_builder(
                "mem0g",
                mem0_index_dir=root,
                mem0_top_k=30,
                mem0_embedder=MockEmbedder(),
            )
            gmem = gbuilder.build(conv, q)
            self.assertEqual(gmem.memory_type, "mem0g")
            self.assertIn("Graph relations:", gmem.text)
        # keep n_extract referenced so ruff/linters don't flag if we later use it
        self.assertTrue(callable(n_extract))


class TestGoldNeverEntersExtractUpdateOrIngestLog(unittest.TestCase):
    def test_gold_string_is_absent_from_extract_prompt_update_prompt_and_log(self):
        processed = _processed()
        extractor = MockFactExtractor()
        updater = MockMemoryUpdater()
        indexer = Mem0Indexer(
            extractor=extractor,
            updater=updater,
            embedder=MockEmbedder(),
            batch_size=2,
            similar_s=10,
        )
        _a, _b, log = indexer.index_conversation(processed)
        blob = json.dumps(log) + extractor.last_user_text + updater.last_prompt
        for pair in iter_speaker_pairs(processed, batch_size=2):
            blob += user_messages_text(pair)
        self.assertNotIn(GOLD_LOCK, blob)
        for block in processed.session_blocks:
            self.assertNotIn(GOLD_LOCK, json.dumps(block.to_dict()))


class TestMissingDumpNamesRunIndexCommand(unittest.TestCase):
    def test_builder_raises_missing_index_error_that_names_run_index(self):
        conv = parse_sample(MINI)
        q = conv.questions[0]
        with tempfile.TemporaryDirectory() as tmp:
            builder = get_memory_builder(
                "mem0",
                mem0_index_dir=Path(tmp) / "missing",
                mem0_embedder=MockEmbedder(),
            )
            with self.assertRaises(MissingMem0IndexError) as ctx:
                builder.build(conv, q)
        self.assertIn("run_index", str(ctx.exception))


class TestMem0BuildersAreQuestionDependent(unittest.TestCase):
    def test_is_question_independent_returns_false_for_mem0_and_mem0g(self):
        self.assertFalse(is_question_independent("mem0"))
        self.assertFalse(is_question_independent("mem0g"))
        self.assertTrue(is_question_independent("session_summaries"))


class TestFactoriesRejectLlmResponseHash(unittest.TestCase):
    def test_extract_update_embed_factories_raise_when_hash_store_is_passed(self):
        sentinel = object()
        with self.assertRaises(ValueError):
            get_fact_extractor("mock", model="gpt-4o-mini", llm_response_hash=sentinel)
        with self.assertRaises(ValueError):
            get_memory_updater("mock", model="gpt-4o-mini", llm_response_hash=sentinel)
        with self.assertRaises(ValueError):
            get_embedder("mock", llm_response_hash=sentinel)


class TestFormatMem0TextTimestampedLines(unittest.TestCase):
    def test_format_uses_timestamp_colon_memory_and_optional_graph_block(self):
        fact = Fact(
            fact_id="a:0000",
            text="Alice started painting",
            timestamp="1 Jan 2023",
        )
        text = format_mem0_text(
            speaker_a="Alice",
            speaker_b="Bob",
            facts_a=[fact],
            facts_b=[],
            edges=None,
        )
        self.assertIn("1 Jan 2023: Alice started painting", text)
        self.assertIn("Speaker Bob memories:", text)


class TestRunIndexCliMockWritesDump(unittest.TestCase):
    def test_run_mem0_index_mock_writes_speaker_json_without_gold(self):
        from src.locomo_eval.mem0.run_index import run_mem0_index

        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "locomo.json"
            data_path.write_text(json.dumps([MINI]), encoding="utf-8")
            out = Path(tmp) / "experiments"
            cfg = {
                "data": {"raw_path": str(data_path), "locomo_commit": "test"},
                "pipeline": {"memory": "mem0"},
                "reader": {"provider": "mock", "model": "gpt-4o-mini"},
                "mem0": {
                    "enable_graph": True,
                    "batch_size": 2,
                    "similar_s": 10,
                    "top_k": 30,
                    "extract": {"provider": "mock", "model": "gpt-4o-mini"},
                    "embed": {"provider": "mock", "model": "text-embedding-3-small"},
                },
                "run": {"output_dir": str(out), "run_id": "smoke_mem0_index"},
            }
            overrides = Namespace(
                data=str(data_path),
                run_id="smoke_mem0_index",
                output_dir=str(out),
                sample_id=None,
                max_samples=1,
                max_sessions=None,
                extractor="mock",
                embedder="mock",
                enable_graph=True,
                overwrite=False,
            )
            index_root = run_mem0_index(cfg, overrides)
            self.assertTrue((index_root / "by_sample" / "conv-mem0" / "speaker_a.json").is_file())
            self.assertTrue((index_root / "by_sample" / "conv-mem0" / "graph.json").is_file())
            log = (index_root / "by_sample" / "conv-mem0" / "ingest_log.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn(GOLD_LOCK, log)
            # resume skip
            again = run_mem0_index(cfg, overrides)
            meta = json.loads((again / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["n_skipped"], 1)


class TestDefaultYamlPinsMem0ReaderAndPrompt(unittest.TestCase):
    def test_memory_condition_yamls_share_gpt4o_mini_and_qa_mem0_v1(self):
        from src.config import load_config

        for rel in (
            "configs/baseline.yaml",
            "configs/raw_chunks.yaml",
            "configs/session_summaries.yaml",
            "configs/mem0.yaml",
            "configs/mem0g.yaml",
        ):
            cfg = load_config(ROOT / rel)
            self.assertEqual(cfg["reader"]["model"], "gpt-4o-mini", rel)
            self.assertEqual(cfg["pipeline"]["prompt_path"], "prompts/qa_mem0_v1.txt", rel)
        mem0 = load_config(ROOT / "configs/mem0.yaml")
        self.assertEqual(mem0["mem0"]["extract"]["model"], "gpt-4o-mini")
        self.assertEqual(int(mem0["mem0"]["top_k"]), 30)


if __name__ == "__main__":
    unittest.main()
