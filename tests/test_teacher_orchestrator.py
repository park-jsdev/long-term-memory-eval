"""Teacher orchestrator, pooling, fusion, and graph-memory builders.

Offline / mock only. Live OpenAI / Anthropic / DeepSeek pings are a CLI
(``python -m src.locomo_eval.ping_teachers``), not this file.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.dataset import parse_sample
from src.locomo_eval.fusion import (
    FUSION_MAJORITY,
    FUSION_NONE,
    FUSION_RESOLVE_CONFIDENCE,
    FUSION_RESOLVE_FIRST,
    FUSION_RESOLVE_RANDOM,
    FUSION_RESOLVE_ROUND_ROBIN,
    FUSION_RESOLVE_TOP_VOTED,
    POOL_EQUAL_WEIGHT,
    POOL_RANDOM,
    POOL_ROUND_ROBIN,
    POOL_SINGLE,
    GraphProposal,
    fuse_majority,
    fuse_proposals,
    fuse_resolve,
    fusion_relation_audit,
    fusion_slot_audit,
    relation_key,
    select_teacher_ids,
)
from src.locomo_eval.memory import (
    FUSED_TEACHER_GRAPH,
    POOLED_TEACHER_GRAPH,
    TEACHER_GRAPH,
    get_memory_builder,
    is_question_independent,
    resolve_memory_name,
)
from src.locomo_eval.teacher_callers import _is_missing_anthropic_workspace
from src.locomo_eval.ping_teachers import format_ping_error, ping_teachers
from src.locomo_eval.preprocess.preprocessing_pipeline import PreprocessingPipeline
from src.locomo_eval.teacher_orchestrator import (
    TeacherOrchestrator,
    format_graph_memory_text,
    iter_session_blocks,
    process_session_block,
)
from src.locomo_eval.teachers import MockTeacher, get_teacher

MINI = {
    "sample_id": "conv-orch",
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
            {"dia_id": "D2:1", "speaker": "Alice", "text": "I got a nursing job in Boston."},
        ],
        "session_3_date_time": "3 Jan 2023",
        "session_3": [
            {"dia_id": "D3:1", "speaker": "Alice", "text": "I love pizza."},
        ],
    },
    "session_summary": {
        "session_1_summary": "Alice started painting.",
        "session_2_summary": "Alice got a nursing job.",
        "session_3_summary": "Alice loves pizza.",
    },
    "qa": [
        {
            "question": "What hobby did Alice begin?",
            "answer": "SECRET_GOLD_NOT_IN_DIALOG",
            "category": 4,
            "evidence": ["D1:1"],
        },
    ],
}


def _processed():
    return PreprocessingPipeline().process(parse_sample(MINI))


def _roster() -> list[MockTeacher]:
    return [
        MockTeacher(model_name="gpt-4o-mini", teacher_id="openai"),
        MockTeacher(model_name="claude-haiku-4-5", teacher_id="anthropic"),
        MockTeacher(model_name="deepseek-v4-flash", teacher_id="deepseek"),
    ]


class TestSelectTeacherIdsPoolPolicies(unittest.TestCase):
    def test_single_returns_only_the_first_teacher_id(self):
        ids = select_teacher_ids(
            ["openai", "anthropic", "deepseek"],
            pool=POOL_SINGLE,
            session_index=2,
            rng=__import__("random").Random(0),
        )
        self.assertEqual(ids, ["openai"])

    def test_round_robin_cycles_teacher_ids_across_sessions(self):
        rng = __import__("random").Random(0)
        roster = ["openai", "anthropic", "deepseek"]
        picked = [
            select_teacher_ids(roster, pool=POOL_ROUND_ROBIN, session_index=i, rng=rng)[0]
            for i in range(3)
        ]
        self.assertEqual(picked, ["openai", "anthropic", "deepseek"])

    def test_equal_weight_returns_every_teacher_id(self):
        ids = select_teacher_ids(
            ["openai", "anthropic", "deepseek"],
            pool=POOL_EQUAL_WEIGHT,
            session_index=0,
            rng=__import__("random").Random(0),
        )
        self.assertEqual(ids, ["openai", "anthropic", "deepseek"])

    def test_random_pool_is_deterministic_with_a_seed(self):
        roster = ["openai", "anthropic", "deepseek"]
        a = select_teacher_ids(
            roster, pool=POOL_RANDOM, session_index=0, rng=__import__("random").Random(7)
        )
        b = select_teacher_ids(
            roster, pool=POOL_RANDOM, session_index=0, rng=__import__("random").Random(7)
        )
        self.assertEqual(a, b)
        self.assertEqual(len(a), 1)
        self.assertIn(a[0], roster)


class TestFusionResolveStrategies(unittest.TestCase):
    def _disagreement_proposals(self) -> list[GraphProposal]:
        return [
            GraphProposal(
                teacher_id="openai",
                provider="mock",
                model="gpt-4o-mini",
                relations=[{"source": "alice", "relationship": "lives_in", "target": "boston"}],
            ),
            GraphProposal(
                teacher_id="anthropic",
                provider="mock",
                model="claude-haiku-4-5",
                relations=[{"source": "alice", "relationship": "lives_in", "target": "nyc"}],
            ),
            GraphProposal(
                teacher_id="deepseek",
                provider="mock",
                model="deepseek-v4-flash",
                relations=[{"source": "alice", "relationship": "lives_in", "target": "la"}],
            ),
        ]

    def test_resolve_top_voted_picks_two_vote_target_on_one_two_split(self):
        proposals = [
            GraphProposal(
                teacher_id="openai",
                provider="mock",
                model="a",
                relations=[{"source": "alice", "relationship": "lives_in", "target": "boston"}],
            ),
            GraphProposal(
                teacher_id="anthropic",
                provider="mock",
                model="b",
                relations=[{"source": "alice", "relationship": "lives_in", "target": "boston"}],
            ),
            GraphProposal(
                teacher_id="deepseek",
                provider="mock",
                model="c",
                relations=[{"source": "alice", "relationship": "lives_in", "target": "nyc"}],
            ),
        ]
        _ents, rels = fuse_resolve(proposals, strategy=FUSION_RESOLVE_TOP_VOTED)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["target"], "boston")

    def test_resolve_first_picks_first_teacher_target_on_one_one_one_split(self):
        proposals = self._disagreement_proposals()
        _ents, rels = fuse_resolve(proposals, strategy=FUSION_RESOLVE_FIRST)
        self.assertEqual(rels[0]["target"], "boston")

    def test_resolve_random_is_seeded(self):
        proposals = self._disagreement_proposals()
        rng = __import__("random").Random(42)
        a = fuse_resolve(proposals, strategy=FUSION_RESOLVE_RANDOM, rng=rng)[1]
        rng = __import__("random").Random(42)
        b = fuse_resolve(proposals, strategy=FUSION_RESOLVE_RANDOM, rng=rng)[1]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 1)

    def test_resolve_round_robin_varies_by_session_index_on_ties(self):
        proposals = self._disagreement_proposals()
        a = fuse_resolve(
            proposals, strategy=FUSION_RESOLVE_ROUND_ROBIN, session_index=0
        )[1][0]["target"]
        b = fuse_resolve(
            proposals, strategy=FUSION_RESOLVE_ROUND_ROBIN, session_index=1
        )[1][0]["target"]
        self.assertNotEqual(a, b)

    def test_resolve_confidence_beats_two_low_confidence_votes(self):
        proposals = [
            GraphProposal(
                teacher_id="openai",
                provider="mock",
                model="a",
                relations=[
                    {
                        "source": "alice",
                        "relationship": "lives_in",
                        "target": "boston",
                        "confidence": 3.0,
                    }
                ],
            ),
            GraphProposal(
                teacher_id="anthropic",
                provider="mock",
                model="b",
                relations=[
                    {
                        "source": "alice",
                        "relationship": "lives_in",
                        "target": "nyc",
                        "confidence": 0.5,
                    }
                ],
            ),
            GraphProposal(
                teacher_id="deepseek",
                provider="mock",
                model="c",
                relations=[
                    {
                        "source": "alice",
                        "relationship": "lives_in",
                        "target": "nyc",
                        "confidence": 0.5,
                    }
                ],
            ),
        ]
        _ents, rels = fuse_resolve(proposals, strategy=FUSION_RESOLVE_CONFIDENCE)
        self.assertEqual(rels[0]["target"], "boston")

    def test_fusion_slot_audit_marks_disagreements(self):
        proposals = self._disagreement_proposals()
        _ents, kept = fuse_resolve(proposals, strategy=FUSION_RESOLVE_FIRST)
        rows = fusion_slot_audit(proposals, kept)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["disagreement"])
        self.assertEqual(rows[0]["n_targets"], 3)
        self.assertIn(rows[0]["kept_target"], {"boston", "nyc", "la"})


class TestFuseMajorityKeepsSharedTriples(unittest.TestCase):
    def test_majority_vote_keeps_triples_seen_twice_and_drops_singletons(self):
        shared = {"source": "alice", "relationship": "started", "target": "painting"}
        proposals = [
            GraphProposal(
                teacher_id="openai",
                provider="mock",
                model="gpt-4o-mini",
                relations=[shared, {"source": "alice", "relationship": "taught_by", "target": "openai"}],
            ),
            GraphProposal(
                teacher_id="anthropic",
                provider="mock",
                model="claude-haiku-4-5",
                relations=[shared, {"source": "alice", "relationship": "taught_by", "target": "anthropic"}],
            ),
            GraphProposal(
                teacher_id="deepseek",
                provider="mock",
                model="deepseek-v4-flash",
                relations=[{"source": "alice", "relationship": "taught_by", "target": "deepseek"}],
            ),
        ]
        _ents, rels = fuse_majority(proposals)
        keys = {relation_key(r) for r in rels}
        self.assertIn(("alice", "started", "painting"), keys)
        self.assertNotIn(("alice", "taught_by", "openai"), keys)


class TestFusionRelationAuditAttributesTeachers(unittest.TestCase):
    def test_fusion_relation_audit_marks_kept_and_lists_proposing_teachers(self):
        shared = {"source": "alice", "relationship": "started", "target": "painting"}
        proposals = [
            GraphProposal(
                teacher_id="openai",
                provider="mock",
                model="gpt-4o-mini",
                relations=[shared],
            ),
            GraphProposal(
                teacher_id="anthropic",
                provider="mock",
                model="claude-haiku-4-5",
                relations=[shared],
            ),
        ]
        _ents, kept = fuse_majority(proposals)
        rows = fusion_relation_audit(proposals, kept)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["proposed_by"], ["openai", "anthropic"])
        self.assertEqual(rows[0]["votes"], 2)
        self.assertTrue(rows[0]["kept"])

    def test_fusion_none_unions_all_teacher_triples(self):
        proposals = [
            GraphProposal(
                teacher_id="a",
                provider="mock",
                model="a",
                relations=[{"source": "alice", "relationship": "taught_by", "target": "a"}],
            ),
            GraphProposal(
                teacher_id="b",
                provider="mock",
                model="b",
                relations=[{"source": "alice", "relationship": "taught_by", "target": "b"}],
            ),
        ]
        _ents, rels = fuse_proposals(proposals, fusion=FUSION_NONE)
        targets = {r["target"] for r in rels}
        self.assertEqual(targets, {"a", "b"})


class TestEmptyOrchestratorStaysPassthrough(unittest.TestCase):
    def test_process_session_block_without_teachers_keeps_passthrough_status(self):
        processed = _processed()
        record = process_session_block(processed.session_blocks[0])
        self.assertEqual(record["status"], "passthrough")
        orch = TeacherOrchestrator()
        self.assertEqual(
            orch.process_session_block(processed.session_blocks[0]),
            record,
        )


class TestOrchestratorWritesLockedMem0Graph(unittest.TestCase):
    def test_single_teacher_graph_contains_taught_by_marker_for_that_teacher(self):
        processed = _processed()
        orch = TeacherOrchestrator(
            [MockTeacher(model_name="gpt-4o-mini", teacher_id="openai")],
            pool=POOL_SINGLE,
            fusion=FUSION_NONE,
        )
        graph = orch.build_graph(processed)
        taught = [e for e in graph.edges if e.valid and e.relationship == "taught_by"]
        self.assertTrue(taught)
        self.assertEqual({e.target for e in taught}, {"openai"})
        self.assertTrue(any(e.relationship == "started" and e.target == "painting" for e in graph.edges if e.valid))

    def test_equal_weight_pool_unions_all_three_taught_by_markers(self):
        processed = _processed()
        orch = TeacherOrchestrator(_roster(), pool=POOL_EQUAL_WEIGHT, fusion=FUSION_NONE)
        graph = orch.build_graph(processed)
        taught = {e.target for e in graph.edges if e.valid and e.relationship == "taught_by"}
        self.assertEqual(taught, {"openai", "anthropic", "deepseek"})

    def test_round_robin_pool_uses_one_teacher_marker_per_session(self):
        processed = _processed()
        self.assertEqual(len(list(iter_session_blocks(processed))), 3)
        orch = TeacherOrchestrator(_roster(), pool=POOL_ROUND_ROBIN, fusion=FUSION_NONE)
        graph = orch.build_graph(processed)
        taught = [e for e in graph.edges if e.valid and e.relationship == "taught_by"]
        self.assertEqual({e.target for e in taught}, {"openai", "anthropic", "deepseek"})
        self.assertEqual(len(taught), 3)

    def test_majority_fusion_drops_per_teacher_markers_and_keeps_shared_facts(self):
        processed = _processed()
        orch = TeacherOrchestrator(
            _roster(),
            pool=POOL_EQUAL_WEIGHT,
            fusion=FUSION_MAJORITY,
        )
        graph = orch.build_graph(processed)
        taught = [e for e in graph.edges if e.valid and e.relationship == "taught_by"]
        self.assertEqual(taught, [])
        valid = [(e.source, e.relationship, e.target) for e in graph.edges if e.valid]
        self.assertIn(("alice", "started", "painting"), valid)
        self.assertIn(("alice", "works_as", "nurse"), valid)


class TestOrchestratedGraphMemoryBuilder(unittest.TestCase):
    def test_resolve_memory_name_accepts_teacher_graph_condition_ids(self):
        self.assertEqual(resolve_memory_name("teacher_graph"), TEACHER_GRAPH)
        self.assertEqual(resolve_memory_name("pooled_teacher_graph"), POOLED_TEACHER_GRAPH)
        self.assertEqual(resolve_memory_name("fused_teacher_graph"), FUSED_TEACHER_GRAPH)

    def test_teacher_graph_builder_is_question_independent_and_formats_edges(self):
        conv = parse_sample(MINI)
        orch = TeacherOrchestrator(
            [MockTeacher(model_name="gpt-4o-mini", teacher_id="openai")],
            pool=POOL_SINGLE,
            fusion=FUSION_NONE,
        )
        builder = get_memory_builder(TEACHER_GRAPH, orchestrator=orch)
        mem = builder.build(conv, conv.questions[0])
        self.assertEqual(mem.memory_type, TEACHER_GRAPH)
        self.assertIn("Graph relations:", mem.text)
        self.assertIn(" -- ", mem.text)
        self.assertTrue(is_question_independent(TEACHER_GRAPH))
        self.assertEqual(mem.teacher_model, "gpt-4o-mini")

    def test_format_graph_memory_text_lists_valid_edges_only(self):
        processed = _processed()
        orch = TeacherOrchestrator(
            [MockTeacher(model_name="gpt-4o-mini", teacher_id="openai")],
            pool=POOL_SINGLE,
        )
        graph = orch.build_graph(processed)
        from src.locomo_eval.mem0.retrieve import format_edge_line

        invalid_line = format_edge_line(graph.edges[0])
        graph.edges[0].valid = False
        text = format_graph_memory_text(graph, speaker_a="Alice", speaker_b="Bob")
        self.assertNotIn(invalid_line, text)
        self.assertIn("Graph relations:", text)


class TestGetTeacherProvidersAndPing(unittest.TestCase):
    def test_get_teacher_mock_anthropic_alias_records_claude_model_id(self):
        teacher = get_teacher("mock", model="claude-haiku-4-5", teacher_id="anthropic")
        self.assertEqual(teacher.model_name, "claude-haiku-4-5")
        self.assertEqual(teacher.provider, "mock")
        self.assertEqual(teacher.teacher_id, "anthropic")

    def test_get_teacher_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            get_teacher("gemini", model="x")

    def test_mock_ping_teachers_all_reply_pong(self):
        rows = ping_teachers(["openai", "anthropic", "deepseek"], mock=True)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["ok"] for row in rows))
        self.assertTrue(all(row["text"] == "pong" for row in rows))

    def test_format_ping_error_uses_repr_when_str_is_empty(self):
        class Blank(Exception):
            def __str__(self) -> str:
                return ""

        text = format_ping_error(Blank())
        self.assertIn("Blank", text)

    def test_missing_anthropic_workspace_message_is_detected(self):
        exc = RuntimeError(
            "anthropic-workspace-id is required when authenticating with an identity-linked API key"
        )
        self.assertTrue(_is_missing_anthropic_workspace(exc))
        self.assertFalse(_is_missing_anthropic_workspace(RuntimeError("rate limit")))


class TestLiveTeacherRosterWiring(unittest.TestCase):
    def _fake_teacher(self, provider, model=None, **kwargs):
        return MockTeacher(
            model_name=model or "mock",
            teacher_id=kwargs.get("teacher_id") or provider,
        )

    def test_pooled_yaml_passes_openai_anthropic_deepseek_when_reader_is_live(self):
        from src.config import load_config
        from src.locomo_eval.run import _build_orchestrator, build_parser

        cfg = load_config(ROOT / "configs" / "pooled_teacher_graph.yaml")
        args = build_parser().parse_args(
            ["--config", str(ROOT / "configs" / "pooled_teacher_graph.yaml")]
        )
        seen = []

        def fake_get_teacher(provider, model=None, **kwargs):
            seen.append(provider)
            return self._fake_teacher(provider, model=model, **kwargs)

        with patch("src.locomo_eval.run.get_teacher", side_effect=fake_get_teacher):
            orch = _build_orchestrator(cfg, args, "pooled_teacher_graph", "openai")
        self.assertEqual(seen, ["openai", "anthropic", "deepseek"])
        self.assertEqual(orch.pool, "equal_weight")
        self.assertEqual(orch.fusion, "none")

    def test_fused_resolve_yaml_keeps_live_roster_and_resolve_policy(self):
        from src.config import load_config
        from src.locomo_eval.run import _build_orchestrator, build_parser

        cfg = load_config(ROOT / "configs" / "fused_teacher_graph_resolve_top_voted.yaml")
        args = build_parser().parse_args(
            [
                "--config",
                str(ROOT / "configs" / "fused_teacher_graph_resolve_top_voted.yaml"),
            ]
        )
        seen = []

        def fake_get_teacher(provider, model=None, **kwargs):
            seen.append(provider)
            return self._fake_teacher(provider, model=model, **kwargs)

        with patch("src.locomo_eval.run.get_teacher", side_effect=fake_get_teacher):
            orch = _build_orchestrator(cfg, args, "fused_teacher_graph", "openai")
        self.assertEqual(seen, ["openai", "anthropic", "deepseek"])
        self.assertEqual(orch.fusion, "resolve_top_voted")

    def test_mock_reader_forces_mock_teachers_on_pooled_roster(self):
        from src.config import load_config
        from src.locomo_eval.run import _build_orchestrator, build_parser

        cfg = load_config(ROOT / "configs" / "pooled_teacher_graph.yaml")
        args = build_parser().parse_args(
            [
                "--config",
                str(ROOT / "configs" / "pooled_teacher_graph.yaml"),
                "--reader",
                "mock",
            ]
        )
        seen = []

        def fake_get_teacher(provider, model=None, **kwargs):
            seen.append(provider)
            return self._fake_teacher(provider, model=model, **kwargs)

        with patch("src.locomo_eval.run.get_teacher", side_effect=fake_get_teacher):
            _build_orchestrator(cfg, args, "pooled_teacher_graph", "mock")
        self.assertEqual(seen, ["mock", "mock", "mock"])


class TestTeacherThinkingMeta(unittest.TestCase):
    def test_openai_reasoning_text_reads_reasoning_content(self):
        from src.locomo_eval.reasoning_extractor import (
            openai_reasoning_text,
            openai_reasoning_tokens,
        )

        class Message:
            reasoning_content = "step one"

        self.assertEqual(openai_reasoning_text(Message()), "step one")

        class Usage:
            class Details:
                reasoning_tokens = 12

            completion_tokens_details = Details()

        self.assertEqual(openai_reasoning_tokens(Usage()), 12)

    def test_split_anthropic_content_separates_thinking_blocks(self):
        from src.locomo_eval.reasoning_extractor import split_anthropic_content

        class Block:
            def __init__(self, type: str, **kwargs):
                self.type = type
                for key, val in kwargs.items():
                    setattr(self, key, val)

        text, thought = split_anthropic_content(
            [
                Block("thinking", thinking="plan"),
                Block("text", text="answer"),
            ]
        )
        self.assertEqual(text, "answer")
        self.assertEqual(thought, "plan")

    def test_mock_chat_teacher_ping_forces_thinking_off(self):
        from src.locomo_eval.teachers import ChatTeacher

        teacher = ChatTeacher(provider="mock", model="mock", thinking=True)
        text, meta = teacher.ping()
        self.assertEqual(text, "pong")
        self.assertFalse(meta["thinking"])
        self.assertEqual(meta.get("reasoning") or "", "")

    def test_mock_chat_teacher_summarize_logs_thinking_when_on(self):
        from src.locomo_eval.teachers import ChatTeacher

        teacher = ChatTeacher(provider="mock", model="mock", thinking=True)
        _text, meta = teacher.summarize_session(
            session_text="Alice paints.",
            date_time="1 Jan 2023",
            speaker_a="Alice",
            speaker_b="Bob",
        )
        self.assertTrue(meta["thinking"])
        self.assertEqual(meta["reasoning"], "[mock-thinking]")

    def test_apply_openai_thinking_false_forces_luna_none(self):
        from src.locomo_eval.models import (
            GPT56_LUNA,
            apply_openai_thinking,
            chat_create_kwargs,
            resolve_model,
        )

        spec = resolve_model(GPT56_LUNA)
        kwargs = chat_create_kwargs(
            spec,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=64,
        )
        out = apply_openai_thinking(spec, kwargs, False)
        self.assertEqual(out.get("reasoning_effort"), "none")


class TestGoldNeverEntersTeacherGraph(unittest.TestCase):
    def test_memory_text_omits_gold_answer_string(self):
        conv = parse_sample(MINI)
        gold = conv.questions[0].answer
        orch = TeacherOrchestrator(_roster(), pool=POOL_EQUAL_WEIGHT, fusion=FUSION_MAJORITY)
        mem = get_memory_builder(FUSED_TEACHER_GRAPH, orchestrator=orch).build(
            conv, conv.questions[0]
        )
        self.assertNotIn(gold, mem.text)


class TestMockPipelineTeacherGraphWritesAuditPack(unittest.TestCase):
    def test_mock_teacher_graph_run_records_pool_and_graph_memory_text(self):
        import io
        from contextlib import redirect_stdout

        from src.config import load_config
        from src.locomo_eval.run import build_parser, run_locomo_pipeline_with_memory_config

        conv = parse_sample(MINI)
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "locomo.json"
            data_path.write_text(
                __import__("json").dumps([MINI]),
                encoding="utf-8",
            )
            out = Path(tmp) / "experiments"
            argv = [
                "--config",
                str(ROOT / "configs" / "teacher_graph.yaml"),
                "--reader",
                "mock",
                "--teacher",
                "mock",
                "--data",
                str(data_path),
                "--output-dir",
                str(out),
                "--run-id",
                "smoke_tg",
                "--max-questions",
                "1",
            ]
            args = build_parser().parse_args(argv)
            cfg = load_config(args.config)
            cfg["data"]["raw_path"] = str(data_path)
            with redirect_stdout(io.StringIO()):
                run_dir = run_locomo_pipeline_with_memory_config(cfg, args)
            meta = __import__("json").loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["memory_type"], TEACHER_GRAPH)
            self.assertEqual(meta["teacher_pool"], "single")
            self.assertEqual(meta["teacher_fusion"], "none")
            row = __import__("json").loads(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertIn("Graph relations:", row["memory_text"])
            self.assertNotIn(conv.questions[0].answer, row["memory_text"])
            self.assertTrue(meta["teacher_thinking"])
            calls = [
                __import__("json").loads(line)
                for line in (run_dir / "memory" / "teacher_calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertTrue(calls)
            self.assertTrue(calls[0]["thinking"])
            self.assertEqual(calls[0]["reasoning"], "[mock-thinking]")
            tid = calls[0]["teacher_id"]
            self.assertTrue(tid)
            self.assertIsNotNone(calls[0].get("sample_id"))
            self.assertIn("entities", calls[0])
            self.assertIn("relations", calls[0])
            teachers_dir = run_dir / "memory" / "teachers"
            self.assertTrue((teachers_dir / "calls.jsonl").is_file())
            self.assertTrue((teachers_dir / "index.jsonl").is_file())
            self.assertTrue((teachers_dir / "fusion.jsonl").is_file())
            self.assertTrue((teachers_dir / "by_teacher" / tid / "calls.jsonl").is_file())
            self.assertTrue((run_dir / "reader" / "traces.jsonl").is_file())
            self.assertTrue((run_dir / "reader" / "predictions.jsonl").is_file())
            graph_index = (run_dir / "memory" / "graph" / "index.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertTrue(graph_index.strip())
            fusion_row = __import__("json").loads(
                (teachers_dir / "fusion.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertIn("relations", fusion_row)
            self.assertEqual(fusion_row["teacher_ids"], [tid])
            reader_trace = __import__("json").loads(
                (run_dir / "reader" / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(reader_trace["role"], "reader")
            self.assertIn("reasoning", reader_trace)
            self.assertEqual(meta["audit_layout"]["teachers"], "memory/teachers/")


if __name__ == "__main__":
    unittest.main()
