"""Model orchestrator and graph-memory builders.

Offline / mock only. Live provider pings are a CLI, not this file.
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
from src.locomo_eval.memory import (
    GRAPH,
    get_memory_builder,
    is_question_independent,
    resolve_memory_name,
)
from src.locomo_eval.model_orchestrator import (
    ModelOrchestrator,
    format_graph_memory_text,
    process_session_block,
)
from src.locomo_eval.writer_callers import _is_missing_anthropic_workspace
from src.locomo_eval.ping_writers import format_ping_error, ping_writers
from src.locomo_eval.preprocess.preprocessing_pipeline import PreprocessingPipeline

from src.locomo_eval.writer_model import MockWriter, get_writer

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


def _roster() -> list[MockWriter]:
    return [
        MockWriter(model_name="gpt-4o-mini", writer_id="openai"),
        MockWriter(model_name="claude-haiku-4-5", writer_id="anthropic"),
        MockWriter(model_name="deepseek-v4-flash", writer_id="deepseek"),
    ]




class TestEmptyOrchestratorStaysPassthrough(unittest.TestCase):
    def test_process_session_block_without_a_writer_keeps_passthrough_status(self):
        processed = _processed()
        record = process_session_block(processed.session_blocks[0])
        self.assertEqual(record["status"], "passthrough")
        orch = ModelOrchestrator()
        self.assertEqual(
            orch.process_session_block(processed.session_blocks[0]),
            record,
        )


class TestOrchestratorWritesLockedMem0Graph(unittest.TestCase):
    def test_graph_contains_taught_by_marker_for_that_writer(self):
        processed = _processed()
        orch = ModelOrchestrator(
            MockWriter(model_name="gpt-4o-mini", writer_id="openai"),
        )
        graph = orch.build_graph(processed)
        taught = [e for e in graph.edges if e.valid and e.relationship == "taught_by"]
        self.assertTrue(taught)
        self.assertEqual({e.target for e in taught}, {"openai"})
        self.assertTrue(any(e.relationship == "started" and e.target == "painting" for e in graph.edges if e.valid))

    def test_one_model_is_the_only_writer(self):
        orch = ModelOrchestrator(_roster()[0])
        self.assertEqual(orch.model.writer_id, "openai")

class TestOrchestratedGraphMemoryBuilder(unittest.TestCase):
    def test_resolve_memory_name_accepts_graph(self):
        self.assertEqual(resolve_memory_name("graph"), GRAPH)

    def test_graph_builder_is_question_independent_and_formats_edges(self):
        conv = parse_sample(MINI)
        orch = ModelOrchestrator(
            MockWriter(model_name="gpt-4o-mini", writer_id="openai"),
        )
        builder = get_memory_builder(GRAPH, orchestrator=orch)
        mem = builder.build(conv, conv.questions[0])
        self.assertEqual(mem.memory_type, GRAPH)
        self.assertIn("Graph relations:", mem.text)
        self.assertIn(" -- ", mem.text)
        self.assertTrue(is_question_independent(GRAPH))
        self.assertEqual(mem.writer_model, "gpt-4o-mini")

    def test_format_graph_memory_text_lists_valid_edges_only(self):
        processed = _processed()
        orch = ModelOrchestrator(
            MockWriter(model_name="gpt-4o-mini", writer_id="openai"),
        )
        graph = orch.build_graph(processed)
        from src.locomo_eval.mem0.retrieve import format_edge_line

        invalid_line = format_edge_line(graph.edges[0])
        graph.edges[0].valid = False
        text = format_graph_memory_text(graph, speaker_a="Alice", speaker_b="Bob")
        self.assertNotIn(invalid_line, text)
        self.assertIn("Graph relations:", text)


class TestGetWriterAndPing(unittest.TestCase):
    def test_get_writer_mock_records_the_requested_model_id(self):
        writer = get_writer("mock", model="claude-haiku-4-5", writer_id="anthropic")
        self.assertEqual(writer.model_name, "claude-haiku-4-5")
        self.assertEqual(writer.provider, "mock")
        self.assertEqual(writer.writer_id, "anthropic")

    def test_get_writer_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            get_writer("gemini", model="x")

    def test_mock_ping_writers_all_reply_pong(self):
        rows = ping_writers(["openai", "anthropic", "deepseek"], mock=True)
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


class TestWriterListIsRejected(unittest.TestCase):
    def test_writers_list_raises_before_any_model_call(self):
        from src.locomo_eval.run import _build_orchestrator, build_parser

        cfg = {
            "writers": [
                {"id": "openai", "provider": "openai", "model": "gpt-4o-mini"},
                {"id": "anthropic", "provider": "anthropic", "model": "claude-haiku-4-5"},
            ],
        }
        args = build_parser().parse_args(
            ["--config", str(ROOT / "configs" / "writers" / "graph.yaml")]
        )
        with self.assertRaises(SystemExit) as ctx:
            _build_orchestrator(cfg, args, "graph", "openai")
        self.assertIn("one writer model", str(ctx.exception))


class TestWriterThinkingMeta(unittest.TestCase):
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

    def test_mock_chat_writer_ping_forces_thinking_off(self):
        from src.locomo_eval.writer_model import ChatWriter

        writer = ChatWriter(provider="mock", model="mock", thinking=True)
        text, meta = writer.ping()
        self.assertEqual(text, "pong")
        self.assertFalse(meta["thinking"])
        self.assertEqual(meta.get("reasoning") or "", "")

    def test_mock_chat_writer_summarize_logs_thinking_when_on(self):
        from src.locomo_eval.writer_model import ChatWriter

        writer = ChatWriter(provider="mock", model="mock", thinking=True)
        _text, meta = writer.summarize_session(
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


class TestGoldNeverEntersGraphMemory(unittest.TestCase):
    def test_memory_text_omits_gold_answer_string(self):
        conv = parse_sample(MINI)
        gold = conv.questions[0].answer
        orch = ModelOrchestrator(
            MockWriter(model_name="gpt-4o-mini", writer_id="openai"),
        )
        mem = get_memory_builder(GRAPH, orchestrator=orch).build(
            conv, conv.questions[0]
        )
        self.assertNotIn(gold, mem.text)


class TestMockGraphRunWritesAuditPack(unittest.TestCase):
    def test_mock_graph_run_records_graph_memory_text(self):
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
                str(ROOT / "configs" / "writers" / "graph.yaml"),
                "--reader",
                "mock",
                "--writer",
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
            self.assertEqual(meta["memory_type"], GRAPH)
            self.assertNotIn("teacher_pool", meta)
            self.assertNotIn("teacher_fusion", meta)
            self.assertNotIn("n_teachers", meta)
            self.assertEqual(meta["writer_model"], "gpt-4o-mini")
            row = __import__("json").loads(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertIn("Graph relations:", row["memory_text"])
            self.assertNotIn(conv.questions[0].answer, row["memory_text"])
            self.assertTrue(meta["writer_thinking"])
            calls = [
                __import__("json").loads(line)
                for line in (run_dir / "memory" / "writer_calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertTrue(calls)
            self.assertTrue(calls[0]["thinking"])
            self.assertEqual(calls[0]["reasoning"], "[mock-thinking]")
            tid = calls[0]["writer_id"]
            self.assertTrue(tid)
            self.assertIsNotNone(calls[0].get("sample_id"))
            self.assertIn("entities", calls[0])
            self.assertIn("relations", calls[0])
            writer_dir = run_dir / "memory" / "writer"
            self.assertTrue((writer_dir / "calls.jsonl").is_file())
            self.assertTrue((writer_dir / "index.jsonl").is_file())
            self.assertFalse((writer_dir / "fusion.jsonl").exists())
            self.assertTrue((writer_dir / "by_writer" / tid / "calls.jsonl").is_file())
            self.assertTrue((run_dir / "reader" / "traces.jsonl").is_file())
            self.assertTrue((run_dir / "reader" / "predictions.jsonl").is_file())
            graph_index = (run_dir / "memory" / "graph" / "index.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertTrue(graph_index.strip())
            reader_trace = __import__("json").loads(
                (run_dir / "reader" / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(reader_trace["role"], "reader")
            self.assertIn("reasoning", reader_trace)
            self.assertEqual(meta["audit_layout"]["writer"], "memory/writer/")
            self.assertEqual(meta["audit_layout"]["version"], "audit_pack.v3")
            self.assertTrue((run_dir / "SUMMARY.md").is_file())
            self.assertTrue((run_dir / "ATTRIBUTION.md").is_file())
            self.assertTrue((run_dir / "attribution.jsonl").is_file())
            self.assertTrue((run_dir / "cost.json").is_file())
            self.assertTrue((run_dir / "config.resolved.yaml").is_file())
            self.assertTrue((run_dir / "memory" / "lineage.jsonl").is_file())
            self.assertTrue((run_dir / "memory" / "graph" / "ingest.jsonl").is_file())
            self.assertTrue((writer_dir / "sessions.jsonl").is_file())
            self.assertIn("audit of claims", (run_dir / "SUMMARY.md").read_text(encoding="utf-8"))
            attr_md = (run_dir / "ATTRIBUTION.md").read_text(encoding="utf-8")
            self.assertIn("LLM roles", attr_md)
            self.assertIn("graph", attr_md)


if __name__ == "__main__":
    unittest.main()
