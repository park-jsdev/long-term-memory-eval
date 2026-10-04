"""One writer model, or none, on each memory YAML. Offline / mock only."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.locomo_eval.run import (
    _build_orchestrator,
    _one_writer,
    build_parser,
    run_locomo_pipeline_with_memory_config,
)
from src.locomo_eval.writer_model import MockWriter

WRITERS = ROOT / "configs" / "writers"
WITH_WRITER = {
    "graph.yaml",
    "agent_codex_mem0_facts.yaml",
}
OVERLAYS = {"openai_mini.yaml"}
WITHOUT_WRITER = {
    "raw_chunks.yaml",
    "session_summaries.yaml",
    "full_context.yaml",
    "rag.yaml",
    "openai_memory.yaml",
    "mem0.yaml",
    "mem0g.yaml",
    "workspace_files.yaml",
    "preprocess.yaml",
}
MINI = [
    {
        "sample_id": "conv-eval",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1 Jan 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Alice", "text": "I started painting."},
                {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice!"},
            ],
        },
        "session_summary": {"session_1_summary": "Alice paints."},
        "qa": [
            {
                "question": "What hobby did Alice begin?",
                "answer": "painting",
                "category": 4,
                "evidence": ["D1:1"],
            }
        ],
    }
]


def _writer_model(cfg: dict) -> str | None:
    writer = cfg.get("writer") or {}
    model = writer.get("model")
    return str(model) if model else None


class TestWriterYamlIsOneModelOrNone(unittest.TestCase):
    def test_every_writer_yaml_is_one_model_or_no_writer(self):
        found = {path.name for path in WRITERS.glob("*.yaml")}
        self.assertEqual(found, WITH_WRITER | OVERLAYS | WITHOUT_WRITER)
        for name in WITH_WRITER:
            cfg = load_config(WRITERS / name)
            self.assertTrue(_writer_model(cfg), name)
            self.assertNotIn("writers", cfg)
        overlay = load_config(WRITERS / "openai_mini.yaml")
        self.assertEqual(overlay["writer"]["model"], "gpt-4o-mini")
        self.assertNotIn("pipeline", overlay)
        for name in WITHOUT_WRITER:
            cfg = load_config(WRITERS / name)
            self.assertIsNone(_writer_model(cfg), name)

    def test_mem0_extract_names_one_model(self):
        for name in ("mem0.yaml", "mem0g.yaml"):
            cfg = load_config(WRITERS / name)
            extract = (cfg.get("mem0") or {}).get("extract") or {}
            self.assertIsInstance(extract.get("model"), str)
            self.assertTrue(extract["model"])


class TestOneWriterRuntime(unittest.TestCase):
    def test_graph_yaml_builds_one_writer(self):
        cfg = load_config(WRITERS / "graph.yaml")
        args = build_parser().parse_args(["--config", str(WRITERS / "graph.yaml")])
        seen = []

        def fake_get_writer(provider, model=None, **kwargs):
            seen.append((provider, model))
            return MockWriter(model_name=model or "mock", writer_id=provider)

        with patch("src.locomo_eval.run.get_writer", side_effect=fake_get_writer):
            orch = _build_orchestrator(cfg, args, "graph", "openai")
        self.assertEqual(seen, [("openai", "gpt-4o-mini")])
        self.assertEqual(orch.model.model_name, "gpt-4o-mini")

    def test_writer_model_flag_overlays_the_one_model(self):
        cfg = load_config(WRITERS / "graph.yaml")
        args = build_parser().parse_args(
            ["--config", str(WRITERS / "graph.yaml"), "--writer-model", "gpt-5"]
        )
        slot = _one_writer(cfg, args, "openai")
        self.assertEqual(slot["model"], "gpt-5")

    def test_writers_list_is_rejected(self):
        cfg = {
            "writers": [
                {"id": "a", "provider": "openai", "model": "gpt-4o-mini"},
                {"id": "b", "provider": "openai", "model": "gpt-5"},
            ]
        }
        args = build_parser().parse_args(["--config", str(WRITERS / "graph.yaml")])
        with self.assertRaises(SystemExit):
            _one_writer(cfg, args, "openai")

    def test_mock_graph_run_records_one_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "locomo.json"
            data_path.write_text(json.dumps(MINI), encoding="utf-8")
            out = Path(tmp) / "experiments"
            args = build_parser().parse_args(
                [
                    "--config",
                    str(WRITERS / "graph.yaml"),
                    "--reader",
                    "mock",
                    "--writer",
                    "mock",
                    "--data",
                    str(data_path),
                    "--output-dir",
                    str(out),
                    "--run-id",
                    "smoke_one_writer",
                    "--max-questions",
                    "1",
                ]
            )
            cfg = load_config(args.config)
            cfg["data"]["raw_path"] = str(data_path)
            with redirect_stdout(io.StringIO()):
                run_dir = run_locomo_pipeline_with_memory_config(cfg, args)
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["writer_model"], "gpt-4o-mini")
            self.assertEqual(meta["memory_type"], "graph")
            calls = [
                json.loads(line)
                for line in (run_dir / "memory" / "writer_calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual({row["writer_id"] for row in calls}, {"mock"})


if __name__ == "__main__":
    unittest.main()
