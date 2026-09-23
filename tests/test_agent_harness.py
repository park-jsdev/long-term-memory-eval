"""Agent-harness protocol: workspace files, trajectory scoring, mock run (no API)."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.locomo_eval.agents import get_agent_runner, render_agent_prompt, resolve_codex_bin
from src.locomo_eval.agents.adapters.codex import (
    ANSWER_BASENAME,
    DISK_READ_PERMISSION,
    WEB_SEARCH_DISABLED,
    CodexAgentRunner,
    _codex_in_dir,
    _codex_subprocess_env,
)
from src.locomo_eval.agents.protocol import (
    EVENT_CATALOG,
    EVENT_ERROR,
    EVENT_MCP,
    EVENT_RETRIEVE,
    EVENT_WEB_SEARCH,
    EVENT_WRITE,
    AgentRequest,
    RetrievalEvent,
)
from src.locomo_eval.agents.comparison import (
    NOT_APPLICABLE,
    comparison_status,
    contract_sha256,
    resolve_contract,
    validate_strict_contract,
)
from src.locomo_eval.agents.trajectory import (
    build_trajectory,
    classify_event,
    events_from_codex_raw,
    failure_mode,
    parse_codex_jsonl,
    parse_structured_answer,
)
from src.locomo_eval.agents.workspace import (
    hide_session_files,
    ingest_structured_notes,
    snapshot_notes,
    write_conversation_workspace,
)
from src.locomo_eval.dataset import load_conversations
from src.locomo_eval.run import run_locomo_pipeline_with_memory_config
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.load_experiment_yaml import load_experiment_yaml

GOLD = "UNIQ_GOLD_REF_ZZZ"
MINI = [
    {
        "sample_id": "conv-agent",
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
                {"dia_id": "D2:1", "speaker": "Alice", "text": "I adopted a cat."},
            ],
        },
        "session_summary": {"session_1_summary": "Alice paints."},
        "qa": [
            {
                "question": "What hobby did Alice begin?",
                "answer": GOLD,
                "category": 4,
                "evidence": ["D1:1"],
            }
        ],
    }
]


def _write_mini(path: Path) -> None:
    path.write_text(json.dumps(MINI), encoding="utf-8")


def _overrides(data_path: Path, output_dir: Path, run_id: str) -> Namespace:
    return Namespace(
        data=str(data_path),
        memory=None,
        reader="mock",
        model="gpt-5",
        prompt=None,
        output_dir=str(output_dir),
        run_id=run_id,
        max_questions=1,
        sample_id=None,
        max_samples=1,
        agent="mock",
        agent_persist="off",
        agent_sessions="full",
        agent_tools="native",
    )


class TestWorkspaceDumpOmitsGold(unittest.TestCase):
    def test_session_files_include_dia_ids_and_exclude_gold_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "locomo.json"
            _write_mini(data)
            conv = load_conversations(data)[0]
            dest = Path(tmp) / "ws"
            manifest = write_conversation_workspace(conv, dest)
            index = (dest / "INDEX.md").read_text(encoding="utf-8")
            session = (dest / "sessions" / "session_1.md").read_text(encoding="utf-8")
            self.assertIn("D1:1", session)
            self.assertIn("Alice: I started painting.", session)
            self.assertNotIn(GOLD, session)
            self.assertNotIn(GOLD, index)
            self.assertNotIn(GOLD, manifest.text)
            self.assertIn("sessions/session_1.md", manifest.text)


class TestTrajectoryFailureModes(unittest.TestCase):
    def test_classify_index_read_as_catalog_and_session_read_as_retrieve(self):
        self.assertEqual(classify_event("read", "INDEX.md", "Sessions:"), EVENT_CATALOG)
        self.assertEqual(
            classify_event("read", "sessions/session_1.md", "Alice (D1:1)"),
            EVENT_RETRIEVE,
        )

    def test_classify_reasoning_and_errors_as_non_retrieval_events(self):
        self.assertEqual(classify_event("reasoning", "reasoning", "text"), EVENT_ERROR)
        self.assertEqual(classify_event("error", "error", ""), EVENT_ERROR)

    def test_classify_web_search_and_mcp_as_outside_workspace_tools(self):
        self.assertEqual(
            classify_event("web_search", "who is Alice", "Alice paints (D1:1)"),
            EVENT_WEB_SEARCH,
        )
        self.assertEqual(classify_event("mcp", "browser", ""), EVENT_MCP)

    def test_web_search_hit_does_not_count_as_evidence_retrieved(self):
        events = [
            RetrievalEvent(
                step=1,
                kind=EVENT_WEB_SEARCH,
                tool="web_search",
                target="Alice painting",
                retrieved_text="Alice: I started painting. (D1:1)",
                retrieved_tokens=8,
            )
        ]
        traj = build_trajectory(events, evidence_ids=["D1:1"], usage={})
        self.assertEqual(traj.n_web_search, 1)
        self.assertTrue(traj.used_non_workspace_tools)
        self.assertEqual(traj.n_retrieval_calls, 0)
        self.assertFalse(traj.evidence_retrieved)
        self.assertEqual(events[0].evidence_ids_hit, ["D1:1"])

    def test_evidence_hit_is_reasoning_failure_when_answer_is_wrong(self):
        events = [
            RetrievalEvent(
                step=1,
                kind=EVENT_RETRIEVE,
                tool="read",
                target="sessions/session_1.md",
                retrieved_text="Alice: I started painting. (D1:1)",
                retrieved_tokens=8,
            )
        ]
        traj = build_trajectory(events, evidence_ids=["D1:1"], usage={})
        self.assertTrue(traj.evidence_retrieved)
        self.assertEqual(traj.memory_recall, 1.0)
        self.assertEqual(traj.unnecessary_retrievals, 0)
        self.assertEqual(
            failure_mode(correct=False, evidence_retrieved=traj.evidence_retrieved),
            "reasoning_failure",
        )

    def test_missed_evidence_is_retrieval_failure_when_answer_is_wrong(self):
        events = [
            RetrievalEvent(
                step=1,
                kind=EVENT_RETRIEVE,
                tool="read",
                target="sessions/session_2.md",
                retrieved_text="Alice: I adopted a cat. (D2:1)",
                retrieved_tokens=8,
            )
        ]
        traj = build_trajectory(events, evidence_ids=["D1:1"], usage={})
        self.assertFalse(traj.evidence_retrieved)
        self.assertEqual(traj.memory_recall, 0.0)
        self.assertEqual(traj.unnecessary_retrievals, 1)
        self.assertEqual(
            failure_mode(correct=False, evidence_retrieved=traj.evidence_retrieved),
            "retrieval_failure",
        )

    def test_correct_without_evidence_is_parametric_success(self):
        self.assertEqual(
            failure_mode(correct=True, evidence_retrieved=False),
            "parametric_success",
        )


class TestCodexJsonlParse(unittest.TestCase):
    def test_parse_codex_jsonl_reads_usage_and_command_output(self):
        raw = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "t1"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item_1",
                            "type": "command_execution",
                            "command": "bash -lc 'cat sessions/session_1.md'",
                            "aggregated_output": "Alice: I started painting. (D1:1)",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item_2",
                            "type": "agent_message",
                            "text": '{"answer": "painting"}',
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 8,
                            "reasoning_output_tokens": 4,
                        },
                    }
                ),
            ]
        )
        events, usage, message = parse_codex_jsonl(raw)
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 8)
        self.assertEqual(usage["reasoning_tokens"], 4)
        self.assertEqual(parse_structured_answer(message), "painting")
        lifted = events_from_codex_raw(events)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(lifted[0].tool, "shell")
        self.assertIn("D1:1", lifted[0].retrieved_text)


class TestMockAgentRetrievesEvidenceFiles(unittest.TestCase):
    def test_mock_runner_reads_evidence_session_and_returns_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "locomo.json"
            _write_mini(data)
            conv = load_conversations(data)[0]
            dest = Path(tmp) / "ws"
            write_conversation_workspace(conv, dest)
            runner = get_agent_runner("mock", model="mock")
            result = runner.run(
                AgentRequest(
                    workspace_dir=dest,
                    sample_id=conv.sample_id,
                    question_id="q0",
                    question="What hobby did Alice begin?",
                    prompt=render_agent_prompt(
                        (ROOT / "prompts" / "agents" / "qa_workspace_v1.txt").read_text(
                            encoding="utf-8"
                        ),
                        "What hobby did Alice begin?",
                    ),
                    evidence_ids=["D1:1"],
                )
            )
            self.assertEqual(result.predicted_answer, "Unknown.")
            self.assertTrue(result.trajectory.evidence_retrieved)
            self.assertGreaterEqual(result.trajectory.n_retrieval_calls, 1)


class TestStubAdaptersFailLoudly(unittest.TestCase):
    def test_claude_code_adapter_is_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            get_agent_runner("claude_code")


class TestResolveCodexBin(unittest.TestCase):
    def test_explicit_path_wins_when_the_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "codex.exe"
            fake.write_bytes(b"fake")
            self.assertEqual(resolve_codex_bin(str(fake)), str(fake))

    def test_codex_bin_env_is_used_when_which_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "codex.exe"
            fake.write_bytes(b"fake")
            with mock.patch.dict(os.environ, {"CODEX_BIN": str(fake)}, clear=False):
                with mock.patch("src.locomo_eval.agents.adapters.codex.shutil.which", return_value=None):
                    self.assertEqual(resolve_codex_bin(), str(fake))

    def test_windows_install_dir_is_found_when_path_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "Programs" / "OpenAI" / "Codex" / "bin"
            bin_dir.mkdir(parents=True)
            fake = bin_dir / "codex.exe"
            fake.write_bytes(b"fake")
            with mock.patch.dict(
                os.environ, {"LOCALAPPDATA": str(tmp), "CODEX_BIN": ""}, clear=False
            ):
                with mock.patch(
                    "src.locomo_eval.agents.adapters.codex.shutil.which",
                    return_value=None,
                ):
                    with mock.patch(
                        "src.locomo_eval.agents.adapters.codex._windows_user_path_dirs",
                        return_value=[],
                    ):
                        self.assertEqual(resolve_codex_bin(), str(fake))

    def test_codex_in_dir_prefers_exe_on_windows_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "codex.exe").write_bytes(b"fake")
            self.assertEqual(_codex_in_dir(folder), folder / "codex.exe")


class TestCodexArgv(unittest.TestCase):
    def test_argv_passes_cd_and_absolute_last_message_path(self):
        runner = CodexAgentRunner(model_name="gpt-5", persist_memory=False)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "experiments" / "run" / "agent" / "workspaces" / "conv-26"
            workspace.mkdir(parents=True)
            argv = runner._argv("codex", workspace, "answer the question")
            root = workspace.resolve()
        self.assertIn("--cd", argv)
        self.assertEqual(argv[argv.index("--cd") + 1], str(root))
        last_message = argv[argv.index("-o") + 1]
        self.assertEqual(last_message, str(root / ANSWER_BASENAME))
        self.assertNotIn("--output-schema", argv)
        self.assertTrue(Path(last_message).is_absolute())
        self.assertFalse(last_message.startswith("experiments"))
        self.assertIn("--ephemeral", argv)
        self.assertIn("read-only", argv)
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ignore-rules", argv)
        self.assertIn(WEB_SEARCH_DISABLED, argv)
        self.assertIn(DISK_READ_PERMISSION, argv)
        self.assertNotIn("--search", argv)

    def test_persist_on_uses_workspace_write_and_skips_ephemeral(self):
        runner = CodexAgentRunner(model_name="gpt-5", persist_memory=True)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            argv = runner._argv("codex", workspace, "q")
        self.assertIn("workspace-write", argv)
        self.assertNotIn("--ephemeral", argv)
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ignore-rules", argv)
        self.assertIn(WEB_SEARCH_DISABLED, argv)
        self.assertIn(DISK_READ_PERMISSION, argv)

    def test_argv_drops_search_flag_from_extra_args(self):
        runner = CodexAgentRunner(
            model_name="gpt-5", extra_args=["--search", "--foo"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            argv = runner._argv("codex", Path(tmp), "q")
        self.assertNotIn("--search", argv)
        self.assertIn("--foo", argv)

    def test_subprocess_env_copies_openai_key_when_codex_key_missing(self):
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test-openai", "CODEX_API_KEY": ""},
            clear=False,
        ):
            env = _codex_subprocess_env()
        self.assertEqual(env["CODEX_API_KEY"], "sk-test-openai")

    def test_subprocess_env_keeps_explicit_codex_key(self):
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test-openai", "CODEX_API_KEY": "sk-codex"},
            clear=False,
        ):
            env = _codex_subprocess_env()
        self.assertEqual(env["CODEX_API_KEY"], "sk-codex")


class TestAgentConfigCompose(unittest.TestCase):
    def test_codex_preset_is_workspace_files_with_persist_off(self):
        cfg = load_config(ROOT / "configs" / "presets" / "agent_codex_gpt5.yaml")
        self.assertEqual(cfg["pipeline"]["memory"], "workspace_files")
        self.assertEqual(cfg["pipeline"]["answer_mode"], "agent")
        self.assertEqual(cfg["agent"]["adapter"], "codex")
        self.assertEqual(cfg["agent"]["model"], "gpt-5")
        self.assertFalse(cfg["agent"]["persist_memory"])
        self.assertEqual(cfg["agent"]["tools"], "native")
        self.assertEqual(cfg["pipeline"]["prompt_path"], "prompts/agents/qa_workspace_v1.txt")
        self.assertEqual(cfg["reader"]["model"], "gpt-5")


class TestComparisonContract(unittest.TestCase):
    def test_workspace_contract_marks_vector_controls_not_applicable(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.txt"
            prompt.write_text("Question: {question}", encoding="utf-8")
            contract = resolve_contract(
                config={"agent": {"comparison": {"strict": False}}},
                adapter="codex",
                model="gpt-5",
                model_snapshot=None,
                prompt_path=prompt,
                memory_type="workspace_files",
            )
        self.assertEqual(contract["retrieval"]["embedding_model"], NOT_APPLICABLE)
        self.assertEqual(contract["retrieval"]["k"], NOT_APPLICABLE)
        self.assertEqual(contract["memory_write"]["model"], NOT_APPLICABLE)
        self.assertEqual(contract["sha256"], contract_sha256(contract))

    def test_strict_rejects_codex_unenforceable_tool_budget(self):
        with self.assertRaisesRegex(ValueError, "cannot enforce"):
            validate_strict_contract(
                {
                    "strict": True,
                    "tool_budget": {
                        "allowed_tools": ["read"],
                        "max_tool_calls": 2,
                        "max_retrieved_tokens": 128,
                        "wall_clock_s": 60,
                    },
                },
                adapter="codex",
                model_snapshot="gpt-5-pinned",
            )

    def test_comparison_status_is_harness_failed_only_when_every_question_failed(self):
        contract = {"strict": False}
        self.assertEqual(
            comparison_status(
                contract, n_questions=100, n_harness_failed=2
            ),
            "incomparable",
        )
        self.assertEqual(
            comparison_status(
                contract, n_questions=100, n_harness_failed=100
            ),
            "harness_failed",
        )
        self.assertEqual(
            comparison_status({"strict": True}, n_questions=10, n_harness_failed=0),
            "comparable",
        )
        self.assertEqual(
            comparison_status(contract, harness_failed=True),
            "harness_failed",
        )


class TestAgentMatrixExpansion(unittest.TestCase):
    def test_poc_expands_to_one_mock_workspace_cell(self):
        specs = expand_run_matrix(
            load_experiment_yaml(ROOT / "configs" / "experiments" / "agent_codex_poc.yaml")
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].experiment_type, "agent")
        self.assertEqual(specs[0].agent, "mock")
        self.assertEqual(specs[0].memory_method, "workspace_files")
        self.assertFalse(specs[0].agent_persist)
        self.assertEqual(specs[0].agent_tools, "native")

    def test_gpt5_matrix_keeps_model_only_and_codex_cells(self):
        specs = expand_run_matrix(
            load_experiment_yaml(ROOT / "configs" / "experiments" / "agent_codex_gpt5.yaml")
        )
        self.assertEqual(len(specs), 2)
        pairs = {(s.agent, s.memory_method) for s in specs}
        self.assertEqual(
            pairs,
            {(None, "full_context"), ("codex", "workspace_files")},
        )
        none = next(s for s in specs if s.agent is None)
        self.assertEqual(none.prompt_path, "prompts/readers/qa_mem0_v1.txt")
        harness = next(s for s in specs if s.agent == "codex")
        self.assertEqual(harness.prompt_path, "prompts/agents/qa_workspace_v1.txt")


class TestMockAgentPipelineSmoke(unittest.TestCase):
    def test_workspace_files_mock_run_writes_agent_audit_and_failure_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "locomo.json"
            _write_mini(data)
            out = Path(tmp) / "experiments"
            cfg = load_config(ROOT / "configs" / "writers" / "workspace_files.yaml")
            cfg["data"]["raw_path"] = str(data)
            overrides = _overrides(data, out, "smoke_agent")
            with redirect_stdout(io.StringIO()):
                run_dir = run_locomo_pipeline_with_memory_config(cfg, overrides)
            self.assertTrue((run_dir / "predictions.jsonl").is_file())
            self.assertTrue((run_dir / "agent" / "trajectory.jsonl").is_file())
            self.assertTrue((run_dir / "agent" / "metrics.json").is_file())
            self.assertTrue((run_dir / "agent" / "workspaces" / "conv-agent" / "INDEX.md").is_file())
            pred = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(pred["predicted_answer"], "Unknown.")
            self.assertTrue(pred["evidence_retrieved"])
            self.assertEqual(pred["failure_mode"], "reasoning_failure")
            self.assertNotIn(GOLD, pred["memory_text"])
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["answer_mode"], "agent")
            self.assertEqual(meta["agent"], "mock")
            self.assertIn("agent", meta["audit_layout"])
            self.assertEqual(
                meta["comparison_contract"]["status"], "incomparable"
            )
            self.assertTrue(
                (run_dir / "agent" / "COMPARISON.md").is_file()
            )
            summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["agent"]["failure_modes"]["reasoning_failure"], 1)
            self.assertEqual(summary["agent"]["n_web_search_sum"], 0)
            self.assertEqual(summary["agent"]["n_mcp_sum"], 0)
            self.assertEqual(summary["agent"]["used_non_workspace_tools_rate"], 0.0)
            self.assertEqual(summary["agent"]["n_harness_failed"], 0)
            self.assertEqual(summary["agent"]["harness_failed_rate"], 0.0)
            self.assertFalse(pred["used_non_workspace_tools"])
            attr = json.loads(
                (run_dir / "attribution.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(attr["role"], "agent")
            self.assertEqual(attr["layer"], "read")


class TestPersistNotesAndIsolation(unittest.TestCase):
    def test_classify_notes_append_as_write_not_retrieve(self):
        self.assertEqual(
            classify_event("shell", ">> memory/notes.md", "- (D1:1) Alice: paints"),
            EVENT_WRITE,
        )
        self.assertEqual(
            classify_event("write", "memory/notes.md", "- (D1:1) Alice: paints"),
            EVENT_WRITE,
        )

    def test_hop_to_evidence_is_first_retrieve_step_with_gold_id(self):
        events = [
            RetrievalEvent(
                step=1, kind="catalog", tool="read", target="INDEX.md", retrieved_text="Sessions"
            ),
            RetrievalEvent(
                step=2,
                kind="retrieve",
                tool="read",
                target="sessions/session_2.md",
                retrieved_text="Alice adopted a cat (D2:1)",
            ),
            RetrievalEvent(
                step=3,
                kind="retrieve",
                tool="read",
                target="sessions/session_1.md",
                retrieved_text="Alice started painting (D1:1)",
            ),
        ]
        traj = build_trajectory(events, evidence_ids=["D1:1"])
        self.assertEqual(traj.hop_to_evidence, 3)
        self.assertFalse(traj.notes_retrieved)
        self.assertEqual(traj.n_write_events, 0)

    def test_persist_off_workspace_prompt_does_not_mention_notes(self):
        text = (ROOT / "prompts" / "agents" / "qa_workspace_v1.txt").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("notes.md", text.lower())
        self.assertIn("5-6 words", text)

    def test_notes_only_hides_sessions_and_keeps_dia_id_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "locomo.json"
            _write_mini(data)
            conv = load_conversations(data)[0]
            dest = Path(tmp) / "ws"
            write_conversation_workspace(conv, dest, persist_memory=True)
            ingest_structured_notes(conv, dest)
            hide_session_files(dest)
            self.assertFalse((dest / "sessions").is_dir())
            self.assertTrue((dest / "sessions_hidden" / "session_1.md").is_file())
            notes = (dest / "memory" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("D1:1", notes)
            self.assertIn("Alice: I started painting.", notes)
            index = (dest / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("memory/notes.md", index)
            self.assertNotIn("sessions/session_1.md", index)

    def test_snapshot_notes_records_growth_after_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ws"
            dest.mkdir()
            notes = dest / "memory"
            notes.mkdir()
            (notes / "notes.md").write_text("# Harness memory\n\n", encoding="utf-8")
            snap = Path(tmp) / "snaps"
            first = snapshot_notes(dest, snap, question_id="q1")
            (notes / "notes.md").write_text(
                "# Harness memory\n\n- (D1:1) Alice: paints\n", encoding="utf-8"
            )
            second = snapshot_notes(
                dest,
                snap,
                question_id="q2",
                previous_sha256=str(first["notes_sha256"]),
            )
            self.assertFalse(first["notes_grew"])
            self.assertTrue(second["notes_grew"])
            self.assertGreater(int(second["notes_bytes"]), int(first["notes_bytes"]))
            self.assertTrue((snap / "q2.md").is_file())

    def test_persist_on_mock_run_writes_notes_ledger_and_write_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "locomo.json"
            _write_mini(data)
            out = Path(tmp) / "experiments"
            cfg = load_config(ROOT / "configs" / "writers" / "workspace_files.yaml")
            cfg["data"]["raw_path"] = str(data)
            overrides = _overrides(data, out, "smoke_persist")
            overrides.agent_persist = "on"
            with redirect_stdout(io.StringIO()):
                run_dir = run_locomo_pipeline_with_memory_config(cfg, overrides)
            ledger = run_dir / "agent" / "notes_ledger.jsonl"
            self.assertTrue(ledger.is_file())
            rec = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
            self.assertGreater(int(rec["notes_bytes"]), 0)
            pred = json.loads(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertGreaterEqual(int(pred.get("n_write_events") or 0), 1)
            self.assertTrue(pred.get("notes_retrieved"))
            self.assertEqual(pred.get("agent_sessions"), "full")
            self.assertTrue(
                (run_dir / "agent" / "notes_snapshots" / "conv-agent" / "conv-agent-q-0.md").is_file()
            )

    def test_notes_only_mock_run_cannot_read_session_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "locomo.json"
            _write_mini(data)
            out = Path(tmp) / "experiments"
            cfg = load_config(ROOT / "configs" / "writers" / "workspace_files.yaml")
            cfg["data"]["raw_path"] = str(data)
            overrides = _overrides(data, out, "smoke_notes_only")
            overrides.agent_persist = "on"
            overrides.agent_sessions = "notes_only"
            with redirect_stdout(io.StringIO()):
                run_dir = run_locomo_pipeline_with_memory_config(cfg, overrides)
            ws = run_dir / "agent" / "workspaces" / "conv-agent"
            self.assertFalse((ws / "sessions").is_dir())
            self.assertTrue((ws / "memory" / "notes.md").is_file())
            pred = json.loads(
                (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(pred.get("agent_sessions"), "notes_only")
            self.assertTrue(pred.get("notes_retrieved"))
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta.get("agent_sessions"), "notes_only")
            self.assertIn("qa_workspace_notes_only_v1", meta.get("prompt_path") or "")


class TestPersistMemoryMatrix(unittest.TestCase):
    def test_persist_memory_expands_to_three_cells_and_rejects_off_notes_only(self):
        specs = expand_run_matrix(
            load_experiment_yaml(
                ROOT / "configs" / "experiments" / "openai_codex_persist_memory.yaml"
            )
        )
        self.assertEqual(len(specs), 3)
        pairs = {(s.agent_persist, s.agent_sessions, s.prompt_path) for s in specs}
        self.assertEqual(
            pairs,
            {
                (
                    False,
                    "full",
                    "prompts/agents/qa_workspace_v1.txt",
                ),
                (
                    True,
                    "full",
                    "prompts/agents/qa_workspace_persist_v1.txt",
                ),
                (
                    True,
                    "notes_only",
                    "prompts/agents/qa_workspace_notes_only_v1.txt",
                ),
            },
        )
        for spec in specs:
            self.assertEqual(spec.memory_method, "workspace_files")
            self.assertEqual(spec.agent, "codex")
            self.assertEqual(spec.reader.catalog_id, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
