"""Codex hook flags and the marker hook. No Codex binary and no API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness.codex.spikes.codex_hello_world import (
    MARKER_ALPHA,
    MARKER_BETA,
    HelloWorldCell,
    build_exec_argv,
    cell_from_mapping,
    feature_line,
    judge_abstain,
    judge_capture,
    judge_injection,
    load_cells,
    prepare_workspace,
    trust_config_argument,
)

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "harness" / "codex" / "plugin" / "hooks" / "marker_hook.py"
CONFIG = ROOT / "configs" / "agents" / "hooks" / "hello_world.yaml"


def _cell(**overrides) -> HelloWorldCell:
    raw = {
        "id": "b_project_hooks",
        "ignore_user_config": False,
        "ignore_rules": True,
        "ephemeral": True,
        "bypass_hook_trust": True,
        "trust_project": True,
        "enable_hooks": True,
        "hook_source": "project",
    }
    raw.update(overrides)
    return cell_from_mapping(raw)


def _run_hook(event: dict, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    child_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(event),
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=str(cwd),
        env=child_env,
        check=False,
    )


class TestBuildExecArgv(unittest.TestCase):
    def test_current_adapter_cell_includes_ignore_user_config_and_ephemeral(self):
        cell = _cell(
            id="a_current_adapter",
            ignore_user_config=True,
            ephemeral=True,
            bypass_hook_trust=False,
            trust_project=False,
            enable_hooks=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            argv = build_exec_argv("codex", Path(tmp), "prompt", "gpt-4o-mini", cell)
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ignore-rules", argv)
        self.assertIn("--ephemeral", argv)
        self.assertNotIn("--dangerously-bypass-hook-trust", argv)
        self.assertNotIn("--enable", argv)
        self.assertEqual(argv[-1], "prompt")

    def test_project_hooks_cell_omits_ignore_user_config_and_bypasses_hook_trust(self):
        cell = _cell(ephemeral=False)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            argv = build_exec_argv("codex", workspace, "prompt", "gpt-4o-mini", cell)
        self.assertNotIn("--ignore-user-config", argv)
        self.assertIn("--ignore-rules", argv)
        self.assertNotIn("--ephemeral", argv)
        self.assertIn("--dangerously-bypass-hook-trust", argv)
        self.assertIn("--enable", argv)
        self.assertEqual(argv[argv.index("--enable") + 1], "hooks")
        projects = next(item for item in argv if str(item).startswith("projects."))
        self.assertIn("trust_level", projects)
        self.assertIn("projects.'", projects)


class TestTrustConfigArgument(unittest.TestCase):
    def test_trust_config_argument_uses_codex_project_key_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            value = trust_config_argument(workspace)
        self.assertIn("trust_level=\"trusted\"", value)
        self.assertIn("projects.'", value)
        if os.name == "nt":
            self.assertIn(str(workspace.resolve()).lower(), value)
            self.assertNotIn("/", value.split("projects.'", 1)[1].split("'", 1)[0])


class TestLoadCells(unittest.TestCase):
    def test_load_cells_returns_four_spec_cells_from_hello_world_yaml(self):
        import yaml

        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cells = load_cells(config)
        self.assertEqual(
            [cell.cell_id for cell in cells],
            [
                "a_current_adapter",
                "b_project_hooks",
                "c_project_hooks_no_ephemeral",
                "d_plugin",
            ],
        )
        self.assertTrue(cells[0].ignore_user_config)
        self.assertFalse(cells[1].ignore_user_config)
        self.assertEqual(cells[3].hook_source, "plugin")


class TestPrepareWorkspace(unittest.TestCase):
    def test_prepare_workspace_writes_project_hooks_and_skips_them_for_plugin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            plugin = root / "plugin"
            prepare_workspace(
                project,
                python_exe=sys.executable,
                script=HOOK,
                hook_source="project",
                policy={"action": "abstain"},
            )
            prepare_workspace(
                plugin,
                python_exe=sys.executable,
                script=HOOK,
                hook_source="plugin",
                policy=None,
            )
            hooks = json.loads((project / ".codex" / "hooks.json").read_text(encoding="utf-8"))
            command = hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
            self.assertIn("UserPromptSubmit", hooks["hooks"])
            self.assertIn("Stop", hooks["hooks"])
            self.assertIn(HOOK.resolve().as_posix(), command)
            self.assertNotIn('"', command)
            policy = json.loads((project / "memory" / "policy.json").read_text(encoding="utf-8"))
            self.assertEqual(policy["action"], "abstain")
            self.assertFalse((plugin / ".codex" / "hooks.json").exists())


class TestJudgeInjection(unittest.TestCase):
    def test_judge_injection_returns_pass_when_hook_logged_inject_and_message_contains_token(self):
        self.assertEqual(
            judge_injection(["inject"], f"SEEN {MARKER_ALPHA}", MARKER_ALPHA),
            "pass",
        )

    def test_judge_injection_returns_hook_ran_context_missing_when_log_has_no_echo(self):
        self.assertEqual(
            judge_injection(["inject"], "MISSING", MARKER_ALPHA),
            "hook_ran_context_missing",
        )

    def test_judge_injection_returns_contamination_when_token_appears_without_a_log(self):
        self.assertEqual(
            judge_injection([], f"SEEN {MARKER_ALPHA}", MARKER_ALPHA),
            "contamination",
        )

    def test_judge_injection_returns_absent_when_hook_and_message_are_empty(self):
        self.assertEqual(judge_injection([], "MISSING", MARKER_ALPHA), "absent")


class TestJudgeAbstain(unittest.TestCase):
    def test_judge_abstain_returns_pass_when_hook_abstains_and_token_is_absent(self):
        self.assertEqual(judge_abstain(["abstain"], "MISSING", MARKER_BETA), "pass")

    def test_judge_abstain_returns_abstain_logged_but_token_seen_when_token_leaks(self):
        self.assertEqual(
            judge_abstain(["abstain"], f"SEEN {MARKER_BETA}", MARKER_BETA),
            "abstain_logged_but_token_seen",
        )


class TestJudgeCapture(unittest.TestCase):
    def test_judge_capture_returns_pass_when_stop_logged_and_capture_line_exists(self):
        self.assertEqual(
            judge_capture(["capture"], "captured turn_id=1 prompt_chars=3\n"),
            "pass",
        )

    def test_judge_capture_returns_absent_when_stop_did_not_run(self):
        self.assertEqual(judge_capture(["inject"], ""), "absent")


class TestFeatureLine(unittest.TestCase):
    def test_feature_line_returns_the_hooks_row_and_not_plugin_hooks(self):
        text = "plugin_hooks  removed  false\nhooks  stable  true\n"
        self.assertTrue(feature_line(text, "hooks").startswith("hooks"))
        self.assertIn("removed", feature_line(text, "plugin_hooks"))


class TestMarkerHookInjectsDefault(unittest.TestCase):
    def test_marker_hook_injects_default_marker_when_policy_file_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            completed = _run_hook({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}, cwd)
            body = json.loads(completed.stdout)
            log = json.loads((cwd / "hook_log.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(
            body["hookSpecificOutput"]["additionalContext"],
            MARKER_ALPHA,
        )
        self.assertEqual(log["action"], "inject")


class TestMarkerHookAbstain(unittest.TestCase):
    def test_marker_hook_returns_empty_object_when_policy_action_is_abstain(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            memory = cwd / "memory"
            memory.mkdir()
            (memory / "policy.json").write_text('{"action": "abstain"}\n', encoding="utf-8")
            completed = _run_hook({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}, cwd)
            body = json.loads(completed.stdout)
            log = json.loads((cwd / "hook_log.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(body, {})
        self.assertEqual(log["action"], "abstain")
        self.assertNotIn(MARKER_BETA, completed.stdout)


class TestMarkerHookInjectsPolicyText(unittest.TestCase):
    def test_marker_hook_injects_policy_text_when_action_is_inject(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            memory = cwd / "memory"
            memory.mkdir()
            (memory / "policy.json").write_text(
                json.dumps({"action": "inject", "text": MARKER_BETA}),
                encoding="utf-8",
            )
            completed = _run_hook({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}, cwd)
            body = json.loads(completed.stdout)
        self.assertEqual(body["hookSpecificOutput"]["additionalContext"], MARKER_BETA)
        self.assertEqual(body["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit")


class TestMarkerHookStop(unittest.TestCase):
    def test_marker_hook_appends_capture_line_and_does_not_continue_on_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            completed = _run_hook(
                {"hook_event_name": "Stop", "turn_id": "t1", "last_assistant_message": "ok"},
                cwd,
            )
            body = json.loads(completed.stdout)
            capture = (cwd / "memory" / "capture.log").read_text(encoding="utf-8")
        self.assertEqual(body, {})
        self.assertNotIn("decision", completed.stdout)
        self.assertIn("captured turn_id=t1", capture)


class TestMarkerHookLogsDecisionFields(unittest.TestCase):
    def test_marker_hook_logs_condition_and_selected_ids_from_the_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            memory = cwd / "memory"
            memory.mkdir()
            (memory / "policy.json").write_text(
                json.dumps(
                    {
                        "action": "inject",
                        "text": "Stage 2 instruction",
                        "condition": "C",
                        "selected_ids": ["hyp-low-maint"],
                    }
                ),
                encoding="utf-8",
            )
            completed = _run_hook({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}, cwd)
            body = json.loads(completed.stdout)
            log = json.loads((cwd / "hook_log.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(body["hookSpecificOutput"]["additionalContext"], "Stage 2 instruction")
        self.assertEqual(log["condition"], "C")
        self.assertEqual(log["selected_ids"], ["hyp-low-maint"])
