"""Plan and score the Codex marker spikes. This module does not call the CLI.

The live runner is run_codex_hello_world.py. Tests import this file so a
flag cell or a hook judgment can be checked without GPT.

Cells a–d are the spec's hello world. Cell a repeats the flags the LoCoMo
adapter uses today. Cells b and c load project hooks. Cell d loads the
same hook from the plugin package.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

MARKER_ALPHA = "HARNESS_MARKER_ALPHA"
MARKER_BETA = "HARNESS_MARKER_BETA"

# Shared with the LoCoMo adapter. Kept here so a spike argv can be read
# next to the cell that adds or drops a flag.
WEB_SEARCH_DISABLED = 'web_search="disabled"'
DISK_READ_PERMISSION = 'sandbox_permissions=["disk-full-read-access"]'


@dataclass(frozen=True)
class HelloWorldCell:
    """One flag combination. hook_source is project or plugin."""

    cell_id: str
    ignore_user_config: bool
    ignore_rules: bool
    ephemeral: bool
    bypass_hook_trust: bool
    trust_project: bool
    enable_hooks: bool
    hook_source: str

    def renamed(self, cell_id: str) -> HelloWorldCell:
        """Same flags, new workspace name. Spike 2 reuses a passing cell."""
        return HelloWorldCell(
            cell_id=cell_id,
            ignore_user_config=self.ignore_user_config,
            ignore_rules=self.ignore_rules,
            ephemeral=self.ephemeral,
            bypass_hook_trust=self.bypass_hook_trust,
            trust_project=self.trust_project,
            enable_hooks=self.enable_hooks,
            hook_source=self.hook_source,
        )


def cell_from_mapping(raw: dict) -> HelloWorldCell:
    """Build a cell from configs/agents/hooks/hello_world.yaml."""
    source = str(raw.get("hook_source") or "project")
    if source not in ("project", "plugin"):
        raise ValueError(f"hook_source must be project or plugin, got {source!r}")
    return HelloWorldCell(
        cell_id=str(raw["id"]),
        ignore_user_config=bool(raw.get("ignore_user_config", False)),
        ignore_rules=bool(raw.get("ignore_rules", True)),
        ephemeral=bool(raw.get("ephemeral", False)),
        bypass_hook_trust=bool(raw.get("bypass_hook_trust", False)),
        trust_project=bool(raw.get("trust_project", False)),
        enable_hooks=bool(raw.get("enable_hooks", False)),
        hook_source=source,
    )


def load_cells(config: dict) -> list[HelloWorldCell]:
    rows = config.get("cells") or []
    if not rows:
        raise ValueError("hello_world config has no cells")
    return [cell_from_mapping(row) for row in rows]


def codex_project_key(workspace: Path) -> str:
    """Path Codex stores under [projects.'...'].

    On Windows the saved key is the resolved path, lowercased, with
    backslashes. A forward-slash or mixed-case key does not match, so
    the project layer stays untrusted and its hooks never load.
    """
    path = str(workspace.resolve())
    if os.name == "nt":
        return path.lower()
    return path


def trust_config_argument(workspace: Path) -> str:
    """One -c value. Single quotes: TOML literal, backslashes are not escapes."""
    key = codex_project_key(workspace)
    if "'" in key:
        raise ValueError(f"project path cannot contain a single quote: {key}")
    return f"projects.'{key}'.trust_level=\"trusted\""


def project_trust_toml(workspace: Path) -> str:
    """config.toml fragment matching Codex's own [projects] tables."""
    key = codex_project_key(workspace)
    return f"[projects.'{key}']\ntrust_level = \"trusted\"\n"


def enclosing_git_root(path: Path) -> Path | None:
    """Git root Codex will treat as the project, if this path sits inside one.

    Project hooks are read from that root's .codex/, not from a subdirectory.
    Spike workspaces have to live outside this repository.
    """
    current = path.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def build_exec_argv(
    binary: str,
    workspace: Path,
    prompt: str,
    model: str,
    cell: HelloWorldCell,
) -> list[str]:
    """Argv for one codex exec. The prompt is last, as the CLI requires."""
    root = workspace.resolve()
    argv = [
        binary,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--cd",
        str(root),
        "--sandbox",
        "read-only",
        "--model",
        model,
        "-o",
        str(root / "last_message.txt"),
        "-c",
        WEB_SEARCH_DISABLED,
        "-c",
        DISK_READ_PERMISSION,
    ]
    if cell.ignore_user_config:
        argv.append("--ignore-user-config")
    if cell.ignore_rules:
        argv.append("--ignore-rules")
    if cell.ephemeral:
        argv.append("--ephemeral")
    if cell.bypass_hook_trust:
        argv.append("--dangerously-bypass-hook-trust")
    if cell.trust_project:
        argv.extend(["-c", trust_config_argument(root)])
    if cell.enable_hooks:
        argv.extend(["--enable", "hooks"])
    argv.append(prompt)
    return argv


def command_line(python_exe: str, script: Path) -> str:
    """Command Codex runs. Quote only when a path contains spaces.

    Wrapping a space-free path in extra quotes makes Windows look for an
    executable whose name includes the quote characters, and the hook
    never starts.
    """
    py = Path(python_exe).resolve().as_posix()
    hook = Path(script).resolve().as_posix()
    if " " in py or " " in hook:
        return f'"{py}" "{hook}"'
    return f"{py} {hook}"


def render_hooks_document(python_exe: str, script: Path) -> str:
    """hooks.json whose command is this interpreter and the marker script.

    The committed plugin file uses ${PLUGIN_ROOT}. The spike rewrites a
    staged copy so the check does not depend on PATH or token expansion.
    """
    command = command_line(python_exe, script)
    handler = {
        "type": "command",
        "command": command,
        "commandWindows": command,
        "timeout": 30,
        "statusMessage": "Harness marker hook",
    }
    document = {
        "description": "Hello-world marker hook staged with an absolute Python.",
        "hooks": {
            "UserPromptSubmit": [{"hooks": [handler]}],
            "Stop": [{"hooks": [{**handler, "statusMessage": "Harness capture hook"}]}],
        },
    }
    return json.dumps(document, indent=2) + "\n"


def prepare_workspace(
    root: Path,
    *,
    python_exe: str,
    script: Path,
    hook_source: str,
    policy: dict | None,
) -> None:
    """Create the workspace Codex will use as --cd.

    Project cells get .codex/hooks.json. Plugin cells must not, or a pass
    could come from the project file instead of the plugin.
    """
    root.mkdir(parents=True, exist_ok=True)
    memory = root / "memory"
    memory.mkdir(exist_ok=True)
    if policy is not None:
        (memory / "policy.json").write_text(
            json.dumps(policy, indent=2) + "\n", encoding="utf-8"
        )
    hooks_path = root / ".codex" / "hooks.json"
    if hook_source == "project":
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text(render_hooks_document(python_exe, script), encoding="utf-8")
    elif hooks_path.exists():
        hooks_path.unlink()


def read_hook_actions(log_path: Path) -> list[str]:
    """Actions recorded by marker_hook.py, in order."""
    if not log_path.is_file():
        return []
    actions: list[str] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("action"):
            actions.append(str(row["action"]))
    return actions


def judge_injection(actions: list[str], final_message: str, token: str) -> str:
    """Shared pass rule from the spec.

    pass — the hook logged inject and the model echoed the token.
    hook_ran_context_missing — the hook ran and the model did not see it.
    contamination — the token showed up with no inject log.
    absent — neither side happened.
    """
    logged = "inject" in actions
    seen = token in (final_message or "")
    if logged and seen:
        return "pass"
    if logged and not seen:
        return "hook_ran_context_missing"
    if seen and not logged:
        return "contamination"
    return "absent"


def judge_abstain(actions: list[str], final_message: str, token: str) -> str:
    """Abstain passes when the hook logged it and the token stayed out."""
    logged = "abstain" in actions
    seen = token in (final_message or "")
    if logged and not seen:
        return "pass"
    if logged and seen:
        return "abstain_logged_but_token_seen"
    if seen and not logged:
        return "contamination"
    return "absent"


def judge_capture(actions: list[str], capture_text: str) -> str:
    """Stop passes when the hook logged capture and wrote a line."""
    if "capture" in actions and "captured " in (capture_text or ""):
        return "pass"
    if "capture" in actions:
        return "capture_logged_but_file_missing"
    if "captured " in (capture_text or ""):
        return "contamination"
    return "absent"


def feature_line(features_text: str, name: str) -> str:
    """One row from `codex features list`, or a marker that it was absent."""
    prefix = name + " "
    for line in features_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix) or stripped.startswith(name + "\t"):
            return stripped
    return f"{name}: not listed"
