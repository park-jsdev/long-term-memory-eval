"""Run the Codex hello-world spikes against the local CLI.

Spike 0 records the CLI version and feature flags. Spikes 1–3 call the
model. Every exec uses a disposable CODEX_HOME so the user's ~/.codex
config and hooks stay untouched. Auth is an API key when one is set;
otherwise auth.json is copied and config.toml is not.

    python harness/codex/spikes/run_codex_hello_world.py --spike 0
    python harness/codex/spikes/run_codex_hello_world.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.codex.spikes.codex_hello_world import (  # noqa: E402
    MARKER_ALPHA,
    MARKER_BETA,
    HelloWorldCell,
    build_exec_argv,
    enclosing_git_root,
    feature_line,
    judge_abstain,
    judge_capture,
    judge_injection,
    load_cells,
    prepare_workspace,
    project_trust_toml,
    read_hook_actions,
    render_hooks_document,
)
from src.locomo_eval.agents.adapters.codex import (  # noqa: E402
    _codex_subprocess_env,
    resolve_codex_bin,
)
from src.locomo_eval.agents.trajectory import parse_codex_jsonl  # noqa: E402
from src.locomo_eval.env import load_env  # noqa: E402

PLUGIN_ID = "harness-marker@harness-spikes"
DEFAULT_CONFIG = ROOT / "configs" / "agents" / "hooks" / "hello_world.yaml"
HOOK_SCRIPT = ROOT / "harness" / "codex" / "plugin" / "hooks" / "marker_hook.py"
PLUGIN_SRC = ROOT / "harness" / "codex" / "plugin"
VERSION_PATH = ROOT / "harness" / "codex" / "VERSION"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run(argv: list[str], env: dict[str, str], timeout: float) -> subprocess.CompletedProcess[str]:
    # DEVNULL: a piped stdin makes `codex exec` append that stream to the
    # prompt ("Reading additional input from stdin...") even when the
    # prompt was already passed as an argument.
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def _user_codex_home() -> Path:
    override = (os.environ.get("CODEX_HOME") or "").strip()
    if override:
        return Path(override)
    return Path.home() / ".codex"


def write_project_trust(codex_home: Path, workspace: Path) -> None:
    """Trust this workspace in the disposable home.

    -c can set the same key, but Codex's on-disk form is a [projects]
    table. Writing the table makes the key visible in the run pack.
    """
    dest = codex_home / "config.toml"
    existing = dest.read_text(encoding="utf-8") if dest.is_file() else ""
    fragment = project_trust_toml(workspace)
    if fragment.strip() in existing:
        return
    dest.write_text(existing + ("\n" if existing else "") + fragment, encoding="utf-8")


def prepare_codex_home(dest: Path) -> str:
    """Empty config home. Copy auth.json only when no API key is set.

    A copied login token plus CODEX_API_KEY has 401'd in this repo.
    config.toml is never copied: it would load the user's real hooks.
    """
    dest.mkdir(parents=True, exist_ok=True)
    env = _codex_subprocess_env()
    if (env.get("CODEX_API_KEY") or "").strip():
        return "api_key"
    source = _user_codex_home() / "auth.json"
    if source.is_file():
        shutil.copyfile(source, dest / "auth.json")
        return "copied_auth_json"
    return "missing"


def child_env(codex_home: Path, *, marker: str, hook_log: Path, capture_log: Path) -> dict[str, str]:
    env = _codex_subprocess_env()
    env["CODEX_HOME"] = str(codex_home)
    env["HARNESS_MARKER"] = marker
    env["HARNESS_HOOK_LOG"] = str(hook_log)
    env["HARNESS_CAPTURE_LOG"] = str(capture_log)
    return env


def stage_plugin(dest_root: Path, python_exe: str) -> Path:
    """Copy the plugin and point its hooks at this interpreter.

    The staged marketplace root is what `codex plugin marketplace add` reads.
    """
    plugin_dest = dest_root / "plugin"
    if plugin_dest.exists():
        shutil.rmtree(plugin_dest)
    shutil.copytree(PLUGIN_SRC, plugin_dest)
    script = plugin_dest / "hooks" / "marker_hook.py"
    (plugin_dest / "hooks" / "hooks.json").write_text(
        render_hooks_document(python_exe, script), encoding="utf-8"
    )
    marketplace = dest_root / ".agents" / "plugins"
    marketplace.mkdir(parents=True, exist_ok=True)
    catalog = {
        "name": "harness-spikes",
        "interface": {"displayName": "Harness spikes"},
        "plugins": [
            {
                "name": "harness-marker",
                "source": {"source": "local", "path": "./plugin"},
                "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
                "category": "Productivity",
            }
        ],
    }
    (marketplace / "marketplace.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    return dest_root


def _append_plugin_enable(config_path: Path) -> None:
    """Record that this disposable home should load the marker plugin.

    A second [features] table would make Codex reject config.toml, so the
    hooks flag is added only when the file does not already have one.
    """
    text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    if PLUGIN_ID in text:
        return
    extra = ""
    if "[features]" not in text:
        extra += "\n[features]\nhooks = true\n"
    extra += f'\n[plugins."{PLUGIN_ID}"]\nenabled = true\n'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text + extra, encoding="utf-8")


def install_plugin(binary: str, env: dict[str, str], marketplace_root: Path, timeout: float) -> str:
    """Register the staged marketplace and install harness-marker. Returns a note."""
    notes: list[str] = []
    add = _run(
        [binary, "plugin", "marketplace", "add", str(marketplace_root), "--json"],
        env,
        timeout,
    )
    notes.append(f"marketplace add exit {add.returncode}: {(add.stdout or add.stderr)[-500:]}")
    if add.returncode != 0:
        return "\n".join(notes)
    added = _run(
        [binary, "plugin", "add", PLUGIN_ID, "--json"],
        env,
        timeout,
    )
    notes.append(f"plugin add exit {added.returncode}: {(added.stdout or added.stderr)[-500:]}")
    _append_plugin_enable(Path(env["CODEX_HOME"]) / "config.toml")
    return "\n".join(notes)


def final_message(workspace: Path, stdout: str) -> str:
    path = workspace / "last_message.txt"
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    try:
        _raw, _usage, last = parse_codex_jsonl(stdout)
    except (ValueError, json.JSONDecodeError, TypeError):
        return ""
    return (last or "").strip()


def write_version(binary: str, version: str, features: str) -> None:
    lines = [
        f"codex_bin: {binary}",
        f"version: {version.strip()}",
        f"hooks: {feature_line(features, 'hooks')}",
        f"plugin_hooks: {feature_line(features, 'plugin_hooks')}",
        f"plugins: {feature_line(features, 'plugins')}",
        f"memories: {feature_line(features, 'memories')}",
        f"recorded_at: {_utc_stamp()}",
        "",
    ]
    VERSION_PATH.write_text("\n".join(lines), encoding="utf-8")


def spike_pin(binary: str, env: dict[str, str], run_dir: Path) -> dict:
    version = _run([binary, "--version"], env, 30)
    features = _run([binary, "features", "list"], env, 30)
    version_text = (version.stdout or version.stderr or "").strip()
    features_text = features.stdout or ""
    (run_dir / "features.txt").write_text(features_text, encoding="utf-8")
    write_version(binary, version_text, features_text)
    return {
        "spike": 0,
        "cell": "pin",
        "verdict": "pass" if version.returncode == 0 else "absent",
        "detail": version_text,
        "hooks": feature_line(features_text, "hooks"),
        "plugin_hooks": feature_line(features_text, "plugin_hooks"),
    }


def execute_cell(
    *,
    binary: str,
    run_dir: Path,
    cell: HelloWorldCell,
    prompt: str,
    model: str,
    timeout: float,
    policy: dict | None,
    token: str,
    judge: str,
) -> dict:
    """One exec. judge is injection, abstain, or capture."""
    # Outside the git repo. Codex binds project hooks to the git root,
    # so a workspace under agent_memory would ignore this .codex/.
    workspace = Path(tempfile.mkdtemp(prefix=f"harness-{cell.cell_id}-"))
    if enclosing_git_root(workspace) is not None:
        raise RuntimeError(f"spike workspace is inside a git repo: {workspace}")
    prepare_workspace(
        workspace,
        python_exe=sys.executable,
        script=HOOK_SCRIPT,
        hook_source=cell.hook_source,
        policy=policy,
    )
    codex_home = run_dir / "codex_homes" / cell.cell_id
    if codex_home.exists():
        shutil.rmtree(codex_home)
    auth_mode = prepare_codex_home(codex_home)
    if cell.trust_project:
        write_project_trust(codex_home, workspace)
    hook_log = run_dir / "logs" / f"{cell.cell_id}.jsonl"
    capture_log = run_dir / "logs" / f"{cell.cell_id}.capture.txt"
    hook_log.parent.mkdir(parents=True, exist_ok=True)
    env = child_env(codex_home, marker=MARKER_ALPHA, hook_log=hook_log, capture_log=capture_log)
    plugin_note = ""
    if cell.hook_source == "plugin":
        staged = stage_plugin(run_dir / "plugin_stage" / cell.cell_id, sys.executable)
        plugin_note = install_plugin(binary, env, staged, timeout)
    argv = build_exec_argv(binary, workspace, prompt, model, cell)
    try:
        completed = _run(argv, env, timeout)
        stdout, stderr, code = completed.stdout or "", completed.stderr or "", completed.returncode
        error = ""
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        code = -1
        error = f"timeout after {timeout}s"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / f"{cell.cell_id}.stdout.jsonl").write_text(stdout, encoding="utf-8")
    (run_dir / "logs" / f"{cell.cell_id}.stderr.txt").write_text(stderr, encoding="utf-8")
    message = final_message(workspace, stdout)
    actions = read_hook_actions(hook_log)
    if not actions:
        actions = read_hook_actions(workspace / "hook_log.jsonl")
    capture_text = capture_log.read_text(encoding="utf-8") if capture_log.is_file() else ""
    if not capture_text:
        workspace_capture = workspace / "memory" / "capture.log"
        if workspace_capture.is_file():
            capture_text = workspace_capture.read_text(encoding="utf-8")
    if judge == "abstain":
        verdict = judge_abstain(actions, message, token)
    elif judge == "capture":
        verdict = judge_capture(actions, capture_text)
    else:
        verdict = judge_injection(actions, message, token)
    return {
        "spike": judge,
        "cell": cell.cell_id,
        "verdict": verdict,
        "actions": actions,
        "message": message[:500],
        "returncode": code,
        "auth": auth_mode,
        "error": error,
        "stderr_tail": stderr[-800:],
        "plugin_note": plugin_note[-800:],
        "token": token,
        "workspace": str(workspace),
    }


def render_report(rows: list[dict], run_dir: Path) -> str:
    lines = [
        "# Codex hello-world results",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "| Spike | Cell | Verdict | Actions | Final message |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        actions = ", ".join(row.get("actions") or []) or "-"
        message = (row.get("message") or row.get("detail") or "").replace("\n", " ")
        lines.append(
            f"| {row.get('spike')} | {row.get('cell')} | {row.get('verdict')} | {actions} | {message} |"
        )
    lines.append("")
    for row in rows:
        if row.get("error") or row.get("stderr_tail") or row.get("plugin_note"):
            lines.append(f"## {row.get('cell')}")
            if row.get("error"):
                lines.append(f"Error: {row['error']}")
            if row.get("plugin_note"):
                lines.append("")
                lines.append("```")
                lines.append(str(row["plugin_note"]))
                lines.append("```")
            if row.get("stderr_tail"):
                lines.append("")
                lines.append("```")
                lines.append(str(row["stderr_tail"]))
                lines.append("```")
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Codex hello-world harness spikes")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--spike", choices=("0", "1", "2", "3", "all"), default="all")
    parser.add_argument("--cell", default="", help="Run only this cell id (spikes 1–3)")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=float, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Write workspaces and argv, do not call Codex")
    args = parser.parse_args(argv)
    load_env()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    cells = load_cells(config)
    if args.cell:
        cells = [cell for cell in cells if cell.cell_id == args.cell]
        if not cells:
            raise SystemExit(f"no cell named {args.cell}")
    model = args.model or str(config.get("model") or "gpt-4o-mini")
    timeout = args.timeout or float(config.get("timeout_s") or 120)
    prompt = str(config.get("prompt") or "").strip()
    binary = resolve_codex_bin(None)
    run_dir = ROOT / "harness" / "codex" / "spikes" / "runs" / _utc_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if binary is None:
        report = "codex binary not found\n"
        (run_dir / "RESULTS.md").write_text(report, encoding="utf-8")
        print(report)
        return 1
    base_env = _codex_subprocess_env()
    which = args.spike
    if which in ("0", "all"):
        rows.append(spike_pin(binary, base_env, run_dir))
    if args.dry_run:
        for cell in cells:
            workspace = run_dir / "workspaces" / cell.cell_id
            prepare_workspace(
                workspace,
                python_exe=sys.executable,
                script=HOOK_SCRIPT,
                hook_source=cell.hook_source,
                policy=None,
            )
            planned = build_exec_argv(binary, workspace, prompt, model, cell)
            (workspace / "argv.json").write_text(json.dumps(planned, indent=2), encoding="utf-8")
            rows.append({"spike": "dry-run", "cell": cell.cell_id, "verdict": "planned", "message": " ".join(planned)})
    elif which in ("1", "all"):
        for cell in cells:
            rows.append(
                execute_cell(
                    binary=binary,
                    run_dir=run_dir,
                    cell=cell,
                    prompt=prompt,
                    model=model,
                    timeout=timeout,
                    policy=None,
                    token=MARKER_ALPHA,
                    judge="injection",
                )
            )
    if not args.dry_run and which in ("2", "3", "all"):
        passed = next((row for row in rows if row.get("verdict") == "pass" and row.get("spike") == "injection"), None)
        chosen = None
        if passed:
            chosen = next((cell for cell in cells if cell.cell_id == passed["cell"]), None)
        if chosen is None:
            chosen = next((cell for cell in load_cells(config) if cell.cell_id == "b_project_hooks"), cells[0])
        if which in ("2", "all"):
            rows.append(
                execute_cell(
                    binary=binary,
                    run_dir=run_dir,
                    cell=chosen.renamed(chosen.cell_id + "_abstain"),
                    prompt=prompt,
                    model=model,
                    timeout=timeout,
                    policy={"action": "abstain"},
                    token=MARKER_BETA,
                    judge="abstain",
                )
            )
            rows.append(
                execute_cell(
                    binary=binary,
                    run_dir=run_dir,
                    cell=chosen.renamed(chosen.cell_id + "_inject"),
                    prompt=prompt,
                    model=model,
                    timeout=timeout,
                    policy={"action": "inject", "text": MARKER_BETA},
                    token=MARKER_BETA,
                    judge="injection",
                )
            )
        if which in ("3", "all"):
            rows.append(
                execute_cell(
                    binary=binary,
                    run_dir=run_dir,
                    cell=chosen.renamed(chosen.cell_id + "_capture"),
                    prompt=prompt,
                    model=model,
                    timeout=timeout,
                    policy={"action": "inject", "text": MARKER_BETA},
                    token=MARKER_BETA,
                    judge="capture",
                )
            )
    report = render_report(rows, run_dir)
    (run_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
