"""Codex CLI adapter: ``codex exec --json`` in a conversation workspace.

Uses the local Codex binary (not Chat Completions). Isolation flags:
``--ephemeral`` / ``--ignore-user-config`` when persist is off,
``--skip-git-repo-check``, read-only sandbox unless persist writes are on.

Auth: ``CODEX_API_KEY`` or the CLI's saved login. Live runs need the binary
on PATH; tests never import this unless the caller asks for ``codex``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ..protocol import AgentRequest, AgentResult, AgentRunner, EVENT_RETRIEVE
from ..trajectory import (
    build_trajectory,
    events_from_codex_raw,
    load_answer_file,
    normalize_usage,
    parse_codex_jsonl,
    parse_structured_answer,
)

# The prompt requires one JSON answer. Avoid ``--output-schema`` because on
# current Codex CLI builds it constrains intermediate agent messages too,
# preventing ordinary workspace tool calls.
ANSWER_BASENAME = "agent_answer.json"

DEFAULT_TIMEOUT_S = 600.0
# Official Windows installer puts the binary here and on the *user* PATH.
# Cursor/conda shells started before that install still have a stale process PATH.
_WINDOWS_BIN_NAMES = ("codex.exe", "codex.cmd", "codex.bat", "codex")
_POSIX_BIN_NAMES = ("codex",)


def resolve_codex_bin(explicit: str | None = None) -> str | None:
    """Find ``codex`` even when this process inherited a stale PATH.

    Order: explicit arg / ``CODEX_BIN`` / ``shutil.which`` / well-known
    install dirs / directories on the Windows *user* PATH.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path)
        return None
    env_bin = os.environ.get("CODEX_BIN")
    if env_bin:
        path = Path(env_bin).expanduser()
        if path.is_file():
            return str(path)
    found = shutil.which("codex") or shutil.which("codex.exe")
    if found:
        return found
    for directory in _codex_search_dirs():
        hit = _codex_in_dir(directory)
        if hit:
            return str(hit)
    return None


def _codex_search_dirs() -> list[Path]:
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA") or (home / "AppData" / "Local"))
    dirs = [
        local / "Programs" / "OpenAI" / "Codex" / "bin",
        home / ".codex" / "packages" / "standalone" / "current" / "bin",
        home / ".local" / "bin",
        home / ".cargo" / "bin",
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
    ]
    seen: set[str] = set()
    out: list[Path] = []
    for directory in dirs + _windows_user_path_dirs():
        key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        out.append(directory)
    return out


def _windows_user_path_dirs() -> list[Path]:
    """User PATH from the registry — not the (possibly stale) process PATH."""
    if os.name != "nt":
        return []
    raw = ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            raw, _typ = winreg.QueryValueEx(key, "Path")
    except OSError:
        return []
    dirs: list[Path] = []
    for part in str(raw or "").split(os.pathsep):
        part = part.strip().strip('"')
        if part:
            dirs.append(Path(os.path.expandvars(part)))
    return dirs


def _codex_in_dir(directory: Path) -> Path | None:
    names = _WINDOWS_BIN_NAMES if os.name == "nt" else _POSIX_BIN_NAMES
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _codex_subprocess_env() -> dict[str, str]:
    """Copy the process env; fill ``CODEX_API_KEY`` from ``OPENAI_API_KEY``.

    Live sandwich runs already load repo-root ``.env``. Codex prefers
    ``CODEX_API_KEY`` (or ChatGPT login). Stale login tokens 401 after the
    schema file is found; reuse the OpenAI key the rest of the pipeline uses.
    """
    env = os.environ.copy()
    if not (env.get("CODEX_API_KEY") or "").strip():
        openai_key = (env.get("OPENAI_API_KEY") or "").strip()
        if openai_key:
            env["CODEX_API_KEY"] = openai_key
    return env


def _missing_codex_message() -> str:
    searched = ", ".join(str(p) for p in _codex_search_dirs()[:6])
    return (
        "codex binary not found. Install Codex CLI "
        "(https://github.com/openai/codex), restart the terminal so PATH "
        "picks up the installer, or set CODEX_BIN / agent.codex_bin. "
        f"Looked in PATH plus: {searched}"
    )


class CodexAgentRunner(AgentRunner):
    """One ``codex exec`` per LoCoMo question. No TUI."""

    adapter_id = "codex"

    def __init__(
        self,
        model_name: str = "gpt-5",
        *,
        persist_memory: bool = False,
        tools: str = "native",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        codex_bin: str | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model_name = model_name
        self.persist_memory = bool(persist_memory)
        self.tools = str(tools or "native")
        self.timeout_s = float(timeout_s)
        self.codex_bin = codex_bin
        self.extra_args = list(extra_args or [])

    def run(self, request: AgentRequest) -> AgentResult:
        binary = resolve_codex_bin(self.codex_bin)
        if not binary:
            raise FileNotFoundError(_missing_codex_message())
        workspace = Path(request.workspace_dir).resolve()
        answer_path = workspace / ANSWER_BASENAME
        argv = self._argv(binary, workspace, request.prompt)
        env = _codex_subprocess_env()
        started = time.perf_counter()
        completed = subprocess.run(
            argv,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_s,
            env=env,
            check=False,
        )
        latency = time.perf_counter() - started
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        raw, usage, last_message = parse_codex_jsonl(stdout)
        events = events_from_codex_raw(raw)
        answer = load_answer_file(answer_path) or parse_structured_answer(last_message)
        if not answer:
            answer = (last_message or "").strip()
        if completed.returncode != 0 and not answer:
            raise RuntimeError(
                f"codex exec failed (exit {completed.returncode}): "
                f"{(stderr or stdout)[-2000:]}"
            )
        usage = normalize_usage(usage)
        trajectory = build_trajectory(
            events, evidence_ids=request.evidence_ids, usage=usage
        )
        successful_reads = [
            event
            for event in events
            if event.kind == EVENT_RETRIEVE and bool(event.retrieved_text.strip())
        ]
        if not successful_reads:
            trajectory.harness_failed = True
            trajectory.harness_failure_reason = (
                "no_successful_workspace_read_or_search_event"
            )
        return AgentResult(
            predicted_answer=answer or "Unknown.",
            trajectory=trajectory,
            latency_s=round(latency, 6),
            usage=usage,
            events_raw=raw,
            call_meta={
                "adapter": self.adapter_id,
                "model": self.model_name,
                "returncode": completed.returncode,
                "argv": argv[1:],
                "tools": self.tools,
                "persist_memory": self.persist_memory,
            },
            stderr=stderr[-8000:],
        )

    def _argv(self, binary: str, workspace: Path, prompt: str) -> list[str]:
        sandbox = "workspace-write" if self.persist_memory else "read-only"
        root = workspace.resolve()
        argv = [
            binary,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--cd",
            str(root),
            "--sandbox",
            sandbox,
            "--model",
            self.model_name,
            "-o",
            str(root / ANSWER_BASENAME),
        ]
        if not self.persist_memory:
            argv.extend(["--ephemeral", "--ignore-user-config"])
        if self.tools == "controlled":
            # Native file tools only: no MCP, no extra servers. Persist the
            # policy in argv so a later adapter can swap in a tool allowlist.
            argv.extend(["--ignore-rules"])
        argv.extend(self.extra_args)
        argv.append(prompt)
        return argv
