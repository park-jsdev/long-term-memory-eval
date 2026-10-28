"""Codex command hook: inject one marker, or abstain, and always log.

Codex starts this process with one JSON event on stdin and reads stdout
as the hook result. Plain text on stdout for UserPromptSubmit becomes
developer context, so logs go to a file and stderr, never stdout.

Spike 1 leaves out memory/policy.json. The hook then injects
HARNESS_MARKER (default HARNESS_MARKER_ALPHA). That separates "the
process never started" from "the process chose to add nothing".

Spike 2 writes memory/policy.json. action=abstain returns {} so Codex
adds no context. action=inject returns additionalContext. A Stop event
appends one capture line and returns {}. decision=block on Stop would
continue the turn, which this hello world must not do.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_MARKER = "HARNESS_MARKER_ALPHA"


def event_name(event: dict) -> str:
    """Codex has used both casings in hook payloads."""
    return str(event.get("hook_event_name") or event.get("hookEventName") or "")


def load_policy(cwd: Path) -> tuple[dict | None, str]:
    """Return (policy, where_it_came_from).

    A missing file is spike 1: inject the default marker. A present file
    is the policy under test. Invalid JSON abstains so a bad file cannot
    look like a successful injection.
    """
    candidates = []
    override = (os.environ.get("HARNESS_POLICY_PATH") or "").strip()
    if override:
        candidates.append(Path(override))
    candidates.append(cwd / "memory" / "policy.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"action": "abstain", "text": ""}, f"invalid:{path}"
        if not isinstance(data, dict):
            return {"action": "abstain", "text": ""}, f"invalid:{path}"
        return data, str(path)
    return None, ""


def response_for_event(event: dict, policy: dict | None, marker: str) -> tuple[dict, str]:
    """Map one hook event to (stdout JSON, log action).

    inject  — additionalContext carries the marker.
    abstain — empty object, which Codex treats as a no-op.
    capture — Stop wrote a side log; the turn is not continued.
    ignore  — some other event; do not invent context.
    """
    name = event_name(event)
    if name == "Stop":
        return {}, "capture"
    if name != "UserPromptSubmit":
        return {}, "ignore"
    if policy is None:
        text = marker
        action = "inject"
    else:
        action = str(policy.get("action") or "abstain").strip().lower()
        text = str(policy.get("text") or "")
        if action != "inject" or not text:
            return {}, "abstain"
    body = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
        }
    }
    return body, "inject"


def append_jsonl(path: Path | None, row: dict) -> None:
    """Best-effort side log. A read-only sandbox may block one of the paths."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"harness hook log failed: {path}: {exc}", file=sys.stderr)


def log_paths(cwd: Path) -> list[Path]:
    """Workspace log plus an outside-sandbox path when the spike sets one."""
    paths = [cwd / "hook_log.jsonl"]
    override = (os.environ.get("HARNESS_HOOK_LOG") or "").strip()
    if override:
        paths.append(Path(override))
    return paths


def capture_paths(cwd: Path) -> list[Path]:
    paths = [cwd / "memory" / "capture.log"]
    override = (os.environ.get("HARNESS_CAPTURE_LOG") or "").strip()
    if override:
        paths.append(Path(override))
    return paths


def append_capture(cwd: Path, event: dict) -> None:
    line = (
        f"captured turn_id={event.get('turn_id') or ''} "
        f"prompt_chars={len(str(event.get('prompt') or event.get('last_assistant_message') or ''))}\n"
    )
    for path in capture_paths(cwd):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError as exc:
            print(f"harness capture failed: {path}: {exc}", file=sys.stderr)


def main() -> int:
    raw = sys.stdin.read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        event = {"_stdin": "invalid_json"}
    if not isinstance(event, dict):
        event = {"_stdin": "not_an_object"}
    cwd = Path.cwd()
    policy, policy_from = load_policy(cwd)
    marker = (os.environ.get("HARNESS_MARKER") or DEFAULT_MARKER).strip() or DEFAULT_MARKER
    body, action = response_for_event(event, policy, marker)
    if action == "capture":
        append_capture(cwd, event)
    selected = []
    condition = ""
    if isinstance(policy, dict):
        # The decision module stores these so a pack can be audited
        # without parsing the injected prose.
        condition = str(policy.get("condition") or "")
        raw_ids = policy.get("selected_ids") or []
        if isinstance(raw_ids, list):
            selected = [str(item) for item in raw_ids]
    row = {
        "event": event_name(event) or "unknown",
        "action": action,
        "marker": marker if action == "inject" else "",
        "policy_from": policy_from,
        "turn_id": event.get("turn_id") or "",
        "condition": condition,
        "selected_ids": selected,
    }
    for path in log_paths(cwd):
        append_jsonl(path, row)
    sys.stdout.write(json.dumps(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
