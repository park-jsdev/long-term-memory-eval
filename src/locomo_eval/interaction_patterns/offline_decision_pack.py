"""Write a dry-run P/A/B/C pack and deliver policies through the hook.

No Codex binary and no API. The hook subprocess is the same contract
as a live exec: stdin JSON, stdout JSON, policy at memory/policy.json.
A live codex exec is a later switch on the adapter, default off.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.data.locomo_plus import CognitiveItem

from .condition_decisions import decide_request, write_decision_files
from .decision_pack_summary import summarize_pack
from .hypothesis_rendering import (
    conventional_summary,
    h2_scores,
    hypothesis_token_budget,
    render_hypotheses,
)
from .hypothesis_record import dump_jsonl
from .low_maintenance_example import SESSION_SUMMARIES, fixture_records
from .plus_item_to_record import (
    hypothesis_from_cognitive_item,
    request_from_cognitive_item,
)

ROOT = Path(__file__).resolve().parents[3]
HOOK = ROOT / "harness" / "codex" / "plugin" / "hooks" / "marker_hook.py"
PROMPT_DIR = ROOT / "prompts" / "patterns"


def prompt_paths() -> dict[str, str]:
    return {
        "prompt_only": str(PROMPT_DIR / "prompt_only_v0.txt"),
        "stage1": str(PROMPT_DIR / "stage1_select_v0.txt"),
        "stage2": str(PROMPT_DIR / "stage2_instruct_v0.txt"),
    }


def user_prompt(condition: str, request_text: str, memory_text: str, guidelines: str) -> str:
    """P carries memories in the user message. Hook conditions do not.

    Putting the same memories in both places would make the hook's
    additionalContext impossible to tell apart from the prompt.
    """
    if condition == "P":
        return (
            f"{guidelines.strip()}\n\n"
            f"Memories:\n{memory_text}\n\n"
            f"Request:\n{request_text}\n"
        )
    return (
        "A hook may add developer context. Follow it when it is present.\n"
        "If no developer context was added, answer without personalization.\n\n"
        f"Request:\n{request_text}\n"
    )


def deliver_policy(directory: Path) -> dict:
    """Run the marker hook once, as Codex would on UserPromptSubmit.

    The workspace cwd is the condition directory. The hook reads
    memory/policy.json and appends hook_log.jsonl itself.
    """
    event = {"hook_event_name": "UserPromptSubmit", "prompt": "poc"}
    completed = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(event),
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=str(directory),
        check=False,
    )
    body = json.loads(completed.stdout) if completed.stdout.strip() else {}
    (directory / "hook_stdout.json").write_text(
        json.dumps(
            {
                "returncode": completed.returncode,
                "body": body,
                "stderr": completed.stderr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return body


def build_personalization_poc_pack(dest: Path, *, top_k: int = 3) -> dict:
    """Write the fixture pack. Responses are DRY_RUN, not model answers."""
    records, requests = fixture_records()
    dest.mkdir(parents=True, exist_ok=True)
    dump_jsonl(
        dest / "annotations" / "hypotheses.jsonl",
        [record.to_dict() for record in records],
    )
    dump_jsonl(
        dest / "annotations" / "requests.jsonl",
        [label.to_dict() for label in requests],
    )
    hypothesis_text = render_hypotheses(records)
    budget = hypothesis_token_budget(records)
    summary = conventional_summary(list(SESSION_SUMMARIES), budget)
    scores = h2_scores(
        records,
        hypothesis_text,
        summary,
        unsupported_phrases=["buys a yacht"],
    )
    (dest / "memory").mkdir(parents=True, exist_ok=True)
    (dest / "memory" / "hypotheses.txt").write_text(hypothesis_text + "\n", encoding="utf-8")
    (dest / "memory" / "conventional_summary.txt").write_text(summary + "\n", encoding="utf-8")
    (dest / "h2_scores.json").write_text(
        json.dumps(scores, indent=2) + "\n",
        encoding="utf-8",
    )
    guidelines = (PROMPT_DIR / "prompt_only_v0.txt").read_text(encoding="utf-8")
    paths = prompt_paths()
    known_ids = [record.record_id for record in records]
    for label in requests:
        decisions = decide_request(
            records,
            label.request,
            label.request_id,
            top_k=top_k,
            prompt_paths=paths,
        )
        for condition, decision in decisions.items():
            folder = dest / "requests" / label.request_id / condition
            write_decision_files(folder, decision)
            prompt = user_prompt(condition, label.request, decision.text, guidelines)
            (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
            (folder / "response.txt").write_text("DRY_RUN\n", encoding="utf-8")
            if decision.hooks_enabled:
                deliver_policy(folder)
    _write_plus_example(dest, top_k=top_k, paths=paths, guidelines=guidelines)
    report = summarize_pack(dest, known_ids)
    (dest / "diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _write_plus_example(dest: Path, *, top_k: int, paths: dict[str, str], guidelines: str) -> None:
    """One synthetic Plus-shaped item, so the pack does not need the download.

    The real count lives in data/raw when fetched. This example only
    proves the cue and the trigger fit the same decision files.
    """
    item = CognitiveItem(
        index=0,
        relation_type="causal",
        cue_dialogue="A: After learning to say no, I felt less stressed.",
        trigger_query="A: I volunteered and now I am overwhelmed.",
        time_gap="two weeks later",
        evidence_ids=[],
        cue_is_insertion=True,
    )
    record = hypothesis_from_cognitive_item(item)
    label = request_from_cognitive_item(item)
    decisions = decide_request(
        [record],
        label.request,
        label.request_id,
        top_k=top_k,
        prompt_paths=paths,
    )
    for condition, decision in decisions.items():
        folder = dest / "requests" / label.request_id / condition
        write_decision_files(folder, decision)
        (folder / "prompt.txt").write_text(
            user_prompt(condition, label.request, decision.text, guidelines),
            encoding="utf-8",
        )
        (folder / "response.txt").write_text("DRY_RUN\n", encoding="utf-8")
        if decision.hooks_enabled:
            deliver_policy(folder)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the personalization PoC pack")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "personalization-poc",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    report = build_personalization_poc_pack(args.out, top_k=args.top_k)
    print(f"wrote {args.out}")
    print(f"bc_identity_ok {report['bc_identity_ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
