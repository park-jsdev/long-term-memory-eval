"""First-pass H1 pack: sheet memory in, harness decision out, replies saved.

The sheet's If/When/How/Then are the score. They are not copied into
Stage 1, Stage 2, or the reply prompts. Reply usefulness is not judged
here. A bare reply with no memory is condition N, a floor beside P/A/B/C.

Chat Completions stands in for the hook channel on the reply call. The
same string is written to policy.json and checked through the marker
hook, which is the delivery path a Codex exec would use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .decision import decide_request, write_decision_files
from .diagnostics import summarize_pack
from .poc import deliver_policy, prompt_paths, user_prompt
from .schema import RequestLabel
from .sheet import load_jsonl, merge_sheet
from .sheet_records import gold_request_from_labeled_row, memory_record_from_labeled_row
from .stage_model import run_harness_stages

ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = ROOT / "prompts" / "patterns"
STARTER = ROOT / "annotations" / "plus_starter.jsonl"
SHEET = ROOT / "annotations" / "plus_sheet.jsonl"

# One easy select, one scope stretch, one state, one abstain, one exception.
FIRST_PASS_IDS = ("plus-0", "plus-3", "plus-101", "plus-188", "plus-301")

REPLY_CONDITIONS = ("N", "P", "A", "B", "C")
REPLY_SYSTEM = (
    "Answer the request. Follow developer context when it is present. "
    "When no memory and no developer context is present, answer from the request alone."
)


def gold_decision_fields_in(text: str, gold: RequestLabel) -> list[str]:
    """Which gold decision strings leaked into a model-facing prompt."""
    hits = []
    if gold.when and gold.when in text:
        hits.append("when")
    if gold.how_then and gold.how_then in text:
        hits.append("how_then")
    return hits


def score_harness_decision(gold: RequestLabel, decisions: dict) -> dict[str, Any]:
    """Stage 1 against the sheet's If. Ids must match the gold set exactly."""
    stage_b = decisions["B"]
    stage_c = decisions["C"]
    predicted = "abstain" if stage_b.abstain else "select"
    ids_exact = set(stage_b.selected_ids) == set(gold.applicable_ids)
    leaked = [
        record_id
        for record_id in set(gold.applicable_ids) | {gold.request_id}
        if record_id and record_id in (stage_c.stage2_instruction or "")
    ]
    return {
        "request_id": gold.request_id,
        "gold_if": gold.if_decision,
        "predicted_if": predicted,
        "if_match": predicted == gold.if_decision and ids_exact and stage_b.selected_ids == stage_c.selected_ids,
        "selected_ids": list(stage_b.selected_ids),
        "record_id_in_instruction": leaked,
        "stage2_empty": not str(stage_c.stage2_instruction or "").strip(),
        "reason": stage_b.reason,
    }


def reply_user_prompt(
    condition: str,
    request_text: str,
    delivered_text: str,
    guidelines: str,
) -> str:
    """What the reply model reads. N has neither memory nor a decision.

    Hook conditions inline the policy text because this call has no
    Codex hook channel. The file on disk is the same string the hook returns.
    """
    if condition == "N":
        return f"Request:\n{request_text.strip()}\n"
    if condition == "P":
        return user_prompt("P", request_text, delivered_text, guidelines)
    if not delivered_text.strip():
        return (
            "No developer context was added. Answer without personalization.\n\n"
            f"Request:\n{request_text.strip()}\n"
        )
    return (
        "Developer context:\n"
        f"{delivered_text.strip()}\n\n"
        "Follow that context when it applies.\n\n"
        f"Request:\n{request_text.strip()}\n"
    )


def labeled_rows(ids: list[str], starter_path: Path, sheet_path: Path) -> list[dict[str, Any]]:
    """Sheet order is ignored. ids is the run list. Human rows win over drafts."""
    starter = load_jsonl(starter_path)
    saved = load_jsonl(sheet_path)
    merged = {str(row.get("record_id")): row for row in merge_sheet(starter, saved)}
    chosen = []
    for record_id in ids:
        row = merged.get(record_id)
        if row is None:
            raise KeyError(f"No sheet row for {record_id}")
        if not str(row.get("scope") or "").strip():
            raise ValueError(f"{record_id} has no scope, so it is not a first-pass label")
        chosen.append(row)
    return chosen


def build_first_pass_e2e_pack(
    dest: Path,
    rows: list[dict[str, Any]],
    completer: Any,
    *,
    answer: bool,
    top_k: int = 3,
) -> dict[str, Any]:
    """Write the pack. answer=False leaves DRY_RUN in the reply files."""
    dest.mkdir(parents=True, exist_ok=True)
    guidelines = (PROMPT_DIR / "prompt_only_v0.txt").read_text(encoding="utf-8")
    stage1_prompt = (PROMPT_DIR / "stage1_select_v0.txt").read_text(encoding="utf-8")
    stage2_prompt = (PROMPT_DIR / "stage2_instruct_v0.txt").read_text(encoding="utf-8")
    paths = prompt_paths()
    scores = []
    known_ids: list[str] = []
    for row in rows:
        record = memory_record_from_labeled_row(row)
        gold = gold_request_from_labeled_row(row)
        known_ids.append(record.record_id)
        stage1, instruction, trace = run_harness_stages(
            completer,
            [record],
            gold.request,
            stage1_prompt=stage1_prompt,
            stage2_prompt=stage2_prompt,
        )
        decisions = decide_request(
            [record],
            gold.request,
            gold.request_id,
            top_k=top_k,
            prompt_paths=paths,
            stage1_bc=stage1,
            stage2_text=instruction,
            stage_prompts_used=True,
        )
        request_dir = dest / "requests" / gold.request_id
        _write_trace(request_dir, trace)
        leaked = []
        for field in ("stage1_prompt", "stage1_system", "stage2_prompt"):
            leaked.extend(gold_decision_fields_in(trace.get(field) or "", gold))
        for condition in ("P", "A", "B", "C"):
            folder = request_dir / condition
            write_decision_files(folder, decisions[condition])
            prompt = reply_user_prompt(
                condition, gold.request, decisions[condition].text, guidelines
            )
            _write_reply(folder, prompt, completer, answer=answer)
            if decisions[condition].hooks_enabled:
                deliver_policy(folder)
            leaked.extend(gold_decision_fields_in(prompt, gold))
        bare = request_dir / "N"
        bare_prompt = reply_user_prompt("N", gold.request, "", guidelines)
        _write_reply(bare, bare_prompt, completer, answer=answer)
        leaked.extend(gold_decision_fields_in(bare_prompt, gold))
        row_score = score_harness_decision(gold, decisions)
        row_score["gold_fields_in_prompts"] = sorted(set(leaked))
        row_score["label_status"] = row.get("label_status")
        scores.append(row_score)
    report = summarize_pack(dest, known_ids)
    matched = sum(1 for row in scores if row["if_match"])
    summary = {
        "label_source": "first_pass_sheet",
        "confirmed_gold": False,
        "response_adaptation": None,
        "if_match_count": matched,
        "item_count": len(scores),
        "items": scores,
        "bc_identity_ok": report["bc_identity_ok"],
        "note": (
            "If/When match is against the first-pass sheet, not confirmed gold. "
            "Each trigger has one memory, so Stage 1 is apply or abstain. "
            "Reply usefulness is not judged."
        ),
    }
    (dest / "e2e_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_trace(request_dir: Path, trace: dict[str, str]) -> None:
    request_dir.mkdir(parents=True, exist_ok=True)
    (request_dir / "stage_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_reply(folder: Path, prompt: str, completer: Any, *, answer: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    if not answer:
        (folder / "response.txt").write_text("DRY_RUN\n", encoding="utf-8")
        return
    messages = [
        {"role": "system", "content": REPLY_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    from .stage_model import complete_text

    text = complete_text(completer, messages, json_mode=False)
    (folder / "response.txt").write_text(text + "\n", encoding="utf-8")


class OpenAICompleter:
    """gpt-4o-mini chat caller. Stage 1 asks for a JSON object."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from src.locomo_eval.readers import OpenAIChatCaller

        self.caller = OpenAIChatCaller(
            model,
            temperature=0.0,
            max_tokens=500,
            thinking=False,
        )

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        extra = {"response_format": {"type": "json_object"}} if json_mode else None
        text, _meta = self.caller.complete(messages, create_extra=extra, thinking=False)
        return text


class WiringCompleter:
    """Offline stand-in. Abstains, so a dry pack is not a model result."""

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        system = messages[0]["content"] if messages else ""
        if "instruction stage" in system:
            return "Consider the stored constraint within its scope."
        if json_mode or "decision stage" in system:
            return json.dumps(
                {"selected_ids": [], "abstain": True, "reason": "dry_run"}
            )
        return "DRY_RUN"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the first-pass H1 PoC")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "personalization-e2e",
    )
    parser.add_argument("--ids", default=",".join(FIRST_PASS_IDS))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the pack with an abstaining stand-in and DRY_RUN replies",
    )
    args = parser.parse_args()
    ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    rows = labeled_rows(ids, STARTER, SHEET)
    completer = WiringCompleter() if args.dry_run else OpenAICompleter(args.model)
    summary = build_first_pass_e2e_pack(
        args.out,
        rows,
        completer,
        answer=not args.dry_run,
    )
    print(f"wrote {args.out}")
    print(f"if_match {summary['if_match_count']}/{summary['item_count']}")
    for item in summary["items"]:
        print(
            f"{item['request_id']} gold={item['gold_if']} "
            f"predicted={item['predicted_if']} match={item['if_match']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
