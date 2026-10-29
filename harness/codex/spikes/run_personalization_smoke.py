"""Delivery smoke for the personalization policy on gpt-4o-mini.

This is not an H1 result. A disposable CODEX_HOME and a workspace
outside the git repo match the hello-world spikes. The delivery token
is added only in this script's policy text. The library instruction
stays free of record ids and free of that token.

    python harness/codex/spikes/run_personalization_smoke.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.codex.spikes.codex_hello_world import MARKER_ALPHA, load_cells  # noqa: E402
from harness.codex.spikes.run_codex_hello_world import (  # noqa: E402
    _utc_stamp,
    execute_cell,
)
from src.locomo_eval.agents.adapters.codex import resolve_codex_bin  # noqa: E402
from src.locomo_eval.env import load_env  # noqa: E402
from src.locomo_eval.interaction_patterns.condition_decisions import decide_request, policy_body  # noqa: E402
from src.locomo_eval.interaction_patterns.low_maintenance_example import fixture_records  # noqa: E402

CONFIG = ROOT / "configs" / "agents" / "hooks" / "hello_world.yaml"
RUNS = ROOT / "harness" / "codex" / "spikes" / "runs"
MODEL = "gpt-4o-mini"
TIMEOUT_S = 120.0

SMOKE_PROMPT = """Reply with exactly one line and do not use tools.
A hook may add developer context.
If that context contains a token beginning with HARNESS_MARKER_,
reply SEEN, then the token, then a short quote of the rest of the context.
If no such token was added, reply MISSING.
Request: {request}
"""


def smoke_policy(decision) -> dict | None:
    """Hook policy. Inject cases carry a smoke-only delivery token."""
    policy = policy_body(decision)
    if policy is None:
        return None
    if policy["action"] == "inject" and policy.get("text"):
        policy = dict(policy)
        policy["text"] = f"{policy['text']}\nDelivery token: {MARKER_ALPHA}"
    return policy


def cases() -> list[dict]:
    """Three cell-b execs: select, abstain, and the security exception."""
    records, requests = fixture_records()
    by_id = {label.request_id: label for label in requests}
    plan = [
        ("c_select", "select-tool", "C", "injection"),
        ("b_abstain", "abstain-fact", "B", "abstain"),
        ("c_override", "override-security", "C", "injection"),
    ]
    out = []
    for cell_id, request_id, condition, judge in plan:
        label = by_id[request_id]
        decision = decide_request(records, label.request, request_id)[condition]
        out.append(
            {
                "cell_id": cell_id,
                "judge": judge,
                "request_id": request_id,
                "condition": condition,
                "policy": smoke_policy(decision),
                "prompt": SMOKE_PROMPT.format(request=label.request),
                "selected_ids": list(decision.selected_ids),
                "instruction": decision.stage2_instruction,
            }
        )
    return out


def main() -> int:
    load_env()
    binary = resolve_codex_bin()
    if not binary:
        print("codex binary not found")
        return 2
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    base = next(cell for cell in load_cells(config) if cell.cell_id == "b_project_hooks")
    run_dir = RUNS / f"personalization-smoke-{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases():
        result = execute_cell(
            binary=binary,
            run_dir=run_dir,
            cell=base.renamed(case["cell_id"]),
            prompt=case["prompt"],
            model=MODEL,
            timeout=TIMEOUT_S,
            policy=case["policy"],
            token=MARKER_ALPHA,
            judge=case["judge"],
        )
        result["request_id"] = case["request_id"]
        result["condition"] = case["condition"]
        result["selected_ids"] = case["selected_ids"]
        result["instruction_has_record_id"] = "hyp-low-maint" in case["instruction"]
        rows.append(result)
        print(
            f"{case['cell_id']} {result['verdict']} actions={result['actions']} "
            f"message={result['message'][:180]}"
        )
    (run_dir / "smoke.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Personalization delivery smoke",
        "",
        f"Model: {MODEL}",
        f"When: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Delivery only. The reply is not a personalization score.",
        "Record ids are in the decision and the hook log, not in the instruction.",
        "",
        "| Cell | Verdict | Actions | Reply |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        reply = (row.get("message") or "").replace("|", "/")
        lines.append(
            f"| {row['cell']} | {row['verdict']} | {','.join(row['actions'])} | {reply} |"
        )
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failed = [row["cell"] for row in rows if row["verdict"] != "pass"]
    print(f"wrote {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
