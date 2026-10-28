"""Local Codex smoke on the first-pass Plus decisions. Not a benchmark.

Stage 1 and Stage 2 already ran. This script only asks gpt-4o-mini,
through codex exec, to answer three saved policies. Cloud Run is not
involved. A delivery token is added here so a pass means the hook
context reached the model.

    python harness/codex/spikes/run_plus_codex_smoke.py
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
from harness.codex.spikes.run_personalization_smoke import smoke_policy  # noqa: E402
from src.locomo_eval.agents.adapters.codex import resolve_codex_bin  # noqa: E402
from src.locomo_eval.env import load_env  # noqa: E402

CONFIG = ROOT / "configs" / "agents" / "hooks" / "hello_world.yaml"
RUNS = ROOT / "harness" / "codex" / "spikes" / "runs"
DEFAULT_PACK = ROOT / "experiments" / "personalization-e2e-same-user"
MODEL = "gpt-4o-mini"
TIMEOUT_S = 120.0

# One instruction inject, one memory inject, one abstain. Same cell-b flags.
PLAN = (
    ("plus0_c", "plus-0", "C", "injection"),
    ("plus0_b", "plus-0", "B", "injection"),
    ("plus188_c", "plus-188", "C", "abstain"),
)

ANSWER_PROMPT = """Answer the request in two or three sentences. Do not use tools.
A hook may add developer context.
If that context contains a token beginning with HARNESS_MARKER_,
put the token alone on the first line, then answer.
If no such token was added, do not invent one.
Request: {request}
"""


def request_text(request_dir: Path) -> str:
    """The bare request. N's prompt has no memory and no decision."""
    raw = (request_dir / "N" / "prompt.txt").read_text(encoding="utf-8")
    marker = "Request:\n"
    if marker not in raw:
        raise ValueError(f"{request_dir} has no bare request")
    return raw.split(marker, 1)[1].strip()


def cases_from_pack(pack: Path) -> list[dict]:
    """Policies already written by the Plus run. Codex does not reselect."""
    out = []
    for cell_id, request_id, condition, judge in PLAN:
        folder = pack / "requests" / request_id / condition
        decision = json.loads((folder / "memory" / "decision.json").read_text(encoding="utf-8"))
        policy = smoke_policy(_PolicyView(decision))
        prompt = ANSWER_PROMPT.format(request=request_text(pack / "requests" / request_id))
        if policy and policy.get("text") and request_id in policy["text"]:
            raise ValueError(f"{cell_id} injected text still contains {request_id}")
        out.append(
            {
                "cell_id": cell_id,
                "judge": judge,
                "request_id": request_id,
                "condition": condition,
                "policy": policy,
                "prompt": prompt,
                "selected_ids": list(decision.get("selected_ids") or []),
            }
        )
    return out


class _PolicyView:
    """The fields smoke_policy already reads off a ConditionDecision."""

    def __init__(self, decision: dict):
        self.hooks_enabled = True
        self.abstain = bool(decision.get("abstain"))
        self.text = str(decision.get("text") or "")
        self.condition = str(decision.get("condition") or "")
        self.selected_ids = list(decision.get("selected_ids") or [])


def main() -> int:
    load_env()
    binary = resolve_codex_bin()
    if not binary:
        print("codex binary not found")
        return 2
    pack = DEFAULT_PACK
    if not (pack / "e2e_report.json").is_file():
        print(f"missing pack {pack}")
        return 2
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    base = next(cell for cell in load_cells(config) if cell.cell_id == "b_project_hooks")
    run_dir = RUNS / f"plus-codex-smoke-{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases_from_pack(pack):
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
        rows.append(result)
        message = (result.get("message") or "").replace("\n", " ")
        print(
            f"{case['cell_id']} {result['verdict']} actions={result['actions']} "
            f"message={message[:180]}"
        )
    (run_dir / "smoke.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Plus Codex delivery smoke",
        "",
        f"Model: {MODEL}",
        f"Pack: {pack}",
        f"When: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Local codex exec only. Cloud Run is not used.",
        "Pass means the hook delivered or withheld the saved policy.",
        "The reply is not a personalization score.",
        "",
        "| Cell | Verdict | Actions | Reply |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        reply = (row.get("message") or "").replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {row['cell']} | {row['verdict']} | {','.join(row['actions'])} | {reply} |"
        )
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failed = [row["cell"] for row in rows if row["verdict"] != "pass"]
    print(f"wrote {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
