"""Write P/A/B/C decision files. The hook does not choose memories.

The default Stage 1 and Stage 2 are unigram stubs so a pack exists
before any prompt is called. A caller can pass model Stage 1 and Stage 2
results instead. B and C share that one Stage 1 result. C does not reselect.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.locomo_eval.mem0_metrics import simple_tokenize

from .hypothesis_record import HypothesisRecord
from .hypothesis_rendering import render_hypothesis, render_memory_for_reply

CONDITIONS = ("P", "A", "B", "C")
# Unigrams that would make every request look applicable.
_STOP = frozenset(
    """
    a an the and or to of in on for with is it that this my i you we they
    was were be as at from by if but not so just have has had do did will
    would can could about into than then there their our your me he she
    his her said what which should use
    """.split()
)


@dataclass
class ConditionDecision:
    """One condition's inspectable decision, separate from the response."""

    condition: str
    request_id: str
    action: str
    selected_ids: list[str]
    abstain: bool
    reason: str
    stage2_instruction: str
    text: str
    hooks_enabled: bool
    stage_prompts_used: bool = False
    prompt_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def content_tokens(text: str) -> set[str]:
    """Content unigrams. Shared with the Mem0 lexical splitter on purpose.

    The full Stage 1 prompt replaces this. Keeping the splitter local to
    that prompt avoids a second definition of "token overlap" in the PoC.
    """
    return {tok for tok in simple_tokenize(text) if tok not in _STOP and len(tok) > 2}


def overlap_count(record: HypothesisRecord, request: str) -> int:
    """How many scope/what words also appear in the request."""
    basis = content_tokens(f"{record.scope} {record.what}")
    return len(basis & content_tokens(request))


def matching_exceptions(record: HypothesisRecord, request: str) -> list[str]:
    """Exception phrases whose content words all appear in the request.

    This is a phrase match, not a reading of conflict. "Security" in the
    request does not fire unless the stored exception's words are present.
    """
    request_tokens = content_tokens(request)
    hits = []
    for exception in record.exceptions:
        words = content_tokens(exception)
        if words and words <= request_tokens:
            hits.append(exception)
    return hits


def _ranked(records: list[HypothesisRecord], request: str) -> list[HypothesisRecord]:
    return sorted(
        records,
        key=lambda record: (-overlap_count(record, request), record.record_id),
    )


def stage1_selection(
    records: list[HypothesisRecord],
    request: str,
    *,
    condition: str,
    top_k: int,
) -> tuple[list[HypothesisRecord], bool, str]:
    """Select records or abstain.

    Condition A never abstains when any record exists: it is the
    relevance baseline with no applicability rule. B and C abstain when
    no scope or what word overlaps the request. An exception match does
    not abstain. Stage 2 is where that exception is named.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    if condition == "A":
        if not records:
            return [], True, "no_records"
        return _ranked(records, request)[:top_k], False, "relevance_topk"
    selected = [record for record in records if overlap_count(record, request) > 0]
    if not selected:
        return [], True, "no_overlap"
    return selected, False, "scope_overlap"


def bounded_instruction(
    record: HypothesisRecord,
    overrides: list[str],
) -> str:
    """Stage 2 line the response model is allowed to see.

    The record id stays in decision.json and the hook log. It is omitted
    here so the injected text does not disclose bookkeeping, and so the
    id cannot become a confound in the answer.
    """
    override = "; ".join(overrides) if overrides else "none stated in the request"
    return (
        f"Consider {record.what} within {record.scope}. "
        f"Do not override: {override}."
    )


def stage2_instruction(
    selected: list[HypothesisRecord],
    request: str,
    *,
    abstain: bool,
    condition: str,
) -> str:
    """Instruction text for C. A and B inject records, not this prose."""
    if condition != "C" or abstain or not selected:
        return ""
    lines = []
    for record in selected:
        lines.append(bounded_instruction(record, matching_exceptions(record, request)))
    return "\n".join(lines)


def policy_body(decision: ConditionDecision) -> dict | None:
    """The file marker_hook.py already understands, plus log fields.

    P returns None so the caller writes no policy and the hook never
    starts. Abstain is an empty text with action abstain.
    """
    if not decision.hooks_enabled:
        return None
    return {
        "action": "abstain" if decision.abstain else "inject",
        "text": "" if decision.abstain else decision.text,
        "condition": decision.condition,
        "selected_ids": list(decision.selected_ids),
    }


def decide_request(
    records: list[HypothesisRecord],
    request_text: str,
    request_id: str,
    *,
    top_k: int = 3,
    prompt_paths: dict[str, str] | None = None,
    stage1_bc: tuple[list[HypothesisRecord], bool, str] | None = None,
    stage2_text: str | None = None,
    stage_prompts_used: bool = False,
) -> dict[str, ConditionDecision]:
    """Four conditions. B and C share one Stage 1 result.

    stage1_bc replaces the unigram stub for B and C. Condition A stays
    the relevance baseline. stage2_text is the model instruction for C.
    stage_prompts_used drops record ids from text the reply model sees.
    """
    paths = dict(prompt_paths or {})
    if stage1_bc is None:
        shared, shared_abstain, shared_reason = stage1_selection(
            records, request_text, condition="B", top_k=top_k
        )
    else:
        shared, shared_abstain, shared_reason = stage1_bc
    relevance, relevance_abstain, relevance_reason = stage1_selection(
        records, request_text, condition="A", top_k=top_k
    )
    out: dict[str, ConditionDecision] = {}
    for condition in CONDITIONS:
        if condition == "A":
            selected, abstain, reason = relevance, relevance_abstain, relevance_reason
        elif condition == "P":
            selected, abstain, reason = records, False, "prompt_only"
        else:
            selected, abstain, reason = shared, shared_abstain, shared_reason
        instruction = stage2_instruction(
            selected, request_text, abstain=abstain, condition=condition
        )
        if condition == "C" and stage2_text is not None and not abstain:
            instruction = stage2_text
        keep_stage1_ids = False
        if condition == "P":
            action = "prompt_only"
            text = _reply_or_audit_text(records, omit_ids=stage_prompts_used)
            hooks = False
        elif condition == "C":
            # An empty Stage 2 still names the Stage 1 ids. Clearing them
            # would look like C reselected, and the B/C pair would be invalid.
            stage1_abstain = abstain
            action = "abstain" if stage1_abstain or not instruction else "inject"
            text = "" if action == "abstain" else instruction
            hooks = True
            abstain = action == "abstain"
            keep_stage1_ids = not stage1_abstain
        else:
            action = "abstain" if abstain else "inject"
            text = "" if abstain else _reply_or_audit_text(selected, omit_ids=stage_prompts_used)
            hooks = True
        if abstain and not keep_stage1_ids:
            selected_ids: list[str] = []
        else:
            selected_ids = [record.record_id for record in selected]
        out[condition] = ConditionDecision(
            condition=condition,
            request_id=request_id,
            action=action,
            selected_ids=selected_ids,
            abstain=abstain,
            reason=reason if condition != "P" else "prompt_only",
            stage2_instruction=instruction,
            text=text,
            hooks_enabled=hooks,
            stage_prompts_used=stage_prompts_used,
            prompt_paths=paths,
        )
    return out


def _reply_or_audit_text(records: list[HypothesisRecord], *, omit_ids: bool) -> str:
    """Pick the rendering. Model runs omit ids. The stub pack keeps them."""
    render = render_memory_for_reply if omit_ids else render_hypothesis
    return "\n\n".join(render(record) for record in records)


def write_decision_files(directory: Path, decision: ConditionDecision) -> None:
    """Write decision.json and, for hook conditions, memory/policy.json."""
    memory = directory / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    (memory / "decision.json").write_text(
        json.dumps(decision.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    policy = policy_body(decision)
    policy_path = memory / "policy.json"
    if policy is None:
        if policy_path.exists():
            policy_path.unlink()
        return
    policy_path.write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
