"""Turn one labeled sheet row into a memory record and a gold decision.

The harness reads the memory only. If, When, and How/Then stay on the
gold label so a later score can use them without the model having seen
them. Draft rows are allowed. Confirmed rows use the same fields.
"""

from __future__ import annotations

from typing import Any

from .hypothesis_record import (
    HypothesisRecord,
    RequestLabel,
    validate_hypothesis,
    validate_request,
)


def memory_record_from_labeled_row(row: dict[str, Any]) -> HypothesisRecord:
    """Who/What/Why from the sheet. Decision fields are not copied.

    Plus cues have no turn ids. The cue index is the evidence id so the
    record still validates as an explicit single-session memory.
    """
    record_id = str(row["record_id"])
    index = record_id.split("-")[-1]
    evidence = list(row.get("evidence_ids") or [f"cue:{index}"])
    record = HypothesisRecord(
        record_id=record_id,
        speaker=str(row.get("speaker") or "A"),
        sample_id=str(row.get("sample_id") or record_id),
        who_basis=str(row.get("who_basis") or ""),
        what_type=str(row.get("what_type") or ""),
        what=str(row.get("what") or ""),
        scope=str(row.get("scope") or ""),
        exceptions=list(row.get("exceptions") or []),
        evidence_ids=evidence,
        session_ids=list(row.get("session_ids") or ["cue"]),
        uncertainty=str(row.get("uncertainty") or ""),
        fixture=False,
    )
    validate_hypothesis(record)
    return record


def gold_request_from_labeled_row(row: dict[str, Any]) -> RequestLabel:
    """The trigger plus the sheet's decision. This object is the score.

    A select row's applicable id is that row's own record. The sheet
    stores one cue per trigger, so there is no second gold memory.
    """
    record_id = str(row["record_id"])
    decision = str(row.get("if_decision") or "")
    applicable = [record_id] if decision == "select" else []
    label = RequestLabel(
        request_id=record_id,
        sample_id=str(row.get("sample_id") or record_id),
        speaker=str(row.get("speaker") or "A"),
        request=str(row.get("request") or ""),
        if_decision=decision,
        when=str(row.get("when") or ""),
        how_then=str(row.get("how_then") or ""),
        applicable_ids=applicable,
        fixture=False,
    )
    validate_request(label)
    return label
