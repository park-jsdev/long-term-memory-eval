"""Hand-written records the PoC pack runs. Not annotation data.

The preference is the proposal's maintenance example, worded so the
unigram stub can see it. A later annotator replaces the wording.
"""

from __future__ import annotations

from .hypothesis_record import (
    HypothesisRecord,
    RequestLabel,
    validate_hypothesis,
    validate_request,
)

LOW_MAINTENANCE = HypothesisRecord(
    record_id="hyp-low-maint",
    speaker="Maya",
    sample_id="fixture-conv",
    who_basis="inferred",
    what_type="taste_or_value",
    what="Prefers low-maintenance software tools",
    scope="software tools",
    exceptions=["explicit security requirement"],
    evidence_ids=["D1:1", "D3:2"],
    session_ids=["1", "3"],
    contradictions=[],
    uncertainty="",
    fixture=True,
)

REQUESTS = (
    RequestLabel(
        request_id="select-tool",
        sample_id="fixture-conv",
        speaker="Maya",
        request="Which software tool should I use for class notes?",
        if_decision="select",
        when="The request asks for a software tool, inside the stored scope.",
        how_then="Prefer a low-maintenance option.",
        applicable_ids=["hyp-low-maint"],
        fixture=True,
    ),
    RequestLabel(
        request_id="abstain-fact",
        sample_id="fixture-conv",
        speaker="Maya",
        request="What is the capital of France?",
        if_decision="abstain",
        when="No stored scope overlaps this factual request.",
        how_then="",
        applicable_ids=[],
        fixture=True,
    ),
    RequestLabel(
        request_id="override-security",
        sample_id="fixture-conv",
        speaker="Maya",
        request=(
            "Pick a software tool that meets the explicit security requirement."
        ),
        if_decision="select",
        when="The tool scope applies, and the request states the stored exception.",
        how_then="Mention maintenance only if it does not weaken the security requirement.",
        applicable_ids=["hyp-low-maint"],
        mandatory_constraints=["explicit security requirement"],
        fixture=True,
    ),
)

# A conventional summary that keeps the topic and drops evidence ids.
# The unsupported phrase is absent here so the fixture score stays clean.
SESSION_SUMMARIES = (
    ("1", "Maya discussed software tools for class notes."),
    ("3", "Maya again wanted software tools that are easy to keep up."),
)


def fixture_records() -> tuple[list[HypothesisRecord], tuple[RequestLabel, ...]]:
    """Return validated fixture rows. Called by the pack writer."""
    records = [LOW_MAINTENANCE]
    for record in records:
        validate_hypothesis(record)
    for label in REQUESTS:
        validate_request(label)
    return records, REQUESTS
