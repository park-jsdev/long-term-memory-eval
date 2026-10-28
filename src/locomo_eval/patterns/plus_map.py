"""Map one LoCoMo-Plus cue into the hypothesis and request records.

A raw cue is one inserted dialogue, so who_basis is explicit. It is not
an inferred pattern. The trigger becomes the held-out request. Stage 1
still uses unigram overlap, which will often abstain, and that result
is informative: the real selector has to read the cue, not share words
with the trigger. LoCoMo-Plus filtered for low lexical overlap on purpose.
"""

from __future__ import annotations

from src.data.locomo_plus import CognitiveItem, trigger_text

from .schema import HypothesisRecord, RequestLabel, validate_hypothesis, validate_request

_WHAT_TYPE = {
    "causal": "stable_constraint",
    "state": "stable_constraint",
    "goal": "recurring_goal",
    "value": "taste_or_value",
}
# Public name for the heuristic sheet. Same map the record builder uses.
WHAT_TYPE_FOR_RELATION = _WHAT_TYPE


def hypothesis_from_cognitive_item(item: CognitiveItem) -> HypothesisRecord:
    """One explicit record. Evidence is the cue itself when no turn ids exist."""
    evidence = item.evidence_ids or [f"cue:{item.index}"]
    record = HypothesisRecord(
        record_id=f"plus-{item.index}",
        speaker="A",
        sample_id=f"locomo-plus-{item.index}",
        who_basis="explicit",
        what_type=_WHAT_TYPE[item.relation_type],
        what=item.cue_dialogue,
        scope=item.relation_type,
        evidence_ids=evidence,
        session_ids=["cue"],
        uncertainty="Single inserted cue. Not a two-session inference.",
        fixture=False,
    )
    validate_hypothesis(record)
    return record


def request_from_cognitive_item(item: CognitiveItem) -> RequestLabel:
    """The trigger is the request. Gold If/When is unset beyond a note.

    Annotators still have to decide select versus abstain. The label here
    records that the decision has not been made, using abstain as the
    conservative placeholder and saying so in when.
    """
    label = RequestLabel(
        request_id=f"plus-{item.index}",
        sample_id=f"locomo-plus-{item.index}",
        speaker="A",
        request=trigger_text(item),
        if_decision="abstain",
        when=(
            "Placeholder. A researcher has not yet judged whether this "
            "cue applies to the trigger."
        ),
        applicable_ids=[],
        fixture=False,
    )
    validate_request(label)
    return label
