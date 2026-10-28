"""Validate Who/What/Why records and per-request If/When labels.

The annotation pass writes JSONL in this shape. A missing scope or an
inferred pattern with no evidence would make H2 unscorable, so those
rows fail here instead of becoming silent memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Closed inventory from the pilot rubric. Assistant style is rare in
# LoCoMo because both speakers are people.
WHAT_TYPES = (
    "recurring_activity",
    "recurring_goal",
    "stable_constraint",
    "taste_or_value",
    "updated_fact",
    "assistant_style",
)
WHO_BASES = ("explicit", "inferred")
IF_DECISIONS = ("select", "abstain")


@dataclass
class HypothesisRecord:
    """One scoped behavioral hypothesis (proposal H2).

    who_basis explicit may cite a single interaction. who_basis inferred
    needs evidence from two sessions. The harness reads these records.
    It does not invent them.
    """

    record_id: str
    speaker: str
    sample_id: str
    who_basis: str
    what_type: str
    what: str
    scope: str
    exceptions: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    uncertainty: str = ""
    fixture: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RequestLabel:
    """One held-out request and the decision a human says is justified.

    This is the H1 annotation unit. applicable_ids are gold. The stub
    decision does not read them, so a later prompt can be scored against
    them without having been trained on them.
    """

    request_id: str
    sample_id: str
    speaker: str
    request: str
    if_decision: str
    when: str
    how_then: str = ""
    applicable_ids: list[str] = field(default_factory=list)
    mandatory_constraints: list[str] = field(default_factory=list)
    fixture: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hypothesis_from_mapping(raw: dict[str, Any]) -> HypothesisRecord:
    """Build a record from JSON. Unknown keys are ignored on purpose."""
    known = HypothesisRecord.__dataclass_fields__
    data = {key: raw[key] for key in known if key in raw}
    return HypothesisRecord(**data)


def request_from_mapping(raw: dict[str, Any]) -> RequestLabel:
    known = RequestLabel.__dataclass_fields__
    data = {key: raw[key] for key in known if key in raw}
    return RequestLabel(**data)


def validate_hypothesis(record: HypothesisRecord) -> None:
    """Reject a row the H2 comparison could not interpret.

    An inferred pattern with no evidence ids looks like a supported
    memory and is not one. An empty scope makes Stage 1's applicability
    check meaningless.
    """
    if record.who_basis not in WHO_BASES:
        raise ValueError(f"{record.record_id}: who_basis must be explicit or inferred")
    if record.what_type not in WHAT_TYPES:
        raise ValueError(f"{record.record_id}: unknown what_type {record.what_type}")
    if not str(record.what).strip():
        raise ValueError(f"{record.record_id}: what is empty")
    if not str(record.scope).strip():
        raise ValueError(f"{record.record_id}: scope is empty")
    if not record.evidence_ids:
        raise ValueError(f"{record.record_id}: evidence ids are required")
    if record.who_basis == "inferred" and len(set(record.session_ids)) < 2:
        raise ValueError(
            f"{record.record_id}: inferred patterns need two sessions"
        )


def validate_request(label: RequestLabel) -> None:
    """Reject a request label that has no If/When decision."""
    if label.if_decision not in IF_DECISIONS:
        raise ValueError(f"{label.request_id}: if_decision must be select or abstain")
    if not str(label.when).strip():
        raise ValueError(f"{label.request_id}: when is empty")
    if not str(label.request).strip():
        raise ValueError(f"{label.request_id}: request text is empty")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
