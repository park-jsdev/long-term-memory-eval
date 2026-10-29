"""Load LoCoMo-Plus cognitive items and count them before any judge.

The upstream file is a list of cue–trigger objects, not LoCoMo QA.
Their evaluation scripts stay out of this repo. Data derived from
LoCoMo is CC BY-NC and belongs in gitignored data/raw/.

A single inserted cue is one interaction. The count therefore reports
two-session support only when evidence ids already span two sessions.
That is the H2 gate. The trigger can still be an H1 request.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RELATION_TYPES = ("causal", "state", "goal", "value")
# Powercontext pinned this Locomo-Plus commit for the cognitive file.
PINNED_COMMIT = "059f4e3d38f7f1f96765e8e2cb7de3097551bffb"
_REQUIRED = ("relation_type", "cue_dialogue", "trigger_query", "time_gap")


@dataclass
class CognitiveItem:
    """One cognitive case, normalized from the upstream object."""

    index: int
    relation_type: str
    cue_dialogue: str
    trigger_query: str
    time_gap: str
    evidence_ids: list[str]
    cue_is_insertion: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def locomo_plus_url() -> str:
    return (
        "https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/"
        f"{PINNED_COMMIT}/data/locomo_plus.json"
    )


def fetch_locomo_plus(dest: Path) -> Path:
    """Download the pinned cognitive JSON. Does not commit it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(locomo_plus_url(), dest)
    return dest


def _session_keys(evidence_ids: list[str]) -> set[str]:
    keys = set()
    for evidence_id in evidence_ids:
        text = str(evidence_id)
        if ":" in text:
            keys.add(text.split(":", 1)[0])
        elif text:
            keys.add(text)
    return keys


def _as_cognitive(raw: dict[str, Any], index: int) -> CognitiveItem:
    for field in _REQUIRED:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"cognitive item {index} missing {field}")
    relation = raw["relation_type"].strip()
    if relation not in RELATION_TYPES:
        raise ValueError(f"cognitive item {index} bad relation_type {relation}")
    evidence = [str(item) for item in raw.get("evidence") or []]
    # A standalone cue was inserted. Evidence ids mean it was already
    # stitched into dialogue turns.
    return CognitiveItem(
        index=index,
        relation_type=relation,
        cue_dialogue=raw["cue_dialogue"].strip(),
        trigger_query=raw["trigger_query"].strip(),
        time_gap=raw["time_gap"].strip(),
        evidence_ids=evidence,
        cue_is_insertion=not evidence,
    )


def load_cognitive_items(path: Path) -> list[CognitiveItem]:
    """Read a cognitive list, or category-6 QA inside LoCoMo-shaped samples."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    if not data:
        return []
    if isinstance(data[0], dict) and "cue_dialogue" in data[0]:
        return [_as_cognitive(item, i) for i, item in enumerate(data)]
    items: list[CognitiveItem] = []
    for sample in data:
        if not isinstance(sample, dict):
            continue
        for qa in sample.get("qa") or []:
            if qa.get("category") not in (6, "6", "cognitive", "Cognitive"):
                continue
            raw = dict(qa)
            raw.setdefault("cue_dialogue", qa.get("cue_dialogue") or "")
            items.append(_as_cognitive(raw, len(items)))
    if not items:
        raise ValueError(f"No cognitive items found in {path}")
    return items


def count_cognitive_items(items: list[CognitiveItem]) -> dict[str, Any]:
    """The one-page count. No model calls."""
    by_relation = {name: 0 for name in RELATION_TYPES}
    two_session = 0
    explicit_single = 0
    insertions = 0
    for item in items:
        by_relation[item.relation_type] += 1
        if item.cue_is_insertion:
            insertions += 1
        if len(_session_keys(item.evidence_ids)) >= 2:
            two_session += 1
        else:
            explicit_single += 1
    return {
        "n_items": len(items),
        "by_relation": by_relation,
        "cue_is_insertion": insertions,
        "cue_is_original_span": len(items) - insertions,
        "two_session_support": two_session,
        "explicit_single_interaction": explicit_single,
        "note": (
            "A raw cue with no evidence ids is one inserted interaction. "
            "H2's inferred-pattern rule needs two sessions. The trigger "
            "can still be used as an H1 request."
        ),
    }


def trigger_text(item: CognitiveItem) -> str:
    """Drop a leading 'A:' so the request is the user's words."""
    text = item.trigger_query.strip()
    if text[:2].lower() == "a:":
        return text[2:].strip()
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Count LoCoMo-Plus cognitive items")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch and not args.path.is_file():
        fetch_locomo_plus(args.path)
    if not args.path.is_file():
        print(f"Not found: {args.path}")
        return 2
    report = count_cognitive_items(load_cognitive_items(args.path))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
