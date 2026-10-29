"""Heuristic comparison of LoCoMo QA and LoCoMo-Plus cues.

These counts are not gold labels and not H1 or H2 results. They use the
same unigram overlap as the decision stub, so they show where that stub
can fire. A later annotator replaces the empty judgment fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.locomo import CATEGORY_NAMES
from src.data.locomo_plus import CognitiveItem, load_cognitive_items, trigger_text
from .condition_decisions import content_tokens
from .plus_item_to_record import WHAT_TYPE_FOR_RELATION
from .speaker_repeat_census import repeated_speaker_phrases

ROOT = Path(__file__).resolve().parents[3]


def overlap_size(left: str, right: str) -> int:
    """Content-word overlap. Zero means the stub's Stage 1 would abstain."""
    return len(content_tokens(left) & content_tokens(right))


def _sessions(conversation: dict) -> list[tuple[str, list[dict]]]:
    nums = sorted(
        int(key.split("_")[-1])
        for key in conversation
        if key.startswith("session_") and "date_time" not in key
    )
    out = []
    for num in nums:
        turns = conversation.get(f"session_{num}") or []
        if turns:
            out.append((str(num), turns))
    return out


def _evidence_text(conversation: dict, evidence_ids: list) -> str:
    wanted = {str(item) for item in evidence_ids}
    parts = []
    for _session_id, turns in _sessions(conversation):
        for turn in turns:
            if str(turn.get("dia_id") or "") in wanted:
                parts.append(str(turn.get("text") or ""))
    return " ".join(parts)


def plus_overlap_summary(items: list[CognitiveItem]) -> dict[str, Any]:
    """How often a Plus trigger shares content words with its cue."""
    overlaps = [overlap_size(item.cue_dialogue, trigger_text(item)) for item in items]
    zero = sum(1 for value in overlaps if value == 0)
    two_session = 0
    for item in items:
        prefixes = set()
        for evidence_id in item.evidence_ids:
            text = str(evidence_id)
            if text:
                prefixes.add(text.split(":", 1)[0])
        if len(prefixes) >= 2:
            two_session += 1
    return {
        "n_items": len(items),
        "n_zero_overlap": zero,
        "fraction_zero_overlap": (zero / len(items)) if items else 0.0,
        "mean_overlap": (sum(overlaps) / len(overlaps)) if overlaps else 0.0,
        "two_session_support": two_session,
    }


def locomo_qa_overlap_summary(samples: list[dict]) -> dict[str, Any]:
    """How often a LoCoMo question shares content words with its evidence turns."""
    by_category: dict[str, list[int]] = {}
    overlaps: list[int] = []
    n_sessions = []
    for sample in samples:
        conversation = sample.get("conversation") or {}
        n_sessions.append(len(_sessions(conversation)))
        for qa in sample.get("qa") or []:
            name = CATEGORY_NAMES.get(qa.get("category"), str(qa.get("category")))
            size = overlap_size(
                str(qa.get("question") or ""),
                _evidence_text(conversation, qa.get("evidence") or []),
            )
            overlaps.append(size)
            by_category.setdefault(name, []).append(size)
    positive = sum(1 for value in overlaps if value > 0)

    def _pack(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"n": 0, "fraction_overlap": 0.0, "mean_overlap": 0.0}
        hit = sum(1 for value in values if value > 0)
        return {
            "n": len(values),
            "fraction_overlap": hit / len(values),
            "mean_overlap": sum(values) / len(values),
        }

    return {
        "n_conversations": len(samples),
        "n_questions": len(overlaps),
        "mean_sessions": (sum(n_sessions) / len(n_sessions)) if n_sessions else 0.0,
        "fraction_question_overlaps_evidence": (positive / len(overlaps)) if overlaps else 0.0,
        "mean_overlap": (sum(overlaps) / len(overlaps)) if overlaps else 0.0,
        "by_category": {name: _pack(values) for name, values in sorted(by_category.items())},
    }


def census_summary(samples: list[dict]) -> dict[str, Any]:
    """Repeated words are an upper bound, not a preference count.

    Words such as "thanks" repeat for every speaker. Names repeat because
    people address each other. What remains is still ordinary vocabulary
    plus some life themes. Annotators have to separate those.
    """
    rows = repeated_speaker_phrases(samples, limit_per_speaker=800)
    names: dict[str, set[str]] = {}
    for sample in samples:
        conversation = sample.get("conversation") or {}
        names[str(sample.get("sample_id") or "")] = {
            str(conversation.get("speaker_a") or "").casefold(),
            str(conversation.get("speaker_b") or "").casefold(),
        }
    raw_counts: dict[tuple[str, str], int] = {}
    speakers_for_phrase: dict[str, set[tuple[str, str]]] = {}
    specific_counts: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        key = (row["sample_id"], row["speaker"])
        raw_counts[key] = int(row["n_repeated_before_cap"])
        speakers_for_phrase.setdefault(row["phrase"], set()).add(key)
    for row in rows:
        if row["phrase"] in names.get(row["sample_id"], set()):
            continue
        key = (row["sample_id"], row["speaker"])
        if len(speakers_for_phrase[row["phrase"]]) <= 3:
            specific_counts.setdefault(key, set()).add(row["phrase"])
    raw_values = sorted(raw_counts.values())
    specific_values = sorted(len(words) for words in specific_counts.values())
    shared = sorted(
        phrase
        for phrase, keys in speakers_for_phrase.items()
        if len(keys) == len(raw_counts) and raw_counts
    )
    return {
        "speakers": len(raw_values),
        "median_repeated_words_including_boilerplate": (
            raw_values[len(raw_values) // 2] if raw_values else 0
        ),
        "boilerplate_shared_by_every_speaker": shared[:12],
        "median_words_used_by_at_most_three_speakers_excluding_names": (
            specific_values[len(specific_values) // 2] if specific_values else 0
        ),
        "note": (
            "Neither median is a count of preferences. The first is dominated "
            "by chat wording. The second still mixes life themes with ordinary words."
        ),
    }


def plus_starter_rows(items: list[CognitiveItem], review_first: int = 20) -> list[dict]:
    """Sheets a person can edit. heuristic_if is unlabeled on purpose."""
    rows = []
    for item in items:
        cue_overlap = overlap_size(item.cue_dialogue, trigger_text(item))
        rows.append(
            {
                "record_id": f"plus-{item.index}",
                "source": "locomo_plus",
                "gold": False,
                "review_first": item.index < review_first,
                "relation_type": item.relation_type,
                "who_basis_suggestion": "explicit",
                "what_type_suggestion": WHAT_TYPE_FOR_RELATION[item.relation_type],
                "what": item.cue_dialogue,
                "scope": "",
                "evidence_note": "Single inserted cue. Not two sessions.",
                "time_gap": item.time_gap,
                "request": trigger_text(item),
                "heuristic_overlap": cue_overlap,
                "heuristic_if": "unlabeled",
                "annotator_if": "",
                "annotator_when": "",
                "annotator_notes": "",
            }
        )
    return rows


def compare(samples: list[dict], items: list[CognitiveItem]) -> dict[str, Any]:
    """Side-by-side counts for the dataset discussion. Not a study result."""
    return {
        "locomo_qa": locomo_qa_overlap_summary(samples),
        "locomo_dialogue_repetition": census_summary(samples),
        "locomo_plus": plus_overlap_summary(items),
        "reading": (
            "Question-evidence overlap is a fact-recall clue. "
            "Cue-trigger overlap near zero is the Plus design. "
            "Repeated words across sessions are H2 material in the dialogues, "
            "not in the Plus cues."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Heuristic LoCoMo vs LoCoMo-Plus counts")
    parser.add_argument("--locomo", type=Path, default=ROOT / "data" / "raw" / "locomo10.json")
    parser.add_argument("--plus", type=Path, default=ROOT / "data" / "raw" / "locomo_plus.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "annotations")
    args = parser.parse_args()
    if not args.locomo.is_file() or not args.plus.is_file():
        print("Need both locomo10.json and locomo_plus.json under data/raw/")
        return 2
    samples = json.loads(args.locomo.read_text(encoding="utf-8"))
    items = load_cognitive_items(args.plus)
    report = compare(samples, items)
    rows = plus_starter_rows(items)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "heuristic_compare.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (args.out_dir / "plus_starter.jsonl").write_text(text, encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"starter rows {len(rows)} review_first {sum(1 for row in rows if row['review_first'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
