"""Plain JSON loaders for LoCoMo conversations and QA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


CONV_START = (
    "Below is a conversation between two people: {a} and {b}. "
    "The conversation takes place over multiple days and the date of each "
    "conversation is written at the beginning of the conversation.\n\n"
)

# Matches LoCoMo paper / task_eval category ids.
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def load_locomo(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in {path}")
    return data


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Load prediction JSON in LoCoMo-style sample list format."""
    return load_locomo(path)


def format_conversation(conversation: dict[str, Any], include_captions: bool = True) -> str:
    """Flatten sessions into the text style used by LoCoMo HF eval."""
    session_nums = sorted(
        int(k.split("_")[-1])
        for k in conversation
        if k.startswith("session_") and "date_time" not in k
    )
    if not session_nums:
        return ""

    first = conversation[f"session_{session_nums[0]}"]
    speakers = list({turn["speaker"] for turn in first})
    a = speakers[0] if speakers else "A"
    b = speakers[1] if len(speakers) > 1 else "B"
    parts = [CONV_START.format(a=a, b=b)]

    for i in session_nums:
        sess_key = f"session_{i}"
        date_key = f"session_{i}_date_time"
        if sess_key not in conversation or not conversation[sess_key]:
            continue
        parts.append(f"DATE: {conversation.get(date_key, '')}\nCONVERSATION:\n")
        for turn in conversation[sess_key]:
            line = f'{turn["speaker"]} said, "{turn["text"]}"'
            if include_captions and turn.get("blip_caption"):
                line += f' and shared {turn["blip_caption"]}.'
            parts.append(line + "\n")
        parts.append("\n")
    return "".join(parts)


def iter_qa_examples(
    samples: list[dict[str, Any]],
    holdout_sample_ids: list[str] | None = None,
    split: str = "all",
) -> Iterator[dict[str, Any]]:
    """Yield flat QA examples with conversation context.

    split:
      - all: every sample
      - train: exclude holdout_sample_ids
      - eval: only holdout_sample_ids (or all if holdout empty)
    """
    holdout = set(holdout_sample_ids or [])
    for sample in samples:
        sid = sample["sample_id"]
        if split == "train" and sid in holdout:
            continue
        if split == "eval" and holdout and sid not in holdout:
            continue

        context = format_conversation(sample["conversation"])
        for qi, qa in enumerate(sample.get("qa", [])):
            yield {
                "sample_id": sid,
                "qa_index": qi,
                "question": qa["question"],
                "answer": qa.get("answer", ""),
                "category": qa["category"],
                "category_name": CATEGORY_NAMES.get(qa["category"], str(qa["category"])),
                "evidence": qa.get("evidence", []),
                "context": context,
            }
