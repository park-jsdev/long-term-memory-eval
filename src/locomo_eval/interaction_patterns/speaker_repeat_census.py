"""List phrases that already recur across sessions. Not gold labels.

Run this before the two-conversation annotation pass. It estimates how
much repeated wording LoCoMo contains and gives annotators a recall
aid. A phrase in the output is not a behavioral hypothesis.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.locomo_eval.mem0_metrics import simple_tokenize

_STOP = frozenset(
    """
    a an the and or to of in on for with is it that this my i you we they
    was were be as at from by if but not so just have has had do did will
    would can could about into than then there their our your me he she
    his her said like really very also just going went get got one
    """.split()
)


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


def repeated_speaker_phrases(
    samples: list[dict],
    *,
    min_sessions: int = 2,
    min_len: int = 4,
    limit_per_speaker: int = 20,
) -> list[dict]:
    """Per speaker, unigrams that show up in at least two sessions.

    The return list is capped for reading. n_repeated_before_cap is the
    yield estimate. Session ids are strings of the LoCoMo session number.
    """
    rows = []
    for sample in samples:
        conversation = sample.get("conversation") or {}
        sample_id = str(sample.get("sample_id") or "")
        by_speaker: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for session_id, turns in _sessions(conversation):
            for turn in turns:
                speaker = str(turn.get("speaker") or "")
                if not speaker:
                    continue
                seen: set[str] = set()
                for tok in simple_tokenize(str(turn.get("text") or "")):
                    if len(tok) < min_len or tok in _STOP or tok in seen:
                        continue
                    seen.add(tok)
                    by_speaker[speaker][tok].add(session_id)
        for speaker, phrases in sorted(by_speaker.items()):
            repeated = [
                (phrase, sessions)
                for phrase, sessions in phrases.items()
                if len(sessions) >= min_sessions
            ]
            repeated.sort(key=lambda item: (-len(item[1]), item[0]))
            capped = repeated[:limit_per_speaker]
            for phrase, sessions in capped:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "speaker": speaker,
                        "phrase": phrase,
                        "session_ids": sorted(sessions, key=int),
                        "n_sessions": len(sessions),
                        "n_repeated_before_cap": len(repeated),
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="LoCoMo recurrence census")
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.locomo.is_file():
        print(f"LoCoMo JSON not found: {args.locomo}")
        return 2
    samples = json.loads(args.locomo.read_text(encoding="utf-8"))
    rows = repeated_speaker_phrases(samples)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    print(
        f"rows {len(rows)} speakers "
        f"{len({(r['sample_id'], r['speaker']) for r in rows})}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
