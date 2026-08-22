"""Join LoCoMo fields onto session documents for audit and naive retrieval.

SessionDocument is the preprocess unit (one LoCoMo session_N). QA gold, observations,
and event_summary are *joined* for human verification — they are not copied into
SessionBlock (which stays gold-free for teachers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from src.data.locomo import CATEGORY_NAMES
from src.locomo_eval.preprocess.data_ingestor import DataIngestor
from src.locomo_eval.preprocess.preprocessing_pipeline import PreprocessingPipeline
from src.locomo_eval.schemas import SessionBlock

_DIA = re.compile(r"^D(\d+):(\S+)$")
_DIA_ALT = re.compile(r"^D:(\d+):(\S+)$")
_EV_SPLIT = re.compile(r"[;\s]+")
_SESSION_KEY = re.compile(r"^session_(\d+)$")
_TOKEN = re.compile(r"[a-z0-9]+")

MemoryField = Literal["turns", "summary"]


def parse_dia_id(dia_id: str) -> tuple[int, str] | None:
    """LoCoMo ``D{session}:{turn}`` → (session_id, turn_token).

    Also accepts ``D:11:26`` (extra colon) and strips leading zeros on the turn
    (``D30:05`` → session 30, turn ``5``).
    """
    s = str(dia_id).strip()
    m = _DIA.match(s) or _DIA_ALT.match(s)
    if not m:
        return None
    turn = m.group(2)
    if turn.isdigit():
        turn = str(int(turn))
    return int(m.group(1)), turn


def canonical_dia_id(dia_id: str) -> str | None:
    parsed = parse_dia_id(dia_id)
    if parsed is None:
        return None
    sid, turn = parsed
    return f"D{sid}:{turn}"


def evidence_tokens(evidence: list[str]) -> list[str]:
    """Split packed evidence strings (``D8:6; D9:17`` or space-separated)."""
    tokens: list[str] = []
    for raw in evidence:
        for part in _EV_SPLIT.split(str(raw).strip()):
            if part and part != "D":
                tokens.append(part)
    return tokens


def evidence_session_ids(evidence: list[str]) -> list[int]:
    sids: list[int] = []
    seen: set[int] = set()
    for raw in evidence_tokens(evidence):
        parsed = parse_dia_id(raw)
        if parsed is None:
            continue
        sid, _ = parsed
        if sid not in seen:
            seen.add(sid)
            sids.append(sid)
    return sids


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


@dataclass
class SessionDocument:
    """One session plus joined supplements (summary / obs / events / images)."""

    sample_id: str
    session_id: int
    session_index: int
    source_key: str
    date_time_raw: str
    date_time_normalized: str | None
    speaker_a: str
    speaker_b: str
    n_turns: int
    n_images: int
    dia_ids: list[str]
    turn_texts: list[str]
    session_summary: str
    observations: list[dict[str, str]]
    events_a: list[str]
    events_b: list[str]
    event_date: str
    schema_version: str = "preprocess_io.v1"

    def turns_text(self) -> str:
        return "\n".join(self.turn_texts)

    def retrieval_text(self, field: MemoryField) -> str:
        if field == "summary" and self.session_summary.strip():
            return self.session_summary
        return self.turns_text()


@dataclass
class QaJoinRow:
    """QA gold joined to session documents via evidence dia_ids (audit / oracle)."""

    sample_id: str
    question_id: str
    qa_index: int
    category: int
    category_name: str
    question: str
    answer: str
    evidence: list[str]
    evidence_session_ids: list[int]
    n_evidence_sessions: int
    evidence_dia_ids_found: int
    all_evidence_in_blocks: bool


def _session_nums(conversation: dict[str, Any]) -> list[int]:
    nums = []
    for key, value in conversation.items():
        m = _SESSION_KEY.match(key)
        if m and isinstance(value, list):
            nums.append(int(m.group(1)))
    return sorted(nums)


def _obs_rows(block: dict[str, Any] | None, session_id: int) -> list[dict[str, str]]:
    if not isinstance(block, dict):
        return []
    rows: list[dict[str, str]] = []
    for speaker, items in block.items():
        if not isinstance(items, list):
            continue
        for item in items:
            text, dia = "", ""
            if isinstance(item, (list, tuple)) and item:
                text = str(item[0])
                dia = str(item[1]) if len(item) > 1 else ""
            else:
                text = str(item)
            rows.append(
                {
                    "speaker": str(speaker),
                    "text": text,
                    "source_dia_id": dia,
                    "session_id": str(session_id),
                }
            )
    return rows


def _event_payload(event_summary: dict[str, Any] | None, session_id: int) -> dict[str, Any]:
    if not isinstance(event_summary, dict):
        return {"date": "", "by_speaker": {}}
    payload = event_summary.get(f"events_session_{session_id}")
    if not isinstance(payload, dict):
        return {"date": "", "by_speaker": {}}
    date = str(payload.get("date") or "")
    by_speaker = {
        str(k): [str(x) for x in v]
        for k, v in payload.items()
        if k != "date" and isinstance(v, list)
    }
    return {"date": date, "by_speaker": by_speaker}


def build_session_documents(sample: dict[str, Any]) -> list[SessionDocument]:
    """Process one raw sample into session documents with joined supplements."""
    conv = DataIngestor().ingest_sample(sample)
    processed = PreprocessingPipeline().process(conv)
    raw_conv = sample.get("conversation") or {}
    summaries = sample.get("session_summary") or {}
    observations = sample.get("observation") or {}
    event_summary = sample.get("event_summary") or {}

    docs: list[SessionDocument] = []
    for block in processed.session_blocks:
        raw_turns = raw_conv.get(block.source_key) or []
        n_images = sum(
            1
            for t in raw_turns
            if isinstance(t, dict) and (t.get("blip_caption") or t.get("img_url"))
        )
        obs = _obs_rows(observations.get(f"session_{block.session_id}_observation"), block.session_id)
        ev = _event_payload(event_summary if isinstance(event_summary, dict) else None, block.session_id)
        events_a = list(ev["by_speaker"].get(block.speaker_a, []))
        events_b = list(ev["by_speaker"].get(block.speaker_b, []))
        docs.append(
            SessionDocument(
                sample_id=block.sample_id,
                session_id=block.session_id,
                session_index=block.session_index,
                source_key=block.source_key,
                date_time_raw=block.date_time_raw,
                date_time_normalized=block.date_time_normalized,
                speaker_a=block.speaker_a,
                speaker_b=block.speaker_b,
                n_turns=len(block.turns),
                n_images=n_images,
                dia_ids=[t.source_dia_id for t in block.turns],
                turn_texts=[t.text for t in block.turns],
                session_summary=str(summaries.get(f"session_{block.session_id}_summary") or ""),
                observations=obs,
                events_a=events_a,
                events_b=events_b,
                event_date=str(ev["date"]),
                schema_version=block.schema_version,
            )
        )
    return docs


def count_raw_sessions(sample: dict[str, Any]) -> int:
    return len(_session_nums(sample.get("conversation") or {}))


def join_qa_to_documents(
    sample: dict[str, Any],
    documents: list[SessionDocument],
) -> list[QaJoinRow]:
    """Attach gold QA to session docs via evidence dia_ids. Oracle join, not a model."""
    by_sid = {d.session_id: d for d in documents}
    dia_index: dict[str, int] = {}
    for doc in documents:
        for dia in doc.dia_ids:
            if not dia:
                continue
            dia_index[dia] = doc.session_id
            canon = canonical_dia_id(dia)
            if canon:
                dia_index[canon] = doc.session_id
    sample_id = str(sample["sample_id"])
    rows: list[QaJoinRow] = []
    for i, qa in enumerate(sample.get("qa") or []):
        evidence = [str(e) for e in (qa.get("evidence") or [])]
        tokens = evidence_tokens(evidence)
        e_sids = evidence_session_ids(evidence)
        found = 0
        for dia in tokens:
            canon = canonical_dia_id(dia) or dia
            sid = dia_index.get(canon) or dia_index.get(dia)
            doc = by_sid.get(sid) if sid is not None else None
            if doc is None:
                continue
            doc_canons = {canonical_dia_id(x) or x for x in doc.dia_ids}
            if dia in doc.dia_ids or canon in doc_canons:
                found += 1
        rows.append(
            QaJoinRow(
                sample_id=sample_id,
                question_id=f"{sample_id}-q-{i}",
                qa_index=i,
                category=int(qa["category"]),
                category_name=CATEGORY_NAMES.get(int(qa["category"]), str(qa["category"])),
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer") or qa.get("adversarial_answer") or ""),
                evidence=evidence,
                evidence_session_ids=e_sids,
                n_evidence_sessions=len(e_sids),
                evidence_dia_ids_found=found,
                all_evidence_in_blocks=found == len(tokens) if tokens else True,
            )
        )
    return rows


def naive_rank_session_ids(
    query: str,
    documents: list[SessionDocument],
    field: MemoryField,
) -> list[int]:
    """Lexical overlap of the *question* with session text. Does not use gold answers."""
    q = _tokens(query)
    scored: list[tuple[int, int, int]] = []
    for doc in documents:
        overlap = len(q & _tokens(doc.retrieval_text(field)))
        scored.append((overlap, -doc.session_index, doc.session_id))
    scored.sort(reverse=True)
    return [sid for _, _, sid in scored]


def recall_evidence_sessions(
    evidence_sids: list[int],
    ranked_sids: list[int],
    k: int,
) -> float:
    if not evidence_sids:
        return 1.0
    top = set(ranked_sids[:k])
    return sum(1 for sid in evidence_sids if sid in top) / len(evidence_sids)


def block_matches_raw_session(block: SessionBlock, raw_turns: list[dict[str, Any]]) -> bool:
    if len(block.turns) != len(raw_turns):
        return False
    for turn, raw in zip(block.turns, raw_turns):
        if turn.source_dia_id != str(raw.get("dia_id", "")):
            return False
        if turn.text != str(raw.get("text", "")):
            return False
        if turn.speaker_raw != str(raw.get("speaker", "")):
            return False
    return True
