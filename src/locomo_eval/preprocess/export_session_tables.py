"""Write inspectable session-document CSVs (no gold in session_documents.csv)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .session_documents import QaJoinRow, build_session_documents, join_qa_to_documents

PREVIEW = 240


def _pipe(values: Iterable[Any]) -> str:
    return "|".join(str(v) for v in values)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _qa_csv_row(row: QaJoinRow) -> dict[str, Any]:
    return {
        "sample_id": row.sample_id,
        "question_id": row.question_id,
        "qa_index": row.qa_index,
        "category": row.category,
        "category_name": row.category_name,
        "question": row.question,
        "answer": row.answer,
        "evidence": _pipe(row.evidence),
        "evidence_session_ids": _pipe(row.evidence_session_ids),
        "n_evidence_sessions": row.n_evidence_sessions,
        "evidence_dia_ids_found": row.evidence_dia_ids_found,
        "all_evidence_in_blocks": row.all_evidence_in_blocks,
    }


def export_tables(samples: list[dict[str, Any]], out_dir: str | Path) -> dict[str, int]:
    out = Path(out_dir)
    session_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    for sample in samples:
        docs = build_session_documents(sample)
        raw_conv = sample.get("conversation") or {}
        for doc in docs:
            session_rows.append(
                {
                    "sample_id": doc.sample_id,
                    "session_id": doc.session_id,
                    "session_index": doc.session_index,
                    "source_key": doc.source_key,
                    "date_time_raw": doc.date_time_raw,
                    "date_time_normalized": doc.date_time_normalized or "",
                    "speaker_a": doc.speaker_a,
                    "speaker_b": doc.speaker_b,
                    "n_turns": doc.n_turns,
                    "n_images": doc.n_images,
                    "n_observations": len(doc.observations),
                    "n_events_a": len(doc.events_a),
                    "n_events_b": len(doc.events_b),
                    "event_date": doc.event_date,
                    "dia_ids": _pipe(doc.dia_ids),
                    "session_summary_chars": len(doc.session_summary),
                    "session_summary_preview": doc.session_summary[:PREVIEW].replace("\n", " "),
                    "turns_preview": doc.turns_text()[:PREVIEW].replace("\n", " / "),
                }
            )
            raw_turns = raw_conv.get(doc.source_key) or []
            for i, turn in enumerate(raw_turns):
                if not isinstance(turn, dict):
                    continue
                turn_rows.append(
                    {
                        "sample_id": doc.sample_id,
                        "session_id": doc.session_id,
                        "turn_index": i,
                        "dia_id": turn.get("dia_id", ""),
                        "speaker": turn.get("speaker", ""),
                        "text": turn.get("text", ""),
                        "has_image": bool(turn.get("blip_caption") or turn.get("img_url")),
                        "blip_caption": turn.get("blip_caption") or "",
                        "img_url": turn.get("img_url") or "",
                    }
                )
                if turn.get("blip_caption") or turn.get("img_url"):
                    image_rows.append(
                        {
                            "sample_id": doc.sample_id,
                            "session_id": doc.session_id,
                            "dia_id": turn.get("dia_id", ""),
                            "speaker": turn.get("speaker", ""),
                            "blip_caption": turn.get("blip_caption") or "",
                            "img_url": turn.get("img_url") or "",
                            "img_query": turn.get("query") or turn.get("img_query") or "",
                        }
                    )
            for obs in doc.observations:
                obs_rows.append(
                    {
                        "sample_id": doc.sample_id,
                        "session_id": doc.session_id,
                        "speaker": obs["speaker"],
                        "source_dia_id": obs["source_dia_id"],
                        "text": obs["text"],
                    }
                )
            for speaker, events in (
                (doc.speaker_a, doc.events_a),
                (doc.speaker_b, doc.events_b),
            ):
                for text in events:
                    event_rows.append(
                        {
                            "sample_id": doc.sample_id,
                            "session_id": doc.session_id,
                            "speaker": speaker,
                            "event_date": doc.event_date,
                            "text": text,
                        }
                    )
        for row in join_qa_to_documents(sample, docs):
            qa_rows.append(_qa_csv_row(row))

    _write_csv(
        out / "session_documents.csv",
        [
            "sample_id",
            "session_id",
            "session_index",
            "source_key",
            "date_time_raw",
            "date_time_normalized",
            "speaker_a",
            "speaker_b",
            "n_turns",
            "n_images",
            "n_observations",
            "n_events_a",
            "n_events_b",
            "event_date",
            "dia_ids",
            "session_summary_chars",
            "session_summary_preview",
            "turns_preview",
        ],
        session_rows,
    )
    _write_csv(
        out / "turns.csv",
        [
            "sample_id",
            "session_id",
            "turn_index",
            "dia_id",
            "speaker",
            "text",
            "has_image",
            "blip_caption",
            "img_url",
        ],
        turn_rows,
    )
    _write_csv(
        out / "qa_joined.csv",
        [
            "sample_id",
            "question_id",
            "qa_index",
            "category",
            "category_name",
            "question",
            "answer",
            "evidence",
            "evidence_session_ids",
            "n_evidence_sessions",
            "evidence_dia_ids_found",
            "all_evidence_in_blocks",
        ],
        qa_rows,
    )
    _write_csv(
        out / "observations.csv",
        ["sample_id", "session_id", "speaker", "source_dia_id", "text"],
        obs_rows,
    )
    _write_csv(
        out / "events.csv",
        ["sample_id", "session_id", "speaker", "event_date", "text"],
        event_rows,
    )
    _write_csv(
        out / "images.csv",
        ["sample_id", "session_id", "dia_id", "speaker", "blip_caption", "img_url", "img_query"],
        image_rows,
    )
    return {
        "n_samples": len(samples),
        "n_sessions": len(session_rows),
        "n_turns": len(turn_rows),
        "n_qa": len(qa_rows),
        "n_observations": len(obs_rows),
        "n_events": len(event_rows),
        "n_images": len(image_rows),
    }
