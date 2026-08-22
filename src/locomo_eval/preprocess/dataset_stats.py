"""Dataset histograms and naive retrieval bars for LoCoMo session documents."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.data.locomo import CATEGORY_NAMES

from .session_documents import (
    MemoryField,
    build_session_documents,
    join_qa_to_documents,
    naive_rank_session_ids,
    recall_evidence_sessions,
)

SINGLE_HOP = 4
MULTI_HOP = 1
K_SINGLE = 5
K_MULTI = 10


def _hist(path: Path, values: list[float | int], title: str, xlabel: str, bins: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    n_unique = len({int(v) for v in values}) if values else 1
    ax.hist(values, bins=min(bins, max(5, n_unique)), color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def retrieval_report(
    samples: list[dict[str, Any]],
    field: MemoryField,
    k_single: int = K_SINGLE,
    k_multi: int = K_MULTI,
) -> dict[str, Any]:
    """Question-only lexical retrieval. Gold answers are not in the query."""
    by_cat: dict[int, list[float]] = defaultdict(list)
    single: list[float] = []
    multi: list[float] = []
    n_oracle_ok = 0
    n_qa = 0
    for sample in samples:
        docs = build_session_documents(sample)
        for row in join_qa_to_documents(sample, docs):
            n_qa += 1
            if row.all_evidence_in_blocks:
                n_oracle_ok += 1
            ranked = naive_rank_session_ids(row.question, docs, field)
            if row.category == SINGLE_HOP and row.evidence_session_ids:
                rec = recall_evidence_sessions(row.evidence_session_ids, ranked, k_single)
                single.append(rec)
                by_cat[row.category].append(rec)
            elif row.category == MULTI_HOP and row.n_evidence_sessions >= 2:
                rec = recall_evidence_sessions(row.evidence_session_ids, ranked, k_multi)
                multi.append(rec)
                by_cat[row.category].append(rec)
            elif row.evidence_session_ids:
                rec = recall_evidence_sessions(row.evidence_session_ids, ranked, k_single)
                by_cat[row.category].append(rec)
    return {
        "field": field,
        "k_single_hop": k_single,
        "k_multi_hop": k_multi,
        "n_qa": n_qa,
        "oracle_join_rate": round(n_oracle_ok / n_qa, 4) if n_qa else None,
        "single_hop_recall": _mean(single),
        "n_single_hop": len(single),
        "multi_hop_recall": _mean(multi),
        "n_multi_hop_multi_session": len(multi),
        "recall_by_category": {
            CATEGORY_NAMES.get(c, str(c)): _mean(vals) for c, vals in sorted(by_cat.items())
        },
    }


def write_dataset_analysis(samples: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    sessions_per_sample: list[int] = []
    turns_per_session: list[int] = []
    chars_per_session: list[int] = []
    qa_per_sample: list[int] = []
    n_ev_sessions: list[int] = []
    cat_counts: dict[str, int] = defaultdict(int)

    for sample in samples:
        docs = build_session_documents(sample)
        sessions_per_sample.append(len(docs))
        for doc in docs:
            turns_per_session.append(doc.n_turns)
            chars_per_session.append(len(doc.turns_text()))
        joins = join_qa_to_documents(sample, docs)
        qa_per_sample.append(len(joins))
        for row in joins:
            cat_counts[row.category_name] += 1
            n_ev_sessions.append(row.n_evidence_sessions)

    _hist(plots / "sessions_per_sample.png", sessions_per_sample, "Sessions per sample", "n sessions")
    _hist(plots / "turns_per_session.png", turns_per_session, "Turns per session", "n turns")
    _hist(plots / "chars_per_session.png", chars_per_session, "Turn-text chars per session", "chars")
    _hist(plots / "qa_per_sample.png", qa_per_sample, "QA items per sample", "n QA")
    _hist(
        plots / "evidence_sessions_per_qa.png",
        n_ev_sessions,
        "Evidence sessions per QA",
        "n sessions in evidence",
        bins=10,
    )

    cat_labels = list(CATEGORY_NAMES.values())
    cat_vals = [cat_counts.get(name, 0) for name in cat_labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cat_labels, cat_vals, color="steelblue")
    ax.set_title("QA count by official category id")
    ax.set_ylabel("n questions")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots / "qa_by_category.png", dpi=120)
    plt.close(fig)

    agent = retrieval_report(samples, "turns")
    memory = retrieval_report(samples, "summary")
    labels = ["single_hop@5 turns", "multi_hop@10 turns", "single_hop@5 summary", "multi_hop@10 summary"]
    values = [
        agent.get("single_hop_recall") or 0.0,
        agent.get("multi_hop_recall") or 0.0,
        memory.get("single_hop_recall") or 0.0,
        memory.get("multi_hop_recall") or 0.0,
    ]
    _bar(
        plots / "naive_retrieval_recall.png",
        labels,
        values,
        "Naive lexical recall of evidence sessions (question only)",
        "recall",
    )

    summary = {
        "n_samples": len(samples),
        "n_sessions": sum(sessions_per_sample),
        "n_qa": sum(qa_per_sample),
        "qa_by_category": dict(cat_counts),
        "naive_agent_turns": agent,
        "naive_memory_summary": memory,
        "note": (
            "Category ids are official LoCoMo task_eval integers "
            "(1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop, 5=adversarial), "
            "not the paper prose numbering (1) single-hop … (5) adversarial."
        ),
    }
    (out / "dataset_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return plots
