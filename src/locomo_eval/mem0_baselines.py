"""Published Mem0 LoCoMo numbers for side-by-side sanity tables.

Source: Chhikara et al., *Mem0: Building Production-Ready AI Agents with
Scalable Long-Term Memory*, arXiv:2504.19413 (Tables 1–2).

F1 / BLEU-1 / J are percentages as printed in the paper. Latency is seconds.
``None`` means the paper did not report that cell.

These are **literature pins**, not live re-runs of Mem0's extract/update
pipeline. Use them in ``scripts.analysis.run_benchmark`` so our autorater
pack can be compared to published baselines.
"""

from __future__ import annotations

from typing import Any

# LoCoMo category ids used in this repo / Mem0 eval (skip 5).
MEM0_JUDGED_CATEGORIES = (1, 2, 3, 4)
ADVERSARIAL_CATEGORY = 5

PAPER_CITE = "Chhikara et al., arXiv:2504.19413"
PAPER_URL = "https://arxiv.org/abs/2504.19413"
MEM0_JUDGE_CODE_URL = (
    "https://github.com/mem0ai/mem0/blob/ece7ff6b/"
    "evaluation/metrics/llm_judge.py"
)


def _cat(
    *,
    f1: float | None,
    bleu1: float | None,
    j: float | None,
    j_std: float | None = None,
) -> dict[str, float | None]:
    return {"f1": f1, "bleu1": bleu1, "j": j, "j_std": j_std}


# Table 1: per question type. Keys match CATEGORY_NAMES in locomo_qa.py.
TABLE1_BY_CATEGORY: dict[str, dict[str, dict[str, float | None]]] = {
    "LoCoMo": {
        "single_hop": _cat(f1=25.02, bleu1=19.75, j=None),
        "multi_hop": _cat(f1=12.04, bleu1=11.16, j=None),
        "open_domain": _cat(f1=40.36, bleu1=29.05, j=None),
        "temporal": _cat(f1=18.41, bleu1=14.77, j=None),
    },
    "ReadAgent": {
        "single_hop": _cat(f1=9.15, bleu1=6.48, j=None),
        "multi_hop": _cat(f1=5.31, bleu1=5.12, j=None),
        "open_domain": _cat(f1=9.67, bleu1=7.66, j=None),
        "temporal": _cat(f1=12.60, bleu1=8.87, j=None),
    },
    "MemoryBank": {
        "single_hop": _cat(f1=5.00, bleu1=4.77, j=None),
        "multi_hop": _cat(f1=5.56, bleu1=5.94, j=None),
        "open_domain": _cat(f1=6.61, bleu1=5.16, j=None),
        "temporal": _cat(f1=9.68, bleu1=6.99, j=None),
    },
    "MemGPT": {
        "single_hop": _cat(f1=26.65, bleu1=17.72, j=None),
        "multi_hop": _cat(f1=9.15, bleu1=7.44, j=None),
        "open_domain": _cat(f1=41.04, bleu1=34.34, j=None),
        "temporal": _cat(f1=25.52, bleu1=19.44, j=None),
    },
    "A-Mem": {
        "single_hop": _cat(f1=27.02, bleu1=20.09, j=None),
        "multi_hop": _cat(f1=12.14, bleu1=12.00, j=None),
        "open_domain": _cat(f1=44.65, bleu1=37.06, j=None),
        "temporal": _cat(f1=45.85, bleu1=36.67, j=None),
    },
    "A-Mem*": {
        "single_hop": _cat(f1=20.76, bleu1=14.90, j=39.79, j_std=0.38),
        "multi_hop": _cat(f1=9.22, bleu1=8.81, j=18.85, j_std=0.31),
        "open_domain": _cat(f1=33.34, bleu1=27.58, j=54.05, j_std=0.22),
        "temporal": _cat(f1=35.40, bleu1=31.08, j=49.91, j_std=0.31),
    },
    "LangMem": {
        "single_hop": _cat(f1=35.51, bleu1=26.86, j=62.23, j_std=0.75),
        "multi_hop": _cat(f1=26.04, bleu1=22.32, j=47.92, j_std=0.47),
        "open_domain": _cat(f1=40.91, bleu1=33.63, j=71.12, j_std=0.20),
        "temporal": _cat(f1=30.75, bleu1=25.84, j=23.43, j_std=0.39),
    },
    "Zep": {
        "single_hop": _cat(f1=35.74, bleu1=23.30, j=61.70, j_std=0.32),
        "multi_hop": _cat(f1=19.37, bleu1=14.82, j=41.35, j_std=0.48),
        "open_domain": _cat(f1=49.56, bleu1=38.92, j=76.60, j_std=0.13),
        "temporal": _cat(f1=42.00, bleu1=34.53, j=49.31, j_std=0.50),
    },
    "OpenAI": {
        "single_hop": _cat(f1=34.30, bleu1=23.72, j=63.79, j_std=0.46),
        "multi_hop": _cat(f1=20.09, bleu1=15.42, j=42.92, j_std=0.63),
        "open_domain": _cat(f1=39.31, bleu1=31.16, j=62.29, j_std=0.12),
        "temporal": _cat(f1=14.04, bleu1=11.25, j=21.71, j_std=0.20),
    },
    "Mem0": {
        "single_hop": _cat(f1=38.72, bleu1=27.13, j=67.13, j_std=0.65),
        "multi_hop": _cat(f1=28.64, bleu1=21.58, j=51.15, j_std=0.31),
        "open_domain": _cat(f1=47.65, bleu1=38.72, j=72.93, j_std=0.11),
        "temporal": _cat(f1=48.93, bleu1=40.51, j=55.51, j_std=0.34),
    },
    "Mem0g": {
        "single_hop": _cat(f1=38.09, bleu1=26.03, j=65.71, j_std=0.45),
        "multi_hop": _cat(f1=24.32, bleu1=18.82, j=47.19, j_std=0.67),
        "open_domain": _cat(f1=49.27, bleu1=40.30, j=75.71, j_std=0.21),
        "temporal": _cat(f1=51.55, bleu1=40.28, j=58.13, j_std=0.44),
    },
}


# Table 2: overall J + latency. RAG rows are the paper's k / chunk-size grid.
TABLE2_OVERALL: list[dict[str, Any]] = [
    {
        "method": "RAG k=1 128",
        "memory_tokens": 128,
        "search_p50": 0.281,
        "search_p95": 0.823,
        "total_p50": 0.774,
        "total_p95": 1.825,
        "j": 47.77,
        "j_std": 0.23,
    },
    {
        "method": "RAG k=1 256",
        "memory_tokens": 256,
        "search_p50": 0.251,
        "search_p95": 0.710,
        "total_p50": 0.745,
        "total_p95": 1.628,
        "j": 50.15,
        "j_std": 0.16,
    },
    {
        "method": "RAG k=2 256",
        "memory_tokens": 256,
        "search_p50": 0.255,
        "search_p95": 0.699,
        "total_p50": 0.802,
        "total_p95": 1.907,
        "j": 60.97,
        "j_std": 0.20,
    },
    {
        "method": "Full-context",
        "memory_tokens": 26031,
        "search_p50": None,
        "search_p95": None,
        "total_p50": 9.870,
        "total_p95": 17.117,
        "j": 72.90,
        "j_std": 0.19,
    },
    {
        "method": "A-Mem",
        "memory_tokens": 2520,
        "search_p50": 0.668,
        "search_p95": 1.485,
        "total_p50": 1.410,
        "total_p95": 4.374,
        "j": 48.38,
        "j_std": 0.15,
    },
    {
        "method": "LangMem",
        "memory_tokens": 127,
        "search_p50": 17.99,
        "search_p95": 59.82,
        "total_p50": 18.53,
        "total_p95": 60.40,
        "j": 58.10,
        "j_std": 0.21,
    },
    {
        "method": "Zep",
        "memory_tokens": 3911,
        "search_p50": 0.513,
        "search_p95": 0.778,
        "total_p50": 1.292,
        "total_p95": 2.926,
        "j": 65.99,
        "j_std": 0.16,
    },
    {
        "method": "OpenAI",
        "memory_tokens": 4437,
        "search_p50": None,
        "search_p95": None,
        "total_p50": 0.466,
        "total_p95": 0.889,
        "j": 52.90,
        "j_std": 0.14,
    },
    {
        "method": "Mem0",
        "memory_tokens": 1764,
        "search_p50": 0.148,
        "search_p95": 0.200,
        "total_p50": 0.708,
        "total_p95": 1.440,
        "j": 66.88,
        "j_std": 0.15,
    },
    {
        "method": "Mem0g",
        "memory_tokens": 3616,
        "search_p50": 0.476,
        "search_p95": 0.657,
        "total_p50": 1.091,
        "total_p95": 2.590,
        "j": 68.44,
        "j_std": 0.17,
    },
]


# Compact set used on the J + latency comparison plot (paper Figure 4).
FIGURE4_METHODS = (
    "Full-context",
    "A-Mem",
    "LangMem",
    "Zep",
    "OpenAI",
    "Mem0",
    "Mem0g",
    "RAG k=2 256",
)

# Methods this repo can clone locally (eval_pipeline --method). External
# systems in FIGURE4 stay literature pins.
LOCAL_PAPER_METHODS = (
    "Full-context",
    "RAG k=2 256",
    "OpenAI",
    "Mem0",
    "Mem0g",
)

# memory_type / --label aliases → Table 2 method names. RAG k/chunk is
# filled in by ``paper_method_from_run`` when the label is just ``rag``.
MEMORY_TYPE_TO_PAPER_METHOD = {
    "full_context": "Full-context",
    "full-context": "Full-context",
    "openai_memory": "OpenAI",
    "openai": "OpenAI",
    "mem0": "Mem0",
    "mem0g": "Mem0g",
}


def table2_by_method() -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in TABLE2_OVERALL}


def literature_overall_j(method: str) -> float | None:
    row = table2_by_method().get(method)
    if not row:
        return None
    return row.get("j")


def rag_paper_method(k: int | None, chunk_size: int | None) -> str:
    """Table 2 RAG row name. Paper's strongest cell is k=2, chunk=256."""
    kk = 2 if k is None else int(k)
    size = 256 if chunk_size is None else int(chunk_size)
    return f"RAG k={kk} {size}"


def paper_method_from_label(label: str | None) -> str | None:
    """Map a run ``memory_type`` / autorater ``--label`` to a Table 2 name."""
    if not label:
        return None
    raw = str(label).strip()
    if raw in table2_by_method():
        return raw
    key = raw.lower().replace(" ", "_")
    if key in MEMORY_TYPE_TO_PAPER_METHOD:
        return MEMORY_TYPE_TO_PAPER_METHOD[key]
    if key == "rag" or key.startswith("rag_"):
        return None
    return None
