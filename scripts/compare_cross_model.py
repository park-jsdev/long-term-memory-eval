"""Compare runs that differ by reader and/or teacher model.

Unlike ``compare_runs.py`` (freeze bottom, vary memory), this script *expects*
model identity to change. It reports answer agreement, score deltas, cache-key
distinctness, and whether logs recorded the swap.

Foundation for later cross-model robustness tables (same memory, different
answer LLM; or same reader, different teacher within a family).

    python scripts/compare_cross_model.py \\
        --runs experiments/c1_mini experiments/c1_luna \\
        --axis reader --out experiments/compare_reader_mini_luna

    python scripts/compare_cross_model.py \\
        --runs experiments/teacher_mini experiments/teacher_luna \\
        --axis teacher --out experiments/compare_teacher_mini_luna
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_runs import load_pack  # noqa: E402
from src.locomo_eval.cache import ResponseCache  # noqa: E402
from src.locomo_eval.metrics import score_row  # noqa: E402
from src.locomo_eval.models import resolve_model  # noqa: E402
from src.locomo_eval.prompts import load_prompt_template, render_qa_prompt  # noqa: E402


def _meta_model(pack: dict, field: str) -> str | None:
    m = pack["metrics"].get(field) or pack["meta"].get(field)
    if m:
        return str(m)
    preds = pack["predictions"]
    if preds and preds[0].get(field):
        return str(preds[0][field])
    return None


def theoretical_cache_key(row: dict, meta: dict, template: str) -> str:
    """Rebuild the reader cache key from stored memory + model (no API)."""
    model = row.get("reader_model") or meta.get("reader_model")
    spec = resolve_model(str(model))
    payload = {
        "provider": "openai",
        "model": spec.model_id,
        "family": spec.family,
        "temperature": float(meta.get("temperature", 0.0) or 0.0),
        "max_tokens": int(meta.get("max_tokens", 64) or 64),
        "max_tokens_field": spec.max_tokens_field,
        "reasoning_effort": spec.reasoning_effort,
        "role": "reader",
        "prompt": render_qa_prompt(
            template, row.get("memory_text") or "", row.get("question") or ""
        ),
    }
    return ResponseCache.make_key(payload)


def cross_model_analysis(
    packs: list[dict],
    prompt_path: Path | None,
    axis: str,
) -> dict[str, Any]:
    """Pairwise robustness stats for the first two run packs."""
    if len(packs) < 2:
        return {"note": "Need ≥2 runs for pairwise analysis.", "axis": axis}

    a, b = packs[0], packs[1]
    common = sorted(set(a["by_qid"]) & set(b["by_qid"]))
    template = None
    if prompt_path and prompt_path.is_file():
        _, template = load_prompt_template(prompt_path)

    reader_a = _meta_model(a, "reader_model")
    reader_b = _meta_model(b, "reader_model")
    teacher_a = _meta_model(a, "teacher_model")
    teacher_b = _meta_model(b, "teacher_model")

    same_memory = same_answer = same_cache = 0
    deltas: list[float] = []
    paired_rows: list[dict[str, Any]] = []

    for qid in common:
        ra, rb = a["by_qid"][qid], b["by_qid"][qid]
        ma, mb = ra.get("memory_text") or "", rb.get("memory_text") or ""
        pa, pb = ra.get("predicted_answer") or "", rb.get("predicted_answer") or ""
        if ma == mb:
            same_memory += 1
        if pa == pb:
            same_answer += 1
        key_a = key_b = None
        if template is not None:
            key_a = theoretical_cache_key(ra, a["meta"], template)
            key_b = theoretical_cache_key(rb, b["meta"], template)
            if key_a == key_b:
                same_cache += 1
        sa = score_row(pa, str(ra.get("reference_answer", "")), int(ra.get("category", 0)))
        sb = score_row(pb, str(rb.get("reference_answer", "")), int(rb.get("category", 0)))
        delta = sb["locomo_f1"] - sa["locomo_f1"]
        deltas.append(delta)
        paired_rows.append(
            {
                "question_id": qid,
                "category": ra.get("category"),
                "question": ra.get("question"),
                "reference_answer": ra.get("reference_answer"),
                "pred_a": pa,
                "pred_b": pb,
                "same_answer": pa == pb,
                "same_memory_text": ma == mb,
                "locomo_f1_a": round(sa["locomo_f1"], 4),
                "locomo_f1_b": round(sb["locomo_f1"], 4),
                "delta_locomo_f1_b_minus_a": round(delta, 4),
                "reader_model_a": ra.get("reader_model") or reader_a,
                "reader_model_b": rb.get("reader_model") or reader_b,
                "teacher_model_a": ra.get("teacher_model") or teacher_a,
                "teacher_model_b": rb.get("teacher_model") or teacher_b,
                "same_cache_key": key_a == key_b if key_a else None,
            }
        )

    n = len(common) or 1
    frac_same_mem = round(same_memory / n, 4) if common else None
    frac_same_ans = round(same_answer / n, 4) if common else None
    frac_same_key = round(same_cache / n, 4) if common and template else None

    reader_differ = bool(reader_a and reader_b and reader_a != reader_b)
    teacher_differ = bool(teacher_a and teacher_b and teacher_a != teacher_b)
    same_reader_family = (
        resolve_model(reader_a).family == resolve_model(reader_b).family
        if reader_a and reader_b
        else None
    )
    same_teacher_family = (
        resolve_model(teacher_a).family == resolve_model(teacher_b).family
        if teacher_a and teacher_b
        else None
    )

    sanity: dict[str, Any] = {
        "logs_record_reader_models": bool(reader_a and reader_b),
        "reader_models_differ": reader_differ,
        "same_reader_family": same_reader_family,
        "logs_record_teacher_models": bool(teacher_a or teacher_b),
        "teacher_models_differ": teacher_differ,
        "same_teacher_family": same_teacher_family,
        "cache_keys_mostly_distinct": (frac_same_key is not None and frac_same_key < 0.05),
    }
    if axis == "reader":
        sanity["expected"] = (
            "Same memory (frozen middle); different reader_model in logs; "
            "different cache keys; answers/scores may differ."
        )
        sanity["memory_held_fixed"] = frac_same_mem == 1.0 if common else None
    elif axis == "teacher":
        sanity["expected"] = (
            "Different teacher_model in logs; memory text should differ; "
            "reader_model should match if the sandwich bottom was frozen."
        )
        sanity["reader_held_fixed"] = reader_a == reader_b
        sanity["memory_texts_differ"] = (
            frac_same_mem is not None and frac_same_mem < 0.05
        )

    return {
        "axis": axis,
        "run_a": a["run_id"],
        "run_b": b["run_id"],
        "reader_model_a": reader_a,
        "reader_model_b": reader_b,
        "teacher_model_a": teacher_a,
        "teacher_model_b": teacher_b,
        "n_common": len(common),
        "fraction_same_memory_text": frac_same_mem,
        "fraction_same_answer": frac_same_ans,
        "fraction_same_cache_key": frac_same_key,
        "n_answer_disagreements": (len(common) - same_answer) if common else 0,
        "mean_delta_locomo_f1_b_minus_a": round(sum(deltas) / n, 4) if common else None,
        "sanity": sanity,
        "paired_rows": paired_rows,
    }


def write_overall_csv(path: Path, packs: list[dict]) -> None:
    fields = [
        "run_id",
        "memory_type",
        "reader_model",
        "reader_family",
        "teacher_model",
        "teacher_family",
        "n",
        "exact_match",
        "token_f1",
        "locomo_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pack in packs:
            m = pack["metrics"]
            meta = pack["meta"]
            overall = m.get("metrics") or {}
            reader = _meta_model(pack, "reader_model")
            teacher = _meta_model(pack, "teacher_model")
            w.writerow(
                {
                    "run_id": pack["run_id"],
                    "memory_type": m.get("memory_type") or meta.get("memory_type"),
                    "reader_model": reader,
                    "reader_family": resolve_model(reader).family if reader else None,
                    "teacher_model": teacher,
                    "teacher_family": resolve_model(teacher).family if teacher else None,
                    "n": m.get("number_of_questions"),
                    "exact_match": overall.get("exact_match"),
                    "token_f1": overall.get("token_f1"),
                    "locomo_f1": overall.get("locomo_f1"),
                }
            )


def write_paired_csv(path: Path, paired: dict) -> None:
    rows = paired.get("paired_rows") or []
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_text_report(path: Path, packs: list[dict], paired: dict) -> None:
    lines = ["# Cross-model robustness", "", f"Axis: **{paired.get('axis')}**", ""]
    for pack in packs:
        overall = (pack["metrics"].get("metrics") or {})
        lines.append(f"### {pack['run_id']}")
        lines.append(f"- reader_model: `{_meta_model(pack, 'reader_model')}`")
        lines.append(f"- teacher_model: `{_meta_model(pack, 'teacher_model')}`")
        lines.append(f"- memory_type: `{pack['metrics'].get('memory_type') or pack['meta'].get('memory_type')}`")
        lines.append(
            f"- scores: locomo_f1={overall.get('locomo_f1')}, "
            f"token_f1={overall.get('token_f1')}, exact_match={overall.get('exact_match')}"
        )
        lines.append("")
    if paired.get("n_common") is None and "note" in paired:
        lines.append(paired["note"])
    else:
        lines.append("## Pairwise")
        lines.append(f"- n_common: {paired.get('n_common')}")
        lines.append(f"- fraction_same_memory_text: {paired.get('fraction_same_memory_text')}")
        lines.append(f"- fraction_same_answer: {paired.get('fraction_same_answer')}")
        lines.append(f"- n_answer_disagreements: {paired.get('n_answer_disagreements')}")
        lines.append(f"- fraction_same_cache_key: {paired.get('fraction_same_cache_key')}")
        lines.append(f"- mean_delta_locomo_f1 (B−A): {paired.get('mean_delta_locomo_f1_b_minus_a')}")
        lines.append("")
        lines.append("## Sanity")
        for k, v in (paired.get("sanity") or {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="experiments/<run_id> dirs")
    p.add_argument("--out", default="experiments/compare_cross_model")
    p.add_argument(
        "--axis",
        choices=["reader", "teacher", "both"],
        default="reader",
        help="Which model slot you intended to swap (guides sanity checks)",
    )
    p.add_argument("--prompt", default="prompts/qa_v1.txt")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    packs = [load_pack(Path(r)) for r in args.runs]
    write_overall_csv(out / "overall.csv", packs)
    paired = cross_model_analysis(packs, Path(args.prompt), args.axis)
    write_paired_csv(out / "paired_questions.csv", paired)
    slim = {k: v for k, v in paired.items() if k != "paired_rows"}
    write_text_report(out / "SUMMARY.md", packs, slim)
    (out / "compare.json").write_text(json.dumps({"runs": [
        {"dir": pack["dir"], "run_id": pack["run_id"], "meta": pack["meta"],
         "metrics": pack["metrics"]}
        for pack in packs
    ], "paired": slim}, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'overall.csv'}")
    print(f"Wrote {out / 'SUMMARY.md'}")
    print(
        f"axis={args.axis} same_answer={slim.get('fraction_same_answer')} "
        f"same_memory={slim.get('fraction_same_memory_text')} "
        f"same_cache_key={slim.get('fraction_same_cache_key')}"
    )


if __name__ == "__main__":
    main()
