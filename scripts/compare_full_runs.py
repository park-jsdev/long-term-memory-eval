"""Sandwich experiment report for two (or more) full run packs.

Use this after two ``python -m src.locomo_eval.run`` calls. It reads
``experiments/<run_id>/`` (predictions + metrics + run_meta) and writes
overall/category tables, memory-text sanity, SUMMARY.md, and the
LoCoMo F1 boxplot/histograms from ``scripts.analysis.compare_predictions``.

For JSONL-only LoCoMo F1 plots (no sandwich SUMMARY), use
``python -m scripts.analysis.compare_predictions`` instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.compare_predictions import write_compare_prediction_plots
from src.locomo_eval.experiments.load import load_qa_pack
from src.locomo_eval.metrics import score_row
from src.locomo_eval.prompts import load_prompt_template
from src.locomo_eval.stats import mean_ci95, wilcoxon_signed_rank


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_pack(run_dir: Path) -> dict[str, Any]:
    return load_qa_pack(run_dir)


def resolve_prompt_path(packs: list[dict], explicit: str | Path | None = None) -> Path | None:
    """Use an explicit prompt or infer the one frozen across run metadata."""
    if explicit:
        return Path(explicit)
    prompt_paths = {
        str(pack["meta"].get("prompt_path"))
        for pack in packs
        if pack["meta"].get("prompt_path")
    }
    if not prompt_paths:
        return None
    if len(prompt_paths) != 1:
        raise ValueError(
            "Run packs record different prompt paths; comparison requires one "
            "frozen answer prompt."
        )
    return Path(prompt_paths.pop())


def write_overall_csv(path: Path, packs: list[dict]) -> None:
    fieldnames = [
        "run_dir",
        "run_id",
        "memory_type",
        "reader_model",
        "prompt_version",
        "n",
        "exact_match",
        "token_f1",
        "locomo_f1",
        "n_new_api_calls",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pack in packs:
            m = pack["metrics"]
            meta = pack["meta"]
            overall = m.get("metrics") or {}
            w.writerow(
                {
                    "run_dir": pack["dir"],
                    "run_id": pack["run_id"],
                    "memory_type": m.get("memory_type") or meta.get("memory_type"),
                    "reader_model": m.get("reader_model") or meta.get("reader_model"),
                    "prompt_version": m.get("prompt_version") or meta.get("prompt_version"),
                    "n": m.get("number_of_questions"),
                    "exact_match": overall.get("exact_match"),
                    "token_f1": overall.get("token_f1"),
                    "locomo_f1": overall.get("locomo_f1"),
                    "n_new_api_calls": meta.get("n_new_api_calls"),
                }
            )


def write_category_csv(path: Path, packs: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_id",
                "memory_type",
                "category",
                "n",
                "exact_match",
                "token_f1",
                "locomo_f1",
            ]
        )
        for pack in packs:
            m = pack["metrics"]
            meta = pack["meta"]
            mem = m.get("memory_type") or meta.get("memory_type")
            for cat, stats in (m.get("by_category") or {}).items():
                w.writerow(
                    [
                        pack["run_id"],
                        mem,
                        cat,
                        stats.get("n"),
                        stats.get("exact_match"),
                        stats.get("token_f1"),
                        stats.get("locomo_f1"),
                    ]
                )


def pair_analysis(packs: list[dict], prompt_path: Path | None) -> dict[str, Any]:
    """Pairwise sanity for the first two runs: memory-text and answer distinctness."""
    if len(packs) < 2:
        return {"note": "Need ≥2 runs for pairwise analysis."}

    a, b = packs[0], packs[1]
    common = sorted(set(a["by_qid"]) & set(b["by_qid"]))
    only_a = sorted(set(a["by_qid"]) - set(b["by_qid"]))
    only_b = sorted(set(b["by_qid"]) - set(a["by_qid"]))

    same_memory = 0
    same_answer = 0
    mem_chars_a = []
    mem_chars_b = []
    deltas = []
    paired_rows = []

    for qid in common:
        ra, rb = a["by_qid"][qid], b["by_qid"][qid]
        ma, mb = ra.get("memory_text") or "", rb.get("memory_text") or ""
        pa, pb = ra.get("predicted_answer") or "", rb.get("predicted_answer") or ""
        if ma == mb:
            same_memory += 1
        if pa == pb:
            same_answer += 1
        mem_chars_a.append(len(ma))
        mem_chars_b.append(len(mb))

        sa = score_row(pa, str(ra.get("reference_answer", "")), int(ra.get("category", 0)))
        sb = score_row(pb, str(rb.get("reference_answer", "")), int(rb.get("category", 0)))
        delta_f1 = sb["locomo_f1"] - sa["locomo_f1"]
        deltas.append(delta_f1)
        paired_rows.append(
            {
                "question_id": qid,
                "category": ra.get("category"),
                "question": ra.get("question"),
                "reference_answer": ra.get("reference_answer"),
                "pred_a": pa,
                "pred_b": pb,
                "same_answer": pa == pb,
                "memory_chars_a": len(ma),
                "memory_chars_b": len(mb),
                "same_memory_text": ma == mb,
                "locomo_f1_a": round(sa["locomo_f1"], 4),
                "locomo_f1_b": round(sb["locomo_f1"], 4),
                "delta_locomo_f1_b_minus_a": round(delta_f1, 4),
            }
        )

    n = len(common) or 1
    prompt_version = None
    if prompt_path and Path(prompt_path).is_file():
        prompt_version, _ = load_prompt_template(prompt_path)
    else:
        prompt_version = a["meta"].get("prompt_version") or b["meta"].get("prompt_version")
    return {
        "run_a": a["run_id"],
        "run_b": b["run_id"],
        "memory_a": a["metrics"].get("memory_type") or a["meta"].get("memory_type"),
        "memory_b": b["metrics"].get("memory_type") or b["meta"].get("memory_type"),
        "n_common": len(common),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "prompt_version": prompt_version,
        "fraction_same_memory_text": round(same_memory / n, 4) if common else None,
        "fraction_same_answer": round(same_answer / n, 4) if common else None,
        "mean_memory_chars_a": round(sum(mem_chars_a) / n, 1) if common else None,
        "mean_memory_chars_b": round(sum(mem_chars_b) / n, 1) if common else None,
        "mean_delta_locomo_f1_b_minus_a": round(sum(deltas) / n, 4) if common else None,
        "delta_locomo_f1_ci95": mean_ci95(deltas) if common else None,
        "wilcoxon_locomo_f1": wilcoxon_signed_rank(deltas) if common else None,
        "sanity": {
            "conditions_look_distinct": (same_memory / n < 0.05) if common else None,
            "note": (
                "High same_memory_text implies a builder bug for "
                "raw_chunks vs session_summaries."
            ),
        },
        "paired_rows": paired_rows,
    }


def write_paired_csv(path: Path, paired: dict) -> None:
    rows = paired.get("paired_rows") or []
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_compare_plots(packs: list[dict], paired: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    written: list[Path] = []

    # LoCoMo F1 by paper category (colors = runs / conditions).
    cats = list((packs[0]["metrics"].get("by_category") or {}).keys())
    if cats:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(cats))
        width = 0.8 / max(len(packs), 1)
        for i, pack in enumerate(packs):
            by = pack["metrics"].get("by_category") or {}
            vals = [
                (by.get(c) or {}).get("locomo_f1")
                if (by.get(c) or {}).get("n")
                else 0
                for c in cats
            ]
            vals = [v if v is not None else 0 for v in vals]
            ax.bar(x + i * width, vals, width, label=pack["run_id"])
        ax.set_xticks(x + width * (len(packs) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("LoCoMo F1")
        ax.set_title("LoCoMo F1 by category")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "locomo_f1_by_category.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # Memory size + answer-change sanity
    if paired.get("n_common"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        ma = paired.get("mean_memory_chars_a") or 0
        mb = paired.get("mean_memory_chars_b") or 0
        axes[0].bar(
            [paired.get("memory_a") or "A", paired.get("memory_b") or "B"],
            [ma, mb],
            color=["#4C78A8", "#F58518"],
        )
        axes[0].set_ylabel("Mean memory chars")
        axes[0].set_title("Memory footprint (paired Qs)")

        frac_same_m = paired.get("fraction_same_memory_text") or 0
        frac_same_a = paired.get("fraction_same_answer") or 0
        axes[1].bar(["same\nmemory", "same\nanswer"], [frac_same_m, frac_same_a], color="#54A24B")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Condition distinctness (lower ≈ better)")
        axes[1].axhline(0.05, color="red", ls="--", lw=1, label="5% alert")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "condition_distinctness.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

        # Per-question delta histogram
        deltas = [r["delta_locomo_f1_b_minus_a"] for r in paired.get("paired_rows") or []]
        if deltas:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(deltas, bins=min(10, max(3, len(deltas))), color="#72B7B2", edgecolor="white")
            ax.axvline(0, color="black", lw=1)
            ax.set_xlabel("Δ LoCoMo F1 (B − A)")
            ax.set_ylabel("Count")
            ax.set_title("Per-question score change")
            fig.tight_layout()
            p = out_dir / "delta_f1_hist.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            written.append(p)

        extra = write_compare_prediction_plots(
            paired.get("paired_rows") or [],
            label_a=str(paired.get("run_a") or "A"),
            label_b=str(paired.get("run_b") or "B"),
            out_dir=out_dir,
        )
        written.extend(extra)

    return written


def condition_diff_lines(packs: list[dict]) -> list[str]:
    """Document what differed between memory conditions (builders + configs)."""
    mems = []
    for pack in packs:
        m = pack["metrics"].get("memory_type") or pack["meta"].get("memory_type")
        mems.append(str(m) if m else "")

    prompt_paths = sorted(
        {
            str(pack["meta"].get("prompt_path") or "<missing>")
            for pack in packs
        }
    )
    reader_models = sorted(
        {
            str(pack["meta"].get("reader_model") or "<missing>")
            for pack in packs
        }
    )
    prompt_display = ", ".join(f"`{value}`" for value in prompt_paths)
    reader_display = ", ".join(f"`{value}`" for value in reader_models)

    lines = [
        "## What differs between conditions",
        "",
        "Sandwich view: **same** LoCoMo questions, answer prompt, answer model, and metrics; "
        "only the **MemoryBuilder** (middle layer) changes what string is pasted into the prompt.",
        "",
        "### Frozen (identical across raw_chunks / session_summaries in this suite)",
        "",
        "| Piece | File / setting |",
        "|-------|----------------|",
        "| Dataset | `data/raw/locomo10.json` (via configs) |",
        f"| Answer prompt | {prompt_display} (`pipeline.prompt_path`) |",
        f"| Answer LLM | OpenAI Chat Completions; {reader_display} |",
        "| Reader impl | `src/locomo_eval/readers.py` (`OpenAIReader`) |",
        "| Metrics | `src/locomo_eval/metrics.py` + `src/metrics/locomo_qa.py` |",
        "| Run (one memory YAML) | `run_locomo_pipeline_with_memory_config` in `src/locomo_eval/run.py` |",
        "",
        "### Variable middle only",
        "",
        "| | **raw_chunks** | **session_summaries** |",
        "|--|--------------------|---------------------------|",
        "| **memory_type** | `raw_chunks` | `session_summaries` |",
        "| **Config** | `configs/raw_chunks.yaml` | `configs/session_summaries.yaml` |",
        "| **Builder class** | `RawConversationMemoryBuilder` | `SessionSummaryMemoryBuilder` |",
        "| **Code** | `src/locomo_eval/memory.py` | same file |",
        "| **Source fields** | `conversation.session_*` turns + dates | `session_summary.session_*_summary` from LoCoMo release |",
        "| **What the LLM sees** | Speaker turns (`Speaker: text`), session DATE headers, dia_ids; optional tail trunc @ `memory_max_chars` | `[Session k]` + released prose summaries only |",
        "| **Structure** | High fidelity, long, little abstraction | Compressed narrative; may drop fine detail / exact dates |",
        "| **Typical length** | Much longer (this run ~74k chars mean) | Shorter (~21k chars mean) |",
        "| **Research question** | Does raw context already work? | Do dataset-style session memories help vs raw? |",
        "",
        "### How config selects the builder",
        "",
        "1. YAML sets `pipeline.memory` (`raw_chunks` or `session_summaries`).",
        "2. `run_locomo_pipeline_with_memory_config` in `run.py` calls `get_memory_builder(...)`.",
        "3. `builder.build(conversation, question)` -> `Memory.text`.",
        "4. That text fills `{memory}` in the configured answer prompt; gold **answer is never** sent to the LLM.",
        "",
        "### Audit pointers for this comparison",
        "",
    ]
    for pack in packs:
        mem = pack["metrics"].get("memory_type") or pack["meta"].get("memory_type")
        lines.append(
            f"- `{pack['run_id']}` (`{mem}`): preds "
            f"`{pack['dir']}/predictions.csv` — compare `memory_preview` / `memory_text`"
        )
    lines.append("- Paired per-question deltas: `paired_questions.csv` in this compare folder")
    lines.append(
        "- Human data model writeup: `docs/agent/HUMANS.md` "
        "(section *Mental model of the data & task*)"
    )
    lines.append("")

    # Optional: flag if non-standard memory labels
    known = {"raw_chunks", "session_summaries"}
    for m in mems:
        if m and m not in known:
            lines.append(
                f"- Note: memory_type `{m}` is outside the canned raw_chunks / session_summaries table above; "
                f"see `src/locomo_eval/memory.py` registry."
            )
            lines.append("")
            break
    return lines


def write_text_report(
    path: Path,
    packs: list[dict],
    paired: dict,
    plot_paths: list[Path],
    bottom_warn: str | None,
) -> None:
    lines: list[str] = []
    lines.append("# Comparison report")
    lines.append("")
    lines.append("## Runs")
    for pack in packs:
        m = pack["metrics"]
        meta = pack["meta"]
        overall = m.get("metrics") or {}
        lines.append(f"### {pack['run_id']}")
        lines.append(f"- dir: `{pack['dir']}`")
        lines.append(f"- memory_type: `{m.get('memory_type') or meta.get('memory_type')}`")
        lines.append(f"- reader_model: `{m.get('reader_model') or meta.get('reader_model')}`")
        lines.append(f"- prompt_version: `{m.get('prompt_version') or meta.get('prompt_version')}`")
        lines.append(f"- n_questions: {m.get('number_of_questions')}")
        lines.append(
            f"- scores: locomo_f1={overall.get('locomo_f1')}, "
            f"token_f1={overall.get('token_f1')}, exact_match={overall.get('exact_match')}"
        )
        lines.append(
            f"- API/meta: new_api={meta.get('n_new_api_calls')}"
        )
        lines.append("")

    lines.extend(condition_diff_lines(packs))

    if bottom_warn:
        lines.append("## WARNING: frozen bottom mismatch")
        lines.append(bottom_warn)
        lines.append("")

    lines.append("## Paired condition sanity (memory text)")
    if paired.get("n_common") is None and "note" in paired:
        lines.append(paired["note"])
    else:
        lines.append(
            f"- paired questions: **{paired.get('n_common')}** "
            f"(only A: {paired.get('n_only_a')}, only B: {paired.get('n_only_b')})"
        )
        lines.append(
            f"- mean memory chars: A={paired.get('mean_memory_chars_a')} "
            f"({paired.get('memory_a')}) vs B={paired.get('mean_memory_chars_b')} "
            f"({paired.get('memory_b')})"
        )
        lines.append(
            f"- fraction same memory text: **{paired.get('fraction_same_memory_text')}** "
            f"(want ~0 for raw_chunks vs session_summaries)"
        )
        lines.append(
            f"- fraction same predicted answer: **{paired.get('fraction_same_answer')}**"
        )
        lines.append(
            f"- mean delta LoCoMo F1 (B-A): **{paired.get('mean_delta_locomo_f1_b_minus_a')}**"
        )
        san = paired.get("sanity") or {}
        lines.append(
            f"- conditions_look_distinct: {san.get('conditions_look_distinct')}"
        )
        lines.append(f"- note: {san.get('note')}")
    lines.append("")

    lines.append("## Category snapshot")
    for pack in packs:
        lines.append(f"### {pack['run_id']}")
        for cat, stats in (pack["metrics"].get("by_category") or {}).items():
            if not stats.get("n"):
                continue
            lines.append(
                f"- {cat} (n={stats.get('n')}): "
                f"locomo_f1={stats.get('locomo_f1')}, token_f1={stats.get('token_f1')}"
            )
        lines.append("")

    if plot_paths:
        lines.append("## Plots")
        for p in plot_paths:
            lines.append(f"- `{p}`")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Paths to experiments/<run_id> dirs",
    )
    p.add_argument("--out", default="experiments/compare", help="Output directory")
    p.add_argument(
        "--prompt",
        default=None,
        help="Override frozen answer prompt (default: infer from run_meta.json)",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    packs = [load_pack(Path(r)) for r in args.runs]
    write_overall_csv(out / "overall.csv", packs)
    write_category_csv(out / "by_category.csv", packs)

    keys = ["reader_model", "prompt_version", "temperature"]
    bottoms = [tuple(pack["meta"].get(k) for k in keys) for pack in packs]
    bottom_warn = None
    if len(set(bottoms)) > 1:
        bottom_warn = "Runs differ on frozen bottom-layer settings:\n" + "\n".join(
            str(b) for b in bottoms
        )
        (out / "WARNING_bottom_mismatch.txt").write_text(bottom_warn, encoding="utf-8")
        print("WARNING: frozen bottom mismatch")

    paired = pair_analysis(packs, resolve_prompt_path(packs, args.prompt))
    write_paired_csv(out / "paired_questions.csv", paired)
    # Drop huge paired_rows from JSON blob (CSV holds them)
    paired_slim = {k: v for k, v in paired.items() if k != "paired_rows"}
    plot_paths = make_compare_plots(packs, paired, out / "plots")
    write_text_report(out / "SUMMARY.md", packs, paired_slim, plot_paths, bottom_warn)

    payload = {
        "runs": [
            {
                "dir": pack["dir"],
                "run_id": pack["run_id"],
                "metrics": pack["metrics"],
                "meta": pack["meta"],
                "n_predictions": len(pack["predictions"]),
            }
            for pack in packs
        ],
        "paired": paired_slim,
        "plots": [str(p) for p in plot_paths],
    }
    (out / "compare.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out / 'overall.csv'}")
    print(f"Wrote {out / 'by_category.csv'}")
    print(f"Wrote {out / 'paired_questions.csv'}")
    print(f"Wrote {out / 'SUMMARY.md'}")
    print(f"Wrote {out / 'compare.json'}")
    for plot in plot_paths:
        print(f"Plot: {plot}")
    print(f"Sanity same_memory={paired_slim.get('fraction_same_memory_text')}")


if __name__ == "__main__":
    main()
