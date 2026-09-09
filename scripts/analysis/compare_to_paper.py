"""Paper Table 2 J vs local LLM-as-a-Judge packs (offline, no API).

Reads finished ``experiments/<run>/autorater/`` and ``autorater_seeds/``
reports. When several live judge packs exist for one method, the plot uses
the **best** J (max), not the paper's mean-of-10 protocol.

    python -m scripts.analysis.compare_to_paper \\
      --runs experiments/full_context_qa experiments/rag_k2_256_qa experiments/mem0_qa \\
      --out experiments/compare_paper_vs_local

Local bars are architecture-clone reproductions, not Mem0 Platform / ChatGPT
Memory product numbers. Mock autorater packs are ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.plots import write_paper_vs_local_bars
from src.locomo_eval.mem0_baselines import (
    FIGURE4_METHODS,
    LOCAL_PAPER_METHODS,
    PAPER_CITE,
    PAPER_URL,
    TABLE1_BY_CATEGORY,
    paper_method_from_label,
    rag_paper_method,
    table2_by_method,
)
from src.locomo_eval.report import write_json
from src.metrics.locomo_qa import CATEGORY_NAMES

CATEGORY_PLOT_ORDER = ("single_hop", "multi_hop", "open_domain", "temporal")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)
    if path.is_dir():
        return path.resolve()
    alt = ROOT / "experiments" / path
    if alt.is_dir():
        return alt.resolve()
    raise FileNotFoundError(f"No run directory at {raw} or experiments/{raw}")


def parse_run_spec(spec: str) -> tuple[Path, str | None]:
    """``path`` or ``path:PaperMethod`` override."""
    if ":" in spec and not Path(spec).is_dir():
        run_part, method = spec.rsplit(":", 1)
        method = method.strip()
        if method:
            return resolve_run_dir(run_part), method
    return resolve_run_dir(spec), None


def iter_autorater_dirs(run_dir: Path) -> list[Path]:
    found: list[Path] = []
    top = run_dir / "autorater"
    if (top / "autorater_metrics.json").is_file():
        found.append(top)
    seeds = run_dir / "autorater_seeds"
    if seeds.is_dir():
        for child in sorted(seeds.iterdir()):
            if child.is_dir() and (child / "autorater_metrics.json").is_file():
                found.append(child)
    return found


def _is_live_judge(metrics: dict[str, Any], meta: dict[str, Any]) -> bool:
    provider = (
        metrics.get("autorater_provider")
        or meta.get("autorater_provider")
        or ""
    )
    kind = metrics.get("score_kind") or ""
    if provider == "mock" or kind == "mock_sanity_not_llm_judge":
        return False
    j = (metrics.get("metrics") or {}).get("llm_judge_pct")
    return j is not None


def _chunk_size_from_index(run_dir: Path, meta: dict[str, Any]) -> int | None:
    index_id = meta.get("rag_index_run_id")
    if not index_id:
        return None
    index_meta = run_dir.parent / str(index_id) / "rag_index" / "run_meta.json"
    if not index_meta.is_file():
        alt = ROOT / "experiments" / str(index_id) / "rag_index" / "run_meta.json"
        index_meta = alt if alt.is_file() else index_meta
    if not index_meta.is_file():
        return None
    raw = _load_json(index_meta).get("chunk_size")
    return int(raw) if raw is not None else None


def infer_paper_method(
    run_dir: Path,
    *,
    metrics: dict[str, Any] | None = None,
    override: str | None = None,
) -> str | None:
    if override:
        mapped = paper_method_from_label(override) or override
        return mapped
    if metrics:
        labeled = paper_method_from_label(metrics.get("memory_type"))
        if labeled:
            return labeled
    meta_path = run_dir / "run_meta.json"
    meta = _load_json(meta_path) if meta_path.is_file() else {}
    labeled = paper_method_from_label(meta.get("memory_type"))
    if labeled:
        return labeled
    memory_type = str(meta.get("memory_type") or "")
    if memory_type == "rag" or str(run_dir.name).startswith("rag"):
        return rag_paper_method(meta.get("rag_k"), _chunk_size_from_index(run_dir, meta))
    return None


def best_local_score(
    run_dir: Path,
    *,
    method_override: str | None = None,
) -> dict[str, Any] | None:
    """Highest live J among autorater + seed packs in one QA run."""
    packs = iter_autorater_dirs(run_dir)
    best: dict[str, Any] | None = None
    considered = 0
    for pack in packs:
        metrics = _load_json(pack / "autorater_metrics.json")
        meta_path = pack / "run_meta.json"
        meta = _load_json(meta_path) if meta_path.is_file() else {}
        if not _is_live_judge(metrics, meta):
            continue
        considered += 1
        j = float((metrics.get("metrics") or {})["llm_judge_pct"])
        if best is None or j > float(best["local_j"]):
            method = infer_paper_method(
                run_dir, metrics=metrics, override=method_override
            )
            m = metrics.get("metrics") or {}
            by_cat = {}
            for name, stats in (metrics.get("by_category") or {}).items():
                by_cat[str(name)] = {
                    "j": stats.get("llm_judge_pct"),
                    "f1": stats.get("mem0_f1_pct"),
                    "bleu1": stats.get("mem0_bleu1_pct"),
                    "n": stats.get("n"),
                }
            best = {
                "method": method,
                "local_j": j,
                "local_f1": m.get("mem0_f1_pct"),
                "local_bleu1": m.get("mem0_bleu1_pct"),
                "n_judged": metrics.get("n_judged"),
                "n_skipped": metrics.get("n_skipped"),
                "n_predictions": metrics.get("number_of_questions"),
                "selected_pack": str(pack),
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "autorater_model": metrics.get("autorater_model")
                or meta.get("autorater_model"),
                "by_category": by_cat,
            }
    if best is None:
        return None
    best["n_judge_packs"] = considered
    return best


def merge_best_by_method(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """If two runs share a paper method, keep the higher J."""
    out: dict[str, dict[str, Any]] = {}
    for score in scores:
        method = score.get("method")
        if not method:
            continue
        prev = out.get(method)
        if prev is None:
            out[method] = dict(score)
            continue
        combined_packs = int(prev.get("n_judge_packs") or 0) + int(
            score.get("n_judge_packs") or 0
        )
        winner = score if float(score["local_j"]) > float(prev["local_j"]) else prev
        merged = dict(winner)
        merged["n_judge_packs"] = combined_packs
        out[method] = merged
    return out


def comparison_rows(
    methods: list[str],
    local_by_method: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    lit = table2_by_method()
    rows: list[dict[str, Any]] = []
    for method in methods:
        paper = lit.get(method) or {}
        local = local_by_method.get(method) or {}
        paper_j = paper.get("j")
        if paper_j is not None:
            paper_j = round(float(paper_j), 2)
        paper_std = paper.get("j_std")
        if paper_std is not None:
            paper_std = round(float(paper_std), 2)
        local_j = local.get("local_j")
        if local_j is not None:
            local_j = round(float(local_j), 2)
        delta = None
        if paper_j is not None and local_j is not None:
            delta = round(float(local_j) - float(paper_j), 2)
        rows.append(
            {
                "method": method,
                "paper_j": paper_j,
                "paper_j_std": paper_std,
                "local_j": local_j,
                "delta_j": delta,
                "local_f1": local.get("local_f1"),
                "local_bleu1": local.get("local_bleu1"),
                "n_judged": local.get("n_judged"),
                "n_judge_packs": local.get("n_judge_packs"),
                "selected_pack": local.get("selected_pack"),
                "run_id": local.get("run_id"),
                "autorater_model": local.get("autorater_model"),
                "cite": PAPER_CITE if paper_j is not None else None,
            }
        )
    return rows


def category_rows(
    local_by_method: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, local in local_by_method.items():
        paper_cats = TABLE1_BY_CATEGORY.get(method) or {}
        if not paper_cats:
            continue
        local_cats = local.get("by_category") or {}
        names = [n for n in CATEGORY_PLOT_ORDER if n in paper_cats or n in local_cats]
        for cat in names:
            paper = paper_cats.get(cat) or {}
            ours = local_cats.get(cat) or {}
            paper_j = paper.get("j")
            local_j = ours.get("j")
            delta = None
            if paper_j is not None and local_j is not None:
                delta = round(float(local_j) - float(paper_j), 2)
            rows.append(
                {
                    "method": method,
                    "category": cat,
                    "category_id": next(
                        (k for k, v in CATEGORY_NAMES.items() if v == cat), None
                    ),
                    "paper_j": paper_j,
                    "paper_j_std": paper.get("j_std"),
                    "local_j": local_j,
                    "delta_j": delta,
                    "paper_f1": paper.get("f1"),
                    "local_f1": ours.get("f1"),
                    "paper_bleu1": paper.get("bleu1"),
                    "local_bleu1": ours.get("bleu1"),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _reset_output(out_dir: Path) -> None:
    for dirname in ("plots", "tables"):
        path = out_dir / dirname
        if path.is_dir():
            shutil.rmtree(path)
    for filename in ("compare.json", "SUMMARY.md"):
        path = out_dir / filename
        if path.is_file():
            path.unlink()


def write_summary_md(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    cat_rows: list[dict[str, Any]],
    missing: list[str],
    plot_paths: list[Path],
) -> None:
    lines = [
        "# LLM-as-a-Judge vs Mem0 paper",
        "",
        f"Judge protocol: Mem0 LLM-as-a-Judge ({PAPER_CITE}).",
        f"Paper: {PAPER_URL}",
        "",
        "Paper bars are **literature pins** (Table 2 overall J). Local bars",
        "are this repo's architecture clones under the same judge prompt.",
        "They are **not** Mem0 Platform, Neo4j, or ChatGPT Memory product runs.",
        "",
        "When a method has several live autorater packs / seeds, this report",
        "plots the **best** J (max), not the paper's mean ± std of 10 judge",
        "runs.",
        "",
        "## Overall J",
        "",
    ]
    for row in rows:
        local = row["local_j"]
        local_s = "—" if local is None else str(local)
        delta = row["delta_j"]
        delta_s = "" if delta is None else f" (Δ {delta:+.2f})"
        lines.append(
            f"- {row['method']}: paper={row['paper_j']} ± {row['paper_j_std']} "
            f"| local={local_s}{delta_s}"
        )
    if missing:
        lines.extend(
            [
                "",
                "## Local packs without a live judge",
                "",
                "These QA folders were passed in but had no non-mock",
                "`autorater_metrics.json` (run `python -m scripts.analysis.run_benchmark --run …` first):",
                "",
            ]
        )
        for name in missing:
            lines.append(f"- `{name}`")
    if cat_rows:
        lines.extend(["", "## Category J (Table 1 methods with a local pack)", ""])
        for row in cat_rows:
            if row.get("paper_j") is None and row.get("local_j") is None:
                continue
            local_s = "—" if row["local_j"] is None else str(row["local_j"])
            paper_s = "—" if row["paper_j"] is None else str(row["paper_j"])
            lines.append(
                f"- {row['method']} / {row['category']}: paper={paper_s} | local={local_s}"
            )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for p in plot_paths:
            lines.append(f"- `{p}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_to_paper(
    run_specs: list[str],
    *,
    out_dir: str | Path,
    methods: list[str] | None = None,
    include_external: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _reset_output(out)

    scores: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in run_specs:
        run_dir, override = parse_run_spec(spec)
        score = best_local_score(run_dir, method_override=override)
        if score is None:
            missing.append(str(run_dir))
            continue
        if score.get("method") is None:
            missing.append(f"{run_dir} (could not map memory_type to a paper method)")
            continue
        scores.append(score)

    local_by_method = merge_best_by_method(scores)
    method_list = list(methods or LOCAL_PAPER_METHODS)
    if include_external:
        method_list = list(FIGURE4_METHODS)
    for name in local_by_method:
        if name not in method_list:
            method_list.append(name)

    rows = comparison_rows(method_list, local_by_method)
    cat_rows = category_rows(local_by_method)

    tables = out / "tables"
    overall_path = tables / "overall.csv"
    _write_csv(
        overall_path,
        rows,
        [
            "method",
            "paper_j",
            "paper_j_std",
            "local_j",
            "delta_j",
            "local_f1",
            "local_bleu1",
            "n_judged",
            "n_judge_packs",
            "selected_pack",
            "run_id",
            "autorater_model",
            "cite",
        ],
    )
    cat_path = tables / "by_category.csv"
    _write_csv(
        cat_path,
        cat_rows,
        [
            "method",
            "category",
            "category_id",
            "paper_j",
            "paper_j_std",
            "local_j",
            "delta_j",
            "paper_f1",
            "local_f1",
            "paper_bleu1",
            "local_bleu1",
        ],
    )

    plots_dir = out / "plots"
    plot_paths: list[Path] = []
    overall_plot = write_paper_vs_local_bars(
        [r["method"] for r in rows],
        [r["paper_j"] for r in rows],
        [r["local_j"] for r in rows],
        path=plots_dir / "j_paper_vs_local.png",
        ylabel="LLM-as-a-Judge J (%)",
        title="LLM-as-a-Judge vs Mem0 paper (best local seed)",
    )
    if overall_plot is not None:
        plot_paths.append(overall_plot)

    cat_with_j = [
        r
        for r in cat_rows
        if r.get("paper_j") is not None or r.get("local_j") is not None
    ]
    if cat_with_j:
        cat_plot = write_paper_vs_local_bars(
            [f"{r['method']} / {r['category']}" for r in cat_with_j],
            [r["paper_j"] for r in cat_with_j],
            [r["local_j"] for r in cat_with_j],
            path=plots_dir / "j_by_category_paper_vs_local.png",
            ylabel="LLM-as-a-Judge J (%)",
            title="Category J vs Mem0 paper Table 1 (best local seed)",
        )
        if cat_plot is not None:
            plot_paths.append(cat_plot)

    report = {
        "cite": PAPER_CITE,
        "paper_url": PAPER_URL,
        "selection": "best_live_judge_j",
        "methods": method_list,
        "rows": rows,
        "by_category": cat_rows,
        "local_packs": scores,
        "missing_local_judge": missing,
        "note": (
            "Local J is the max over live autorater packs for each method. "
            "Do not treat OSS clones as paper Table 1–2 Platform numbers."
        ),
    }
    write_json(out / "compare.json", report)
    write_summary_md(
        out / "SUMMARY.md",
        rows=rows,
        cat_rows=cat_rows,
        missing=missing,
        plot_paths=plot_paths,
    )
    return {
        "out_dir": str(out),
        "rows": rows,
        "by_category": cat_rows,
        "missing_local_judge": missing,
        "plots": [str(p) for p in plot_paths],
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Grouped paper vs local LLM-as-a-Judge J (best seed, no API). "
            "Pass finished QA run folders that already have autorater packs."
        )
    )
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="experiments/<run_id> (optional :PaperMethod override)",
    )
    p.add_argument(
        "--out",
        default="experiments/compare_paper_vs_local",
        help="Report directory",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Paper method names on the x-axis (default: locally clonable Table 2 set)",
    )
    p.add_argument(
        "--include-external",
        action="store_true",
        help="Also plot A-Mem / LangMem / Zep paper-only bars (Figure 4 set)",
    )
    args = p.parse_args(argv)
    result = compare_to_paper(
        args.runs,
        out_dir=args.out,
        methods=args.methods,
        include_external=args.include_external,
    )
    print("Wrote", result["out_dir"])
    for row in result["rows"]:
        print(
            f"  {row['method']}: paper={row['paper_j']} local={row['local_j']} "
            f"delta={row['delta_j']}"
        )
    for plot in result["plots"]:
        print("Plot:", plot)
    for missing in result["missing_local_judge"]:
        print("No live judge:", missing)


if __name__ == "__main__":
    main()
