"""Year-stagnation diagnosis for graph sandwich cells.

Uses finished audit packs plus optional campaign Parquet. No LLM.
Separates technical invalidity (parse fallback, empty graphs, gold leak)
from scientific ceilings (frozen gpt-4o-mini reader, timeless triples).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.locomo_eval.experiment_pack.verify_pack import (
    SCHEMA_VERSION,
    PackReport,
    discover_run_dirs,
    verify_pack,
)

MOVE_EPS = 0.02
TOKEN_EPS_FRAC = 0.15
WRITER_GENERATION = {
    "gpt-5": "2025",
    "deepseek-chat": "2025",
    "gpt-5.6-terra": "2026",
    "deepseek-v4-flash": "2026",
}
WRITER_FAMILY = {
    "gpt-5": "openai",
    "gpt-5.6-terra": "openai",
    "deepseek-chat": "deepseek",
    "deepseek-v4-flash": "deepseek",
}


@dataclass
class YearPair:
    family: str
    thinking: str
    metric: str
    value_2025: float | None
    value_2026: float | None
    delta: float | None
    label: str


@dataclass
class GraphYearDiagnosis:
    schema_version: str
    headline: str
    technical: list[str]
    scientific: list[str]
    incomplete: list[str]
    pairs: list[YearPair] = field(default_factory=list)
    pack_verdicts: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "headline": self.headline,
            "technical": self.technical,
            "scientific": self.scientific,
            "incomplete": self.incomplete,
            "pairs": [asdict(p) for p in self.pairs],
            "pack_verdicts": self.pack_verdicts,
        }


def writer_generation(model: str | None) -> str | None:
    return WRITER_GENERATION.get(str(model or ""))


def writer_family(model: str | None) -> str | None:
    return WRITER_FAMILY.get(str(model or ""))


def classify_delta(delta: float | None, eps: float = MOVE_EPS) -> str:
    if delta is None:
        return "missing"
    if delta > eps:
        return "improving"
    if delta < -eps:
        return "declining"
    return "not_moving"


def load_graph_cells_from_parquet(experiment_dir: str | Path) -> list[dict[str, Any]]:
    """Score/token cells from aggregate Parquet (full LoCoMo), if present."""
    root = Path(experiment_dir)
    runs_path = root / "aggregate" / "runs.parquet"
    examples_path = root / "aggregate" / "examples.parquet"
    if not runs_path.is_file():
        return []
    import pandas as pd

    runs = pd.read_parquet(runs_path)
    rows: list[dict[str, Any]] = []
    examples = None
    if examples_path.is_file():
        examples = pd.read_parquet(examples_path)
    for rec in runs.to_dict(orient="records"):
        if str(rec.get("memory_method") or "") != "graph":
            continue
        writer = rec.get("writer_model")
        thinking = str(rec.get("thinking") or "off")
        cell: dict[str, Any] = {
            "run_id": rec.get("run_id"),
            "writer_model": writer,
            "generation": writer_generation(writer),
            "family": writer_family(writer),
            "thinking": thinking,
            "locomo_f1": _as_float(rec.get("locomo_f1") or rec.get("primary_score")),
            "agent_input_tokens_mean": None,
            "judge_score": None,
        }
        if examples is not None and len(examples):
            sub = examples[
                (examples["memory_method"] == "graph")
                & (examples["writer_model"] == writer)
                & (examples["thinking"].astype(str) == thinking)
            ]
            if len(sub):
                cell["agent_input_tokens_mean"] = _as_float(
                    sub["agent_input_tokens"].mean()
                )
                judged = sub[sub["question_category"] != 5]
                if "judge_score" in judged.columns and judged["judge_score"].notna().any():
                    cell["judge_score"] = _as_float(judged["judge_score"].mean())
        rows.append(cell)
    return rows


def diagnose_graph_year_stagnation(
    experiment_dirs: list[str | Path],
    *,
    pack_reports: list[PackReport] | None = None,
    cells: list[dict[str, Any]] | None = None,
    eps: float = MOVE_EPS,
) -> GraphYearDiagnosis:
    """Explain why graph scores did not move 2025 → 2026."""
    loaded: list[dict[str, Any]] = list(cells or [])
    if not loaded:
        for directory in experiment_dirs:
            loaded.extend(load_graph_cells_from_parquet(directory))
    cells = loaded
    reports = list(pack_reports or [])
    seen_ids = {r.run_id for r in reports}
    if not pack_reports:
        for directory in experiment_dirs:
            for run_dir in discover_run_dirs(directory):
                report = verify_pack(run_dir)
                reports.append(report)
                seen_ids.add(report.run_id)
    for directory in experiment_dirs:
        extra = Path(directory)
        if not extra.is_dir():
            continue
        parent = extra.parent if (extra / "aggregate").is_dir() else extra
        prefix = extra.name + "-smoke-" if (extra / "aggregate").is_dir() else ""
        if not prefix or not parent.is_dir():
            continue
        for child in sorted(parent.iterdir()):
            if child.name.startswith(prefix) and (child / "run_meta.json").is_file():
                if child.name in seen_ids:
                    continue
                report = verify_pack(child)
                reports.append(report)
                seen_ids.add(report.run_id)

    graph_reports = [
        r for r in reports if r.memory_type == "graph"
    ]
    technical: list[str] = []
    scientific: list[str] = []
    incomplete: list[str] = []
    verdicts = []
    for report in sorted(graph_reports, key=lambda r: r.run_id):
        verdicts.append(
            {
                "run_id": report.run_id,
                "verdict": report.verdict,
                "pack_kind": report.pack_kind,
            }
        )
        if report.verdict == "invalid":
            fails = [
                c.id
                for c in report.checks
                if c.status == "fail" and c.severity == "error"
            ]
            technical.append(
                f"{report.run_id} is invalid ({', '.join(fails) or 'see checks'})."
            )
        if report.pack_kind in {"thin_catalog", "incomplete"}:
            incomplete.append(
                f"{report.run_id} has no memory/graph dump "
                "(aggregate catalog only; run collect-full to inspect triples)."
            )
        for check in report.checks:
            if (
                check.id == "graph.parse_rate"
                and check.status == "fail"
                and check.severity == "error"
            ):
                technical.append(f"{report.run_id}: {check.detail}")
            if check.id == "prompt.graph_timeless" and check.status == "pass":
                note = (
                    "graph_v1 strips timestamps on purpose; temporal "
                    "LoCoMo items cannot improve just because the writer year moved."
                )
                if note not in scientific:
                    scientific.append(note)
            if check.id == "config.sandwich_reader" and check.status == "pass":
                note = (
                    "Sandwich reader is frozen gpt-4o-mini. A better 2026 writer "
                    "only helps if {memory} changes in a way that mini can use."
                )
                if note not in scientific:
                    scientific.append(note)

    pairs = _year_pairs(cells, eps=eps)
    score_pairs = [p for p in pairs if p.metric in {"locomo_f1", "judge_score"}]
    improving = [p for p in score_pairs if p.label == "improving"]
    token_pairs = [p for p in pairs if p.metric == "agent_input_tokens_mean"]
    similar_input = [
        p
        for p in token_pairs
        if p.value_2025
        and p.value_2026
        and abs(p.delta or 0) / max(p.value_2025, 1.0) <= TOKEN_EPS_FRAC
    ]
    if score_pairs and not improving:
        labels = sorted({p.label for p in score_pairs})
        scientific.append(
            "Matched 2025->2026 graph F1/J cells did not improve "
            f"(eps={eps}; labels={labels}). Year did not buy graph accuracy "
            "under the frozen reader."
        )
    gap = _session_summary_gap(experiment_dirs)
    if gap:
        scientific.append(gap)
    smoke_parse = [
        r
        for r in graph_reports
        if r.verdict == "invalid"
        and "smoke" in r.run_id
        and any(c.id == "graph.parse_rate" and c.status == "fail" for c in r.checks)
    ]
    healthy_smokes = [
        r
        for r in graph_reports
        if "smoke" in r.run_id
        and r.verdict in {"valid", "valid_with_warnings"}
        and any(c.id == "graph.parse_rate" and c.status == "pass" for c in r.checks)
    ]
    if smoke_parse and (score_pairs or healthy_smokes):
        scientific.append(
            "Parse fallback_mock was measured on 19-session smokes, not the "
            "full LoCoMo cells. Invalid smokes do not by themselves prove the "
            "1986-question year table is mock-injected; collect-full those cells."
        )
    if similar_input:
        scientific.append(
            "Reader input size on graph stayed within "
            f"{TOKEN_EPS_FRAC:.0%} year-on-year, so gpt-4o-mini saw a similar "
            "triple dump, not a new memory method."
        )
    thinking_hurt = _thinking_hurt(cells, eps=eps)
    if thinking_hurt:
        technical.append(
            "Graph thinking-on scored at or below thinking-off. JSON extract "
            "plus fallback_mock on truncated reasoning is a technical risk; "
            "it is not evidence that a larger window helps graphs."
        )
        scientific.append(
            "Even when parse succeeds, timeless triples + a 64-token mini "
            "reader are a weak fit for LoCoMo temporal/multi-hop items; "
            "session summaries keep dates and narrative."
        )

    if not cells:
        incomplete.append(
            "No graph rows in aggregate/runs.parquet; year scores "
            "cannot be confirmed from these paths."
        )
    if not technical and not scientific and not incomplete:
        scientific.append(
            "No pack errors and no year table; inspect dumps before claiming a bug."
        )

    headline = _headline(
        technical,
        scientific,
        incomplete,
        improving,
        n_score_pairs=len(score_pairs),
    )
    # Dedup while preserving order
    technical = list(dict.fromkeys(technical))
    scientific = list(dict.fromkeys(scientific))
    incomplete = list(dict.fromkeys(incomplete))
    return GraphYearDiagnosis(
        schema_version=SCHEMA_VERSION,
        headline=headline,
        technical=technical,
        scientific=scientific,
        incomplete=incomplete,
        pairs=pairs,
        pack_verdicts=verdicts,
    )


def _year_pairs(cells: list[dict[str, Any]], *, eps: float) -> list[YearPair]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for cell in cells:
        family = cell.get("family")
        thinking = str(cell.get("thinking") or "off")
        gen = cell.get("generation")
        if not family or gen not in {"2025", "2026"}:
            continue
        by_key.setdefault((family, thinking), {})[gen] = cell
    pairs: list[YearPair] = []
    for family, thinking in sorted(by_key):
        left = by_key[(family, thinking)].get("2025")
        right = by_key[(family, thinking)].get("2026")
        for metric in ("locomo_f1", "judge_score", "agent_input_tokens_mean"):
            a = (left or {}).get(metric)
            b = (right or {}).get(metric)
            delta = None
            if a is not None and b is not None:
                delta = float(b) - float(a)
            label = classify_delta(delta, eps=eps)
            if metric == "agent_input_tokens_mean" and a and b:
                frac = abs(delta or 0) / max(float(a), 1.0)
                if frac <= TOKEN_EPS_FRAC:
                    label = "not_moving"
            pairs.append(
                YearPair(
                    family=family,
                    thinking=thinking,
                    metric=metric,
                    value_2025=_as_float(a),
                    value_2026=_as_float(b),
                    delta=None if delta is None else round(float(delta), 4),
                    label=label,
                )
            )
    return pairs


def _thinking_hurt(cells: list[dict[str, Any]], *, eps: float) -> bool:
    by: dict[tuple[str, str], dict[str, float]] = {}
    for cell in cells:
        f1 = cell.get("locomo_f1")
        if f1 is None:
            continue
        key = (str(cell.get("generation")), str(cell.get("family")))
        by.setdefault(key, {})[str(cell.get("thinking") or "off")] = float(f1)
    for scores in by.values():
        off = scores.get("off")
        on = scores.get("on")
        if off is not None and on is not None and (on - off) <= eps:
            return True
    return False


def _headline(
    technical: list[str],
    scientific: list[str],
    incomplete: list[str],
    improving: list[YearPair],
    n_score_pairs: int = 0,
) -> str:
    if technical and n_score_pairs and not improving:
        return (
            "Technical issues exist, and year-matched graph scores still did "
            "not improve; do not attribute the flat or down line to a better "
            "2026 writer."
        )
    if technical:
        return "Pack logs show technical defects; fix those before a year-move claim."
    if n_score_pairs and not improving and scientific:
        return (
            "Graph year-stagnation looks scientific: frozen mini reader + "
            "timeless triples; matched F1/J did not improve."
        )
    if improving:
        return "Some matched graph cells improved; inspect pairs before calling it flat."
    if incomplete:
        return "Audit is incomplete (thin catalogs / missing parquet); cannot close the question."
    if scientific:
        return (
            "Pack dumps are consistent with a scientific ceiling "
            "(frozen reader + timeless graph), but year scores were not in this invocation."
        )
    return "Not enough graph evidence to decide bug vs science."


def _session_summary_gap(experiment_dirs: list[str | Path]) -> str | None:
    """Same sandwich reader and writer year: narrative summaries vs timeless graph."""
    graph: list[float] = []
    summaries: list[float] = []
    for directory in experiment_dirs:
        path = Path(directory) / "aggregate" / "runs.parquet"
        if not path.is_file():
            continue
        import pandas as pd

        frame = pd.read_parquet(path)
        if "locomo_f1" not in frame.columns:
            continue
        graph.extend(
            float(v)
            for v in frame.loc[
                frame["memory_method"] == "graph", "locomo_f1"
            ].dropna()
        )
        summaries.extend(
            float(v)
            for v in frame.loc[
                frame["memory_method"] == "session_summaries", "locomo_f1"
            ].dropna()
        )
    if not graph or not summaries:
        return None
    gmean = sum(graph) / len(graph)
    smean = sum(summaries) / len(summaries)
    gap = smean - gmean
    if gap < 0.05:
        return None
    return (
        f"session_summaries mean LoCoMo F1 {smean:.3f} vs graph "
        f"{gmean:.3f} (gap {gap:.3f}) under the same frozen reader and writer "
        "year, so the bottleneck is graph encoding, not summary generation."
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_graph_year_markdown(diag: GraphYearDiagnosis) -> str:
    lines = [
        "## graph year stagnation",
        "",
        diag.headline,
        "",
        "### Technical (logs / bugs)",
    ]
    if diag.technical:
        lines.extend(f"- {item}" for item in diag.technical)
    else:
        lines.append("- none flagged")
    lines += ["", "### Scientific (design, given valid packs)"]
    if diag.scientific:
        lines.extend(f"- {item}" for item in diag.scientific)
    else:
        lines.append("- none flagged")
    lines += ["", "### Incomplete audit"]
    if diag.incomplete:
        lines.extend(f"- {item}" for item in diag.incomplete)
    else:
        lines.append("- none")
    if diag.pairs:
        lines += [
            "",
            "| family | thinking | metric | 2025 | 2026 | delta | label |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for pair in diag.pairs:
            lines.append(
                "| {family} | {thinking} | {metric} | {v5} | {v6} | {delta} | {label} |".format(
                    family=pair.family,
                    thinking=pair.thinking,
                    metric=pair.metric,
                    v5=_fmt(pair.value_2025),
                    v6=_fmt(pair.value_2026),
                    delta=_fmt(pair.delta),
                    label=pair.label,
                )
            )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.3f}"
