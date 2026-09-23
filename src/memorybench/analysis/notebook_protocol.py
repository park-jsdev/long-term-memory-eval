"""Notebook pre-test / post-test helpers (YAML-driven, no plot code in notebooks)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.memorybench.analysis.cost import CostReport, render_cost
from src.memorybench.analysis.load_campaign import CampaignConfig
from src.memorybench.analysis.report import (
    ReportResult,
    ROOT,
    dataframe_html,
    dataframe_markdown,
    notebook_show,
)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def notebook_pretest(
    cfg: CampaignConfig,
    experiment_id: str | None = None,
    *,
    root: Path | None = None,
) -> CostReport | None:
    """Show freeze, hypotheses, expected cells, and priced volume estimate."""
    root = root or ROOT
    _display_markdown(_pretest_markdown(cfg, experiment_id))
    out_dir = _out_dir(cfg, experiment_id, root)
    cost = render_cost(cfg, out_dir, root=root, experiment_id=experiment_id)
    if cost is None:
        _display_markdown("_No `cost:` block in this campaign YAML._")
        return None
    expected = cost.by_cell.copy()
    if not expected.empty and "usd_expected" in expected.columns:
        cols = [
            c
            for c in (
                "experiment",
                "memory_method",
                "role",
                "model",
                "prompt_tokens",
                "completion_tokens",
                "usd_expected",
            )
            if c in expected.columns
        ]
        _display_markdown("### Pre-test cost estimate")
        _display_table(expected[cols])
    if not cost.by_stage.empty:
        _display_markdown("### Pre-test cost by stage")
        _display_table(cost.by_stage)
    if not cost.parked.empty:
        _display_markdown("### Parked (not launched)")
        _display_table(cost.parked)
    return cost


def notebook_posttest(
    cfg: CampaignConfig,
    experiment_id: str | None = None,
    *,
    report: ReportResult | list[ReportResult] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Score pre-test expectations against the finished pack + priced actuals.

    ``run_report`` returns a list (one result per experiment, campaign last).
    Pass ``report=reports[-1]`` or the whole list. A list as the second
    positional argument is that report list, not an experiment id.
    """
    root = root or ROOT
    if report is None and not isinstance(experiment_id, (str, type(None))):
        report = experiment_id  # type: ignore[assignment]
        experiment_id = None
    report = _as_report(report)
    if not isinstance(experiment_id, str):
        experiment_id = None
    if report is not None:
        notebook_show(report)
    checks = _posttest_rows(cfg, experiment_id, report, root)
    table = pd.DataFrame(checks)
    _display_markdown("### Post-test")
    if table.empty:
        _display_markdown("_No pre-test expectations declared._")
        return table
    _display_table(table)
    failed = table[table["result"] == FAIL]
    if not failed.empty:
        _display_markdown("**Post-test failures:** " + ", ".join(failed["check"].astype(str)))
    elif (table["result"] == PASS).any():
        _display_markdown("Post-test: declared expectations matched the pack where data exists.")
    cost = report.cost if report is not None else None
    if cost is not None and not cost.by_cell.empty:
        _display_markdown("### Cost expected vs actual")
        _display_table(cost.by_cell)
        for path in cost.plot_paths:
            _display_image(path)
    return table


def _as_report(
    report: ReportResult | list[ReportResult] | None,
) -> ReportResult | None:
    """``run_report`` yields a list; post-test uses the campaign (last) item."""
    if report is None:
        return None
    if isinstance(report, list):
        return report[-1] if report else None
    return report


def _pretest_markdown(cfg: CampaignConfig, experiment_id: str | None) -> str:
    lines = [
        f"## Pre-test — {cfg.title}",
        "",
        cfg.freeze_note,
        "",
        f"Config: `{cfg.source_path.as_posix()}`",
        "",
    ]
    refs = (
        [cfg.experiments[experiment_id]]
        if experiment_id
        else list(cfg.experiments.values())
    )
    for ref in refs:
        lines.append(f"### `{ref.id}` — `{ref.name}`")
        if ref.subset:
            lines.append(f"- subset: {ref.subset}")
        pretest = ref.pretest
        if pretest is None:
            lines.append("- no `pretest:` block")
            lines.append("")
            continue
        if pretest.scientific_claim is False:
            lines.append("- **not a scientific claim**")
        elif pretest.scientific_claim is True:
            lines.append("- scientific claim: yes (still not Mem0 paper J)")
        if pretest.n_cells is not None:
            lines.append(f"- expected cells: **{pretest.n_cells}**")
        if pretest.n_questions is not None:
            lines.append(f"- expected questions / cell: **{pretest.n_questions}**")
        for hyp in pretest.hypotheses:
            lines.append(f"- hypothesis: {hyp}")
        lines.append("")
    if cfg.cost and cfg.cost.parked:
        names = [str(p.get("display_name") or p.get("api_model_id")) for p in cfg.cost.parked]
        lines.append("Parked (costed, not launched): " + ", ".join(names))
        lines.append("")
    return "\n".join(lines)


def _posttest_rows(
    cfg: CampaignConfig,
    experiment_id: str | None,
    report: ReportResult | None,
    root: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    refs = (
        [cfg.experiments[experiment_id]]
        if experiment_id
        else list(cfg.experiments.values())
    )
    missing = set(report.missing_packs if report else [])
    for ref in refs:
        pretest = ref.pretest
        pack_missing = ref.name in missing
        if pretest is None:
            continue
        label = ref.id
        if pretest.scientific_claim is False:
            rows.append(
                {
                    "check": f"{label}.scientific_claim",
                    "result": SKIP,
                    "detail": "declared not a scientific claim (smoke / subset)",
                }
            )
        if pack_missing:
            rows.append(
                {
                    "check": f"{label}.pack",
                    "result": SKIP,
                    "detail": f"pack not local: {ref.name}",
                }
            )
            continue
        n_cells, n_questions = _pack_counts(root, ref)
        if pretest.n_cells is not None:
            ok = n_cells == pretest.n_cells
            rows.append(
                {
                    "check": f"{label}.n_cells",
                    "result": PASS if ok else (SKIP if n_cells is None else FAIL),
                    "detail": f"expected {pretest.n_cells}, actual {n_cells}",
                }
            )
        if pretest.n_questions is not None and n_questions is not None:
            ok = n_questions == pretest.n_questions
            rows.append(
                {
                    "check": f"{label}.n_questions",
                    "result": PASS if ok else FAIL,
                    "detail": f"expected {pretest.n_questions}, actual {n_questions}",
                }
            )
        cost = report.cost if report is not None else None
        if cost is not None and not cost.by_stage.empty:
            stage = cost.by_stage[cost.by_stage["experiment"] == ref.id]
            if not stage.empty:
                expected = stage.iloc[0].get("usd_expected")
                actual = stage.iloc[0].get("usd_actual")
                rows.append(_cost_check(label, expected, actual))
    if report is not None:
        rows.extend(_paper_compare_column_checks(report))
    return rows


_PAPER_COMPARE_GROUP_COLS = frozenset(
    {"paper_method", "live_source", "compare_source", "reader_stack"}
)


def _paper_compare_column_checks(report: ReportResult) -> list[dict[str, str]]:
    """Fail the notebook if a paper-compare table averaged methods together."""
    rows: list[dict[str, str]] = []
    for item in report.results:
        if item.skipped or item.table is None or item.table.empty:
            continue
        missing = [
            col
            for col in item.spec.group_by
            if col in _PAPER_COMPARE_GROUP_COLS and col not in item.table.columns
        ]
        if not missing:
            continue
        rows.append(
            {
                "check": f"{item.spec.id}.group_columns",
                "result": FAIL,
                "detail": (
                    f"grouped table dropped {missing}; "
                    "memory methods or sources were averaged together"
                ),
            }
        )
    return rows


def _cost_check(label: str, expected: object, actual: object) -> dict[str, str]:
    if expected is None or actual is None or pd.isna(expected) or pd.isna(actual):
        return {
            "check": f"{label}.usd",
            "result": SKIP,
            "detail": f"expected={expected}, actual={actual}",
        }
    exp = float(expected)
    act = float(actual)
    if exp <= 0:
        ok = act == 0
    else:
        ratio = act / exp
        ok = 0.5 <= ratio <= 2.0
    return {
        "check": f"{label}.usd",
        "result": PASS if ok else FAIL,
        "detail": f"expected ${exp:.2f}, actual ${act:.2f} (band 0.5–2.0×)",
    }


def _pack_counts(root: Path, ref) -> tuple[int | None, int | None]:
    pack = root / ref.pack if not ref.pack.is_absolute() else ref.pack
    runs_path = pack / "aggregate" / "runs.parquet"
    if not runs_path.is_file():
        return None, None
    runs = pd.read_parquet(runs_path)
    n_cells = len(runs)
    n_questions = None
    if "num_examples" in runs.columns and len(runs):
        n_questions = int(runs["num_examples"].iloc[0])
    return n_cells, n_questions


def _out_dir(cfg: CampaignConfig, experiment_id: str | None, root: Path) -> Path:
    if experiment_id:
        ref = cfg.experiments[experiment_id]
        pack = root / ref.pack if not ref.pack.is_absolute() else ref.pack
        return pack / cfg.output_subdir
    return root / "experiments" / "_campaign" / cfg.id / cfg.output_subdir


def _display_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:
        print(text)


def _display_table(table: pd.DataFrame) -> None:
    try:
        from IPython.display import HTML, display

        display(HTML(dataframe_html(table)))
    except ImportError:
        print(dataframe_markdown(table))


def _display_image(path: Path) -> None:
    try:
        from IPython.display import Image, display

        display(Image(filename=str(path)))
    except ImportError:
        print("plot", path)
