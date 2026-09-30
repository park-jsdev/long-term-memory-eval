"""Campaign analysis YAML + offline report (no API)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_runner.analysis.load_campaign import (
    AnalysisSpec,
    ExperimentAnalysisRef,
    PlotSpec,
    load_campaign_yaml,
)
from src.experiment_runner.analysis.plots import _place_legend_outside
from src.experiment_runner.analysis.notebook_protocol import notebook_posttest
from src.experiment_runner.analysis.report import (
    ReportResult,
    _plot_stem,
    _plot_title,
    _takeaway_sections,
    _takeaways_markdown,
    _write_plot,
    annotate_model_family,
    dataframe_html,
    dataframe_markdown,
    load_pack,
    mean_table,
    render_analysis,
    render_campaign,
    render_experiment,
    run_report,
)
from scripts.analysis.campaign_tables import (
    COMPARE_SOURCE_CODEX,
    annotate_generation,
    annotate_mem0_latency,
    annotate_memory_lane,
    annotate_paper_compare,
    annotate_reader_stack,
    annotate_system_harness,
    annotate_thinking,
    annotate_writer_harness,
    attach_run_agent_audit,
    attach_run_judge_score,
    copy_run_identity_to_examples,
    join_search_latency,
    mean_table,
    repair_run_harness_status,
    annotate_workspace_diagnostics,
)
from src.locomo_eval.mem0_baselines import (
    LOCOMO_2024_SUMMARY_RAG_F1,
    LOCOMO_2024_SUMMARY_RAG_N,
)

VALID_PLOT_KINDS = {"bar", "grouped_bar", "metrics_grouped_bar", "line"}
VALID_SOURCES = {"examples", "runs"}


def _notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source") or [])
        for cell in notebook.get("cells") or []
        if cell.get("cell_type") == "code"
    )


def _write_synthetic_pack(root: Path, name: str, rows: list[dict]) -> None:
    aggregate = root / "experiments" / name / "aggregate"
    aggregate.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(aggregate / "examples.parquet", index=False)


def _write_synthetic_campaign(root: Path) -> Path:
    config_dir = root / "configs" / "analysis"
    config_dir.mkdir(parents=True)
    path = config_dir / "synthetic.yaml"
    path.write_text(
        """
campaign:
  id: synthetic
  title: Synthetic campaign
defaults:
  source: examples
  metrics: [locomo_f1]
  output_subdir: analysis
campaign_analyses:
  - id: family_by_category
    experiments: [alpha]
    group_by: [model_family, question_category]
    plots:
      - kind: grouped_bar
        x: question_category
        hue: model_family
        y: locomo_f1
experiments:
  alpha:
    name: alpha
    pack: experiments/alpha
    family_from: reader
    analyses:
      - id: by_category
        group_by: [reader_display_name, question_category]
        plots:
          - kind: grouped_bar
            x: question_category
            hue: reader_display_name
            y: locomo_f1
  beta:
    name: beta
    pack: experiments/beta
    family_from: reader
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path



class TestGroupedBarKeepsReaderAndMemory(unittest.TestCase):

    def test_category_plot_table_does_not_average_readers_across_memory(self):
        spec = AnalysisSpec(
            id="by_category",
            title="Reader × memory method × category",
            group_by=("reader_display_name", "memory_method", "question_category"),
            metrics=("locomo_f1",),
            plots=(PlotSpec(kind="grouped_bar", x="question_category", y="locomo_f1"),),
            source="examples",
        )
        df = pd.DataFrame(
            [
                {
                    "reader_display_name": "GPT-5",
                    "memory_method": "full_context",
                    "question_category": 1,
                    "locomo_f1": 1.0,
                },
                {
                    "reader_display_name": "GPT-5",
                    "memory_method": "rag",
                    "question_category": 1,
                    "locomo_f1": 0.5,
                },
                {
                    "reader_display_name": "DeepSeek-V3",
                    "memory_method": "full_context",
                    "question_category": 1,
                    "locomo_f1": 0.25,
                },
                {
                    "reader_display_name": "DeepSeek-V3",
                    "memory_method": "rag",
                    "question_category": 1,
                    "locomo_f1": 0.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp))
        self.assertEqual(len(result.table), 4)
        self.assertEqual(
            set(result.table["reader_display_name"]), {"GPT-5", "DeepSeek-V3"}
        )
        self.assertEqual(set(result.table["memory_method"]), {"full_context", "rag"})
        gpt_full = result.table[
            (result.table["reader_display_name"] == "GPT-5")
            & (result.table["memory_method"] == "full_context")
        ].iloc[0]
        self.assertAlmostEqual(float(gpt_full["locomo_f1"]), 1.0)

    def test_grouped_bar_keeps_scores_when_hue_is_boolean_persist(self):
        from scripts.analysis.campaign_plots import _pivot_grouped, write_grouped_bar

        df = pd.DataFrame(
            {
                "agent_sessions": ["full", "full", "notes_only"],
                "agent_persist": [False, True, True],
                "judge_score": [0.58, 0.57, 0.12],
            }
        )
        xs, hues, pivot = _pivot_grouped(
            df, x_col="agent_sessions", hue_col="agent_persist", metric="judge_score"
        )
        self.assertEqual(xs, ["full", "notes_only"])
        self.assertEqual(set(hues), {"False", "True"})
        self.assertAlmostEqual(float(pivot.loc["full", "False"]), 0.58)
        self.assertAlmostEqual(float(pivot.loc["full", "True"]), 0.57)
        self.assertAlmostEqual(float(pivot.loc["notes_only", "True"]), 0.12)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "persist.png"
            written = write_grouped_bar(
                df,
                x_col="agent_sessions",
                hue_col="agent_persist",
                metric="judge_score",
                path=dest,
                title="judge score",
            )
            self.assertEqual(written, dest)
            self.assertTrue(dest.is_file())
            self.assertGreater(dest.stat().st_size, 1000)


class TestJudgeScorePlotDoesNotReuseLocomoF1(unittest.TestCase):
    def test_missing_judge_score_skips_instead_of_drawing_locomo_f1(self):
        spec = AnalysisSpec(
            id="writer_artifacts",
            title="Codex artifacts",
            group_by=("memory_method", "writer_model"),
            metrics=("locomo_f1", "judge_score"),
            plots=(
                PlotSpec(kind="grouped_bar", x="memory_method", y="locomo_f1"),
                PlotSpec(kind="grouped_bar", x="memory_method", y="judge_score"),
            ),
            source="runs",
        )
        df = pd.DataFrame(
            [
                {
                    "memory_method": "graph",
                    "writer_model": "gpt-4o-mini",
                    "locomo_f1": 0.211,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp))
        names = [path.name for path in result.plot_paths]
        self.assertTrue(any("locomo_f1" in name for name in names))
        self.assertFalse(any("judge_score" in name for name in names))
        self.assertNotIn("judge_score", result.table.columns)

    def test_both_metrics_write_distinct_y_axes(self):
        spec = AnalysisSpec(
            id="writer_artifacts",
            title="Codex artifacts",
            group_by=("memory_method", "writer_model"),
            metrics=("locomo_f1", "judge_score"),
            plots=(
                PlotSpec(kind="grouped_bar", x="memory_method", y="locomo_f1"),
                PlotSpec(kind="grouped_bar", x="memory_method", y="judge_score"),
            ),
            source="runs",
        )
        df = pd.DataFrame(
            [
                {
                    "memory_method": "graph",
                    "writer_model": "gpt-4o-mini",
                    "locomo_f1": 0.211,
                    "judge_score": 0.2429,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp))
        names = [path.name for path in result.plot_paths]
        self.assertEqual(len(names), 2)
        self.assertTrue(any("locomo_f1" in name for name in names))
        self.assertTrue(any("judge_score" in name for name in names))
        self.assertAlmostEqual(float(result.table.iloc[0]["judge_score"]), 0.2429)

    def test_runs_receive_cat5_excluded_mean_j_from_examples(self):
        runs = pd.DataFrame(
            [{"run_id": "cell-a", "memory_method": "graph", "locomo_f1": 0.211}]
        )
        examples = pd.DataFrame(
            [
                {"run_id": "cell-a", "question_category": 1, "judge_score": 1.0},
                {"run_id": "cell-a", "question_category": 1, "judge_score": 0.0},
                {"run_id": "cell-a", "question_category": 5, "judge_score": 0.0},
            ]
        )
        out = attach_run_judge_score(runs, examples)
        self.assertAlmostEqual(float(out.iloc[0]["judge_score"]), 0.5)

    def test_runs_locomo_f1_token_f1_and_em_drop_category_five(self):
        runs = pd.DataFrame(
            [
                {
                    "run_id": "cell-a",
                    "memory_method": "workspace_files",
                    "locomo_f1": 0.211,
                    "token_f1": 0.200,
                    "exact_match": 0.100,
                }
            ]
        )
        examples = pd.DataFrame(
            [
                {
                    "run_id": "cell-a",
                    "question_category": 1,
                    "locomo_f1": 1.0,
                    "token_f1": 1.0,
                    "exact_match": 1.0,
                    "judge_score": 1.0,
                },
                {
                    "run_id": "cell-a",
                    "question_category": 5,
                    "locomo_f1": 0.0,
                    "token_f1": 0.0,
                    "exact_match": 0.0,
                    "judge_score": 0.0,
                },
            ]
        )
        out = attach_run_judge_score(runs, examples)
        self.assertAlmostEqual(float(out.iloc[0]["locomo_f1"]), 1.0)
        self.assertAlmostEqual(float(out.iloc[0]["token_f1"]), 1.0)
        self.assertAlmostEqual(float(out.iloc[0]["exact_match"]), 1.0)
        self.assertAlmostEqual(float(out.iloc[0]["judge_score"]), 1.0)

    def test_load_pack_copies_example_j_onto_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agg = root / "experiments" / "pack" / "aggregate"
            agg.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "run_id": "cell-a",
                        "memory_method": "graph",
                        "writer_model": "gpt-4o-mini",
                        "locomo_f1": 0.211,
                    }
                ]
            ).to_parquet(agg / "runs.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "run_id": "cell-a",
                        "question_category": 1,
                        "judge_score": 0.5,
                        "locomo_f1": 0.8,
                        "reader_provider": "openai",
                        "writer_model": "gpt-4o-mini",
                    },
                    {
                        "run_id": "cell-a",
                        "question_category": 5,
                        "judge_score": 0.0,
                        "locomo_f1": 0.0,
                        "reader_provider": "openai",
                        "writer_model": "gpt-4o-mini",
                    },
                ]
            ).to_parquet(agg / "examples.parquet", index=False)
            ref = ExperimentAnalysisRef(
                id="writers",
                name="pack",
                pack=Path("experiments/pack"),
                notebook=None,
                role="sandwich",
                family_from="writer",
                subset=None,
                analyses=(),
            )
            loaded = load_pack(root, ref)
        self.assertIsNotNone(loaded)
        runs, _examples = loaded
        self.assertAlmostEqual(float(runs.iloc[0]["judge_score"]), 0.5)
        self.assertAlmostEqual(float(runs.iloc[0]["locomo_f1"]), 0.8)

    def test_runs_receive_tool_audit_means_including_category_five(self):
        runs = pd.DataFrame(
            [{"run_id": "cell-a", "memory_method": "workspace_files", "locomo_f1": 0.3}]
        )
        examples = pd.DataFrame(
            [
                {
                    "run_id": "cell-a",
                    "question_category": 1,
                    "n_web_search": 0,
                    "n_mcp": 0,
                    "used_non_workspace_tools": 0,
                    "n_retrieval_calls": 2,
                    "memory_recall": 1.0,
                },
                {
                    "run_id": "cell-a",
                    "question_category": 5,
                    "n_web_search": 2,
                    "n_mcp": 0,
                    "used_non_workspace_tools": 1,
                    "n_retrieval_calls": 0,
                    "memory_recall": 0.0,
                },
            ]
        )
        out = attach_run_agent_audit(runs, examples)
        self.assertAlmostEqual(float(out.iloc[0]["n_web_search"]), 1.0)
        self.assertAlmostEqual(float(out.iloc[0]["used_non_workspace_tools"]), 0.5)
        self.assertAlmostEqual(float(out.iloc[0]["n_retrieval_calls"]), 1.0)

    def test_load_pack_backfills_agent_identity_and_audit_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "experiments" / "pack"
            agg = pack / "aggregate"
            agg.mkdir(parents=True)
            manifest = pack / "manifest"
            manifest.mkdir()
            (manifest / "runs.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "cell-a",
                        "agent": "codex",
                        "agent_persist": True,
                        "agent_tools": "native",
                        "writer": {"provider": "codex", "api_model_id": "gpt-4o-mini"},
                        "comparison_contract": {"status": "incomparable"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "run_id": "cell-a",
                        "experiment_name": "locomo-openai-codex-poc-writers",
                        "memory_method": "workspace_files",
                        "writer_model": "gpt-4o-mini",
                        "locomo_f1": 0.3,
                    }
                ]
            ).to_parquet(agg / "runs.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "run_id": "cell-a",
                        "question_category": 1,
                        "n_web_search": 0,
                        "n_mcp": 0,
                        "used_non_workspace_tools": 0,
                        "n_retrieval_calls": 3,
                        "memory_recall": 1.0,
                        "judge_score": 1.0,
                        "reader_provider": "openai",
                        "writer_model": "gpt-4o-mini",
                    }
                ]
            ).to_parquet(agg / "examples.parquet", index=False)
            ref = ExperimentAnalysisRef(
                id="readers",
                name="pack",
                pack=Path("experiments/pack"),
                notebook=None,
                role="agent_reader",
                family_from="reader",
                subset=None,
                analyses=(),
            )
            loaded = load_pack(root, ref)
        self.assertIsNotNone(loaded)
        runs, _examples = loaded
        self.assertEqual(str(runs.iloc[0]["agent"]), "codex")
        self.assertEqual(bool(runs.iloc[0]["agent_persist"]), True)
        self.assertEqual(str(runs.iloc[0]["comparison_status"]), "incomparable")
        self.assertEqual(str(runs.iloc[0]["writer_harness"]), "codex")
        self.assertAlmostEqual(float(runs.iloc[0]["n_retrieval_calls"]), 3.0)

    def test_annotate_writer_harness_labels_chat_completions_teachers(self):
        df = pd.DataFrame(
            [
                {
                    "experiment_name": "locomo-openai-agent-writers",
                    "writer_model": "gpt-5",
                    "writer_provider": "openai",
                },
                {
                    "experiment_name": "locomo-openai-codex-writers",
                    "writer_model": "gpt-5",
                    "writer_provider": "codex",
                },
            ]
        )
        out = annotate_writer_harness(df)
        self.assertEqual(list(out["writer_harness"]), ["chat_completions", "codex"])

    def test_annotate_memory_lane_maps_persist_off_workspace_to_full_context(self):
        df = pd.DataFrame(
            [
                {"memory_method": "full_context", "agent_persist": None},
                {"memory_method": "workspace_files", "agent_persist": False},
                {"memory_method": "workspace_files", "agent_persist": True},
                {"memory_method": "session_summaries", "agent_persist": None},
                {"memory_method": "graph", "agent_persist": None},
                {"memory_method": "agent_codex_mem0_facts", "agent_persist": None},
            ]
        )
        out = annotate_memory_lane(df)
        self.assertEqual(
            list(out["memory_lane"]),
            [
                "full_context",
                "full_context",
                None,
                "session_summaries",
                "graph",
                None,
            ],
        )

    def test_annotate_system_harness_labels_reader_and_writer_paths(self):
        df = pd.DataFrame(
            [
                {
                    "memory_method": "full_context",
                    "agent": None,
                    "writer_model": None,
                    "writer_provider": None,
                },
                {
                    "memory_method": "workspace_files",
                    "agent": "codex",
                    "agent_persist": False,
                    "writer_model": None,
                    "writer_provider": None,
                },
                {
                    "memory_method": "session_summaries",
                    "agent": None,
                    "writer_model": "gpt-4o-mini",
                    "writer_provider": "openai",
                },
                {
                    "memory_method": "graph",
                    "agent": None,
                    "writer_model": "gpt-4o-mini",
                    "writer_provider": "codex",
                },
            ]
        )
        out = annotate_system_harness(df)
        self.assertEqual(
            list(out["system_harness"]),
            ["chat_completions", "codex", "chat_completions", "codex"],
        )

    def test_mean_table_memory_lane_keeps_six_ceiling_cells(self):
        df = pd.DataFrame(
            [
                {
                    "memory_method": "full_context",
                    "agent": None,
                    "writer_model": None,
                    "writer_provider": None,
                    "agent_persist": None,
                    "question_category": 1,
                    "judge_score": 0.74,
                    "locomo_f1": 0.53,
                },
                {
                    "memory_method": "workspace_files",
                    "agent": "codex",
                    "writer_model": None,
                    "writer_provider": None,
                    "agent_persist": False,
                    "question_category": 1,
                    "judge_score": 0.50,
                    "locomo_f1": 0.22,
                },
                {
                    "memory_method": "workspace_files",
                    "agent": "codex",
                    "writer_model": None,
                    "writer_provider": None,
                    "agent_persist": True,
                    "question_category": 1,
                    "judge_score": 0.49,
                    "locomo_f1": 0.20,
                },
                {
                    "memory_method": "session_summaries",
                    "agent": None,
                    "writer_model": "gpt-4o-mini",
                    "writer_provider": "openai",
                    "agent_persist": None,
                    "question_category": 1,
                    "judge_score": 0.69,
                    "locomo_f1": 0.47,
                },
                {
                    "memory_method": "session_summaries",
                    "agent": None,
                    "writer_model": "gpt-4o-mini",
                    "writer_provider": "codex",
                    "agent_persist": None,
                    "question_category": 1,
                    "judge_score": 0.57,
                    "locomo_f1": 0.41,
                },
                {
                    "memory_method": "graph",
                    "agent": None,
                    "writer_model": "gpt-4o-mini",
                    "writer_provider": "openai",
                    "agent_persist": None,
                    "question_category": 1,
                    "judge_score": 0.36,
                    "locomo_f1": 0.26,
                },
                {
                    "memory_method": "graph",
                    "agent": None,
                    "writer_model": "gpt-4o-mini",
                    "writer_provider": "codex",
                    "agent_persist": None,
                    "question_category": 1,
                    "judge_score": 0.26,
                    "locomo_f1": 0.19,
                },
                {
                    "memory_method": "agent_codex_mem0_facts",
                    "agent": None,
                    "writer_model": "gpt-4o-mini",
                    "writer_provider": "codex",
                    "agent_persist": None,
                    "question_category": 1,
                    "judge_score": 0.40,
                    "locomo_f1": 0.33,
                },
            ]
        )
        work = annotate_memory_lane(annotate_system_harness(df))
        work = work[work["memory_lane"].isin(
            ["full_context", "session_summaries", "graph"]
        )]
        table = mean_table(
            work, ["memory_lane", "system_harness"], ["judge_score", "locomo_f1"]
        )
        self.assertEqual(len(table), 6)
        keys = set(zip(table["memory_lane"], table["system_harness"]))
        self.assertEqual(
            keys,
            {
                ("full_context", "chat_completions"),
                ("full_context", "codex"),
                ("session_summaries", "chat_completions"),
                ("session_summaries", "codex"),
                ("graph", "chat_completions"),
                ("graph", "codex"),
            },
        )


class TestHarnessStatusRepair(unittest.TestCase):
    def test_partial_harness_failures_are_incomparable_not_cell_failed(self):
        runs = pd.DataFrame(
            [
                {
                    "run_id": "cell-a",
                    "comparison_status": "harness_failed",
                    "locomo_f1": 0.2,
                }
            ]
        )
        examples = pd.DataFrame(
            [
                {
                    "run_id": "cell-a",
                    "failure_mode": "reasoning_failure",
                    "question_category": 1,
                },
                {
                    "run_id": "cell-a",
                    "failure_mode": "harness_execution_failure",
                    "question_category": 1,
                },
            ]
        )
        out = repair_run_harness_status(runs, examples)
        self.assertEqual(str(out.iloc[0]["comparison_status"]), "incomparable")
        self.assertEqual(int(out.iloc[0]["n_harness_failed"]), 1)
        self.assertAlmostEqual(float(out.iloc[0]["harness_failed_rate"]), 0.5)

    def test_all_failed_keeps_harness_failed_status(self):
        runs = pd.DataFrame(
            [{"run_id": "cell-a", "comparison_status": "harness_failed"}]
        )
        examples = pd.DataFrame(
            [
                {"run_id": "cell-a", "failure_mode": "harness_execution_failure"},
                {"run_id": "cell-a", "failure_mode": "harness_execution_failure"},
            ]
        )
        out = repair_run_harness_status(runs, examples)
        self.assertEqual(str(out.iloc[0]["comparison_status"]), "harness_failed")
        self.assertAlmostEqual(float(out.iloc[0]["harness_failed_rate"]), 1.0)

    def test_load_pack_repairs_stored_any_failed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "experiments" / "pack"
            agg = pack / "aggregate"
            agg.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "run_id": "cell-a",
                        "memory_method": "workspace_files",
                        "comparison_status": "harness_failed",
                        "locomo_f1": 0.2,
                    }
                ]
            ).to_parquet(agg / "runs.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "run_id": "cell-a",
                        "question_category": 1,
                        "failure_mode": "reasoning_failure",
                        "judge_score": 1.0,
                        "n_retrieval_calls": 4,
                        "memory_recall": 1.0,
                    },
                    {
                        "run_id": "cell-a",
                        "question_category": 1,
                        "failure_mode": "harness_execution_failure",
                        "judge_score": 0.0,
                        "n_retrieval_calls": 0,
                        "memory_recall": 0.0,
                    },
                ]
            ).to_parquet(agg / "examples.parquet", index=False)
            ref = ExperimentAnalysisRef(
                id="readers",
                name="pack",
                pack=Path("experiments/pack"),
                notebook=None,
                role="agent_reader",
                family_from="reader",
                subset=None,
                analyses=(),
            )
            loaded = load_pack(root, ref)
        self.assertIsNotNone(loaded)
        runs, _examples = loaded
        self.assertEqual(str(runs.iloc[0]["comparison_status"]), "incomparable")
        self.assertAlmostEqual(float(runs.iloc[0]["harness_failed_rate"]), 0.5)

    def test_failure_mode_count_table_labels_and_keeps_category_five(self):
        spec = AnalysisSpec(
            id="failure_modes",
            title="Failure modes",
            group_by=("failure_mode",),
            metrics=("n",),
            plots=(PlotSpec(kind="grouped_bar", x="failure_mode", y="n"),),
            source="examples",
        )
        df = pd.DataFrame(
            [
                {"failure_mode": "reasoning_failure", "question_category": 1},
                {"failure_mode": "harness_execution_failure", "question_category": 5},
            ]
        )
        table = mean_table(df, ["failure_mode"], ["n"])
        self.assertEqual(int(table["n"].sum()), 2)
        self.assertIn("wrong after retrieve", list(table["failure_mode"]))
        self.assertIn("no workspace read", list(table["failure_mode"]))
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp))
        self.assertEqual(int(result.table["n"].sum()), 2)

    def test_workspace_diagnostics_bin_recall_and_judge_vs_f1(self):
        df = pd.DataFrame(
            [
                {
                    "generated_answer": "Melanie is going camping next month.",
                    "reference_answer": "June 2023",
                    "memory_recall": 1.0,
                    "n_retrieval_calls": 1,
                    "judge_score": 1.0,
                    "locomo_f1": 0.0,
                    "evidence_retrieved": True,
                    "failure_mode": "reasoning_failure",
                },
                {
                    "generated_answer": "June 2023",
                    "reference_answer": "June 2023",
                    "memory_recall": 0.5,
                    "n_retrieval_calls": 12,
                    "judge_score": 1.0,
                    "locomo_f1": 1.0,
                    "evidence_retrieved": False,
                    "failure_mode": "parametric_success",
                },
            ]
        )
        out = annotate_workspace_diagnostics(df)
        self.assertEqual(out.iloc[0]["recall_bin"], "full")
        self.assertEqual(out.iloc[1]["recall_bin"], "partial")
        self.assertEqual(out.iloc[0]["judge_vs_f1"], "j_only")
        self.assertEqual(out.iloc[1]["judge_vs_f1"], "both_correct")
        self.assertEqual(out.iloc[0]["failure_mode_judge"], "none")
        self.assertEqual(out.iloc[1]["failure_mode_judge"], "parametric_success")
        self.assertEqual(int(out.iloc[0]["answer_n_words"]), 6)
        self.assertEqual(out.iloc[1]["retrieval_calls_bin"], "9-20")
        table = mean_table(out, ["recall_bin"], ["n"])
        self.assertIn("all gold ids retrieved", list(table["recall_bin"]))
        self.assertIn("partial gold-id chain", list(table["recall_bin"]))

    def test_annotate_hop_qidx_and_notes_bins(self):
        df = pd.DataFrame(
            [
                {
                    "question_id": "conv-26-q-0",
                    "hop_to_evidence": 2,
                    "notes_bytes": 128,
                    "judge_score": 1.0,
                    "locomo_f1": 0.4,
                },
                {
                    "question_id": "conv-26-q-12",
                    "hop_to_evidence": None,
                    "notes_bytes": 0,
                    "judge_score": 0.0,
                    "locomo_f1": 0.0,
                },
            ]
        )
        out = annotate_workspace_diagnostics(df)
        self.assertEqual(int(out.iloc[0]["qidx"]), 0)
        self.assertEqual(out.iloc[0]["qidx_bin"], "0-4")
        self.assertEqual(out.iloc[1]["qidx_bin"], "10-19")
        self.assertEqual(out.iloc[0]["hop_bin"], "2")
        self.assertEqual(out.iloc[1]["hop_bin"], "never")
        self.assertEqual(out.iloc[0]["notes_bytes_bin"], "1-256")
        self.assertEqual(out.iloc[1]["notes_bytes_bin"], "empty")

    def test_overlay_collected_agent_audit_fills_hop_from_trajectory(self):
        from scripts.analysis.agent_harness import overlay_collected_agent_audit

        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            agent = pack / "collected" / "runs" / "cell-a" / "agent"
            agent.mkdir(parents=True)
            (agent / "trajectory.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "conv-26-q-0",
                        "evidence_ids_required": ["D1:1"],
                        "events": [
                            {
                                "step": 1,
                                "kind": "catalog",
                                "target": "INDEX.md",
                                "retrieved_text": "Sessions",
                            },
                            {
                                "step": 2,
                                "kind": "retrieve",
                                "target": "sessions/session_1.md",
                                "retrieved_text": "painting (D1:1)",
                                "evidence_ids_hit": ["D1:1"],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (agent / "notes_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "conv-26-q-0",
                        "notes_bytes": 220,
                        "notes_words": 40,
                        "notes_grew": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            examples = pd.DataFrame(
                [
                    {
                        "run_id": "cell-a",
                        "question_id": "conv-26-q-0",
                        "locomo_f1": 0.2,
                    }
                ]
            )
            out = overlay_collected_agent_audit(examples, pack)
        self.assertEqual(int(out.iloc[0]["hop_to_evidence"]), 2)
        self.assertEqual(int(out.iloc[0]["notes_bytes"]), 220)
        self.assertTrue(bool(out.iloc[0]["notes_grew"]))

    def test_copy_run_persist_onto_examples(self):
        examples = pd.DataFrame(
            [{"run_id": "cell-a", "generated_answer": "hi", "locomo_f1": 0.2}]
        )
        runs = pd.DataFrame(
            [{"run_id": "cell-a", "agent": "codex", "agent_persist": True}]
        )
        out = copy_run_identity_to_examples(examples, runs)
        self.assertEqual(str(out.iloc[0]["agent"]), "codex")
        self.assertEqual(bool(out.iloc[0]["agent_persist"]), True)


class TestAdversarialExcludedFromOverall(unittest.TestCase):
    def test_overall_metrics_drop_category_5_without_yaml_exclude(self):
        spec = AnalysisSpec(
            id="by_reader",
            title="Per-reader cell means",
            group_by=("reader_display_name",),
            metrics=("locomo_f1", "judge_score"),
            plots=(PlotSpec(kind="metrics_grouped_bar"),),
            source="examples",
        )
        df = pd.DataFrame(
            [
                {
                    "reader_display_name": "GPT-5",
                    "question_category": 1,
                    "locomo_f1": 1.0,
                    "judge_score": 1.0,
                },
                {
                    "reader_display_name": "GPT-5",
                    "question_category": 5,
                    "locomo_f1": 0.0,
                    "judge_score": 0.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp))
        self.assertEqual(len(result.table), 1)
        self.assertAlmostEqual(float(result.table.iloc[0]["locomo_f1"]), 1.0)
        self.assertAlmostEqual(float(result.table.iloc[0]["judge_score"]), 1.0)
        self.assertEqual(int(result.table.iloc[0]["n"]), 1)

    def test_category_plots_keep_category_5(self):
        spec = AnalysisSpec(
            id="by_category",
            title="Per-reader LoCoMo F1 by category",
            group_by=("reader_display_name", "question_category"),
            metrics=("locomo_f1",),
            plots=(
                PlotSpec(
                    kind="grouped_bar", x="question_category", y="locomo_f1"
                ),
            ),
            source="examples",
        )
        df = pd.DataFrame(
            [
                {
                    "reader_display_name": "GPT-5",
                    "question_category": 1,
                    "locomo_f1": 1.0,
                },
                {
                    "reader_display_name": "GPT-5",
                    "question_category": 5,
                    "locomo_f1": 0.25,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp))
        labels = set(result.table["question_category"])
        self.assertIn("1 multi-hop", labels)
        self.assertIn("5 adversarial", labels)
        adv = result.table[result.table["question_category"] == "5 adversarial"].iloc[0]
        self.assertAlmostEqual(float(adv["locomo_f1"]), 0.25)


class TestAnalysisYamlContract(unittest.TestCase):
    def test_every_campaign_analysis_references_configured_experiments(self):
        for path in sorted((ROOT / "configs" / "analysis").glob("*.yaml")):
            with self.subTest(path=path.name):
                cfg = load_campaign_yaml(path)
                configured = set(cfg.experiments)
                for spec in cfg.campaign_analyses:
                    self.assertTrue(spec.experiments)
                    self.assertLessEqual(set(spec.experiments), configured)

    def test_every_analysis_uses_valid_sources_and_plot_columns(self):
        for path in sorted((ROOT / "configs" / "analysis").glob("*.yaml")):
            cfg = load_campaign_yaml(path)
            specs = list(cfg.campaign_analyses)
            for ref in cfg.experiments.values():
                specs.extend(ref.analyses)
            for spec in specs:
                with self.subTest(path=path.name, analysis=spec.id):
                    self.assertIn(spec.source, VALID_SOURCES)
                    self.assertTrue(spec.group_by)
                    self.assertTrue(spec.metrics)
                    self.assertTrue(spec.plots)
                    for plot in spec.plots:
                        self.assertIn(plot.kind, VALID_PLOT_KINDS)
                        if plot.x:
                            self.assertIn(plot.x, spec.group_by)
                        if plot.hue:
                            self.assertIn(plot.hue, spec.group_by)
                        if plot.y:
                            self.assertIn(plot.y, spec.metrics)

    def test_analysis_ids_are_unique_within_each_output_scope(self):
        for path in sorted((ROOT / "configs" / "analysis").glob("*.yaml")):
            cfg = load_campaign_yaml(path)
            campaign_ids = [spec.id for spec in cfg.campaign_analyses]
            self.assertEqual(len(campaign_ids), len(set(campaign_ids)), path.name)
            for ref in cfg.experiments.values():
                ids = [spec.id for spec in ref.analyses]
                ids.extend(
                    spec.id
                    for spec in cfg.campaign_analyses
                    if ref.id in spec.experiments
                )
                with self.subTest(path=path.name, experiment=ref.id):
                    self.assertEqual(len(ids), len(set(ids)))


class TestNotebookMarkdownTables(unittest.TestCase):
    def test_dataframe_markdown_keeps_columns_as_pipe_table(self):
        table = pd.DataFrame(
            [
                {
                    "memory_method": "workspace_files",
                    "agent_persist": False,
                    "n": 1540,
                    "locomo_f1": 0.24815758,
                    "judge_score": 0.67824675,
                }
            ]
        )
        text = dataframe_markdown(table)
        self.assertIn("| memory_method | agent_persist | n | locomo_f1 | judge_score |", text)
        self.assertIn("| --- | --- | --- | --- | --- |", text)
        self.assertIn("workspace_files", text)
        self.assertIn("false", text)
        self.assertIn("| 1540 |", text)
        self.assertIn("0.2482", text)
        self.assertNotIn("0.24815758", text)

    def test_takeaways_markdown_uses_pipe_table_not_to_string(self):
        table = pd.DataFrame(
            [
                {
                    "takeaway_id": "harness_reader_vs_chat_reader",
                    "title": "Codex vs full_context",
                    "claim": "harness_reader_weaker_than_chat_reader",
                    "finding": "J drops a little.",
                    "left_label": "workspace_codex",
                    "right_label": "chat_full_context",
                    "metric": "judge_score",
                    "n_left": 3080.0,
                    "n_right": 1540.0,
                    "left_value": 0.678247,
                    "right_value": 0.743506,
                    "delta": -0.065259,
                }
            ]
        )
        lines = _takeaways_markdown(table, csv_path=None)
        text = "\n".join(lines)
        self.assertIn("| left_label | right_label | metric |", text)
        self.assertIn("workspace_codex", text)
        self.assertNotIn("     left_label", text)

    def test_takeaway_sections_keep_metric_table_out_of_heading_markdown(self):
        table = pd.DataFrame(
            [
                {
                    "takeaway_id": "harness_retrieval_not_persist",
                    "title": "Harness value is retrieval, not persist-on notes",
                    "claim": "harness_retrieval_not_memory_method",
                    "finding": "J barely moves.",
                    "left_label": "persist_on",
                    "right_label": "persist_off",
                    "metric": "judge_score",
                    "n_left": 1.0,
                    "n_right": 1.0,
                    "left_value": 0.683766,
                    "right_value": 0.672727,
                    "delta": 0.011039,
                }
            ]
        )
        sections = _takeaway_sections(table, csv_path=None)
        headings = [heading for heading, _metrics in sections]
        self.assertTrue(any("J barely moves." in h for h in headings))
        self.assertFalse(any("| left_label |" in h for h in headings))
        metrics = [frame for _heading, frame in sections if frame is not None]
        self.assertEqual(len(metrics), 1)
        html = dataframe_html(metrics[0])
        self.assertIn("<table", html)
        self.assertIn("persist_on", html)
        self.assertIn("<td>0.6838</td>", html)


class TestAnalysisNotebookContract(unittest.TestCase):

    def test_openai_agent_notebooks_are_thin_yaml_wrappers(self):
        pairs = (
            (
                "campaign_openai_codex_poc.yaml",
                "17_openai_codex_poc_analysis.ipynb",
            ),
            (
                "campaign_openai_agents.yaml",
                "17_openai_agent_reader_writer_analysis.ipynb",
            ),
            (
                "campaign_openai_codex_persist_memory.yaml",
                "17_openai_codex_persist_memory_analysis.ipynb",
            ),
            (
                "campaign_openai_mini_vs_codex_writers.yaml",
                "17_openai_mini_vs_codex_writers_analysis.ipynb",
            ),
            (
                "campaign_openai_model_harness_gaps.yaml",
                "17_openai_model_harness_gaps_analysis.ipynb",
            ),
        )
        for yaml_name, notebook_name in pairs:
            with self.subTest(notebook=notebook_name):
                code = _notebook_code(ROOT / "notebooks" / notebook_name)
                self.assertIn(yaml_name, code)
                self.assertIn("notebook_pretest(camp, root=ROOT)", code)
                self.assertIn("notebook_posttest(camp,", code)
                self.assertIn("report=", code)
                self.assertIn("run_report(YAML, root=ROOT)", code)
                self.assertNotIn("matplotlib", code)
                self.assertNotIn("groupby(", code)


class TestMeanTableAndFamily(unittest.TestCase):
    def test_mean_table_groups_reader_and_keeps_count(self):
        df = pd.DataFrame(
            {
                "reader_display_name": ["GPT-5", "GPT-5", "Claude Sonnet 4.5"],
                "memory_method": ["full_context", "full_context", "full_context"],
                "locomo_f1": [1.0, 0.5, 0.0],
                "judge_score": [1.0, 1.0, 0.0],
            }
        )
        table = mean_table(df, ["reader_display_name"], ["locomo_f1", "judge_score"])
        self.assertEqual(len(table), 2)
        gpt = table[table["reader_display_name"] == "GPT-5"].iloc[0]
        self.assertAlmostEqual(float(gpt["locomo_f1"]), 0.75)
        self.assertEqual(int(gpt["n"]), 2)

    def test_annotate_model_family_from_reader_provider(self):
        df = pd.DataFrame(
            {
                "reader_provider": ["openai", "anthropic", "deepseek"],
                "locomo_f1": [0.8, 0.1, 0.7],
            }
        )
        out = annotate_model_family(df, "reader")
        self.assertEqual(
            list(out["model_family"]), ["OpenAI", "Anthropic", "DeepSeek"]
        )

    def test_annotate_model_family_from_writer_model_id(self):
        df = pd.DataFrame(
            {
                "writer_model": ["gpt-5", "claude-sonnet-4-5-20250929", "deepseek-chat"],
                "locomo_f1": [0.5, 0.4, 0.3],
            }
        )
        out = annotate_model_family(df, "writer")
        self.assertEqual(
            list(out["model_family"]), ["OpenAI", "Anthropic", "DeepSeek"]
        )

    def test_mean_table_maps_locomo_category_ids_to_eval_names(self):
        df = pd.DataFrame(
            {
                "question_category": [2, 1, 1],
                "locomo_f1": [0.5, 1.0, 0.0],
            }
        )
        table = mean_table(df, ["question_category"], ["locomo_f1"])
        self.assertEqual(
            list(table["question_category"]),
            ["1 multi-hop", "2 temporal"],
        )
        self.assertAlmostEqual(float(table.iloc[0]["locomo_f1"]), 0.5)

    def test_mean_table_p50_p95_use_source_latency_column(self):
        df = pd.DataFrame(
            {
                "model_family": ["OpenAI"] * 4,
                "agent_latency_seconds": [1.0, 2.0, 3.0, 4.0],
            }
        )
        table = mean_table(
            df,
            ["model_family"],
            ["agent_latency_seconds", "agent_latency_seconds_p50", "agent_latency_seconds_p95"],
        )
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(float(table.iloc[0]["agent_latency_seconds"]), 2.5)
        self.assertAlmostEqual(float(table.iloc[0]["agent_latency_seconds_p50"]), 2.5)
        self.assertGreaterEqual(float(table.iloc[0]["agent_latency_seconds_p95"]), 3.5)

    def test_plot_title_uses_y_metric_not_analysis_section_title(self):
        spec = AnalysisSpec(
            id="reader_thinking_tokens",
            title="Reader reasoning tokens vs latency",
            group_by=("model_family",),
            metrics=("agent_reasoning_tokens", "agent_latency_seconds"),
            plots=(),
            source="examples",
        )
        tokens = PlotSpec(kind="grouped_bar", y="agent_reasoning_tokens")
        latency = PlotSpec(kind="grouped_bar", y="agent_latency_seconds")
        search = PlotSpec(kind="grouped_bar", y="search_latency_seconds_p50")
        total = PlotSpec(kind="grouped_bar", y="total_latency_seconds_p95")
        self.assertEqual(
            _plot_title(tokens, spec, "agent_reasoning_tokens"),
            "Reader reasoning tokens",
        )
        self.assertEqual(
            _plot_title(latency, spec, "agent_latency_seconds"),
            "Reader generate latency (s)",
        )
        self.assertEqual(
            _plot_title(search, spec, "search_latency_seconds_p50"),
            "Search latency (s) p50",
        )
        self.assertEqual(
            _plot_title(total, spec, "total_latency_seconds_p95"),
            "Total latency (s) p95",
        )

    def test_annotate_mem0_latency_keeps_rag_search_missing_and_zeros_full_context(self):
        df = pd.DataFrame(
            {
                "memory_method": ["full_context", "rag"],
                "agent_latency_seconds": [1.0, 2.0],
            }
        )
        out = annotate_mem0_latency(df)
        self.assertEqual(float(out.loc[0, "search_latency_seconds"]), 0.0)
        self.assertTrue(pd.isna(out.loc[1, "search_latency_seconds"]))
        self.assertEqual(float(out.loc[0, "total_latency_seconds"]), 1.0)
        self.assertEqual(float(out.loc[1, "total_latency_seconds"]), 2.0)

    def test_join_search_latency_fills_rag_from_predictions_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "exp"
            pack.mkdir()
            run_id = "cell-rag"
            run_dir = root / run_id
            run_dir.mkdir()
            (run_dir / "predictions.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "search_latency_s": 0.42,
                        "latency_s": 1.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            df = pd.DataFrame(
                {
                    "run_id": [run_id],
                    "question_id": ["q1"],
                    "memory_method": ["rag"],
                    "agent_latency_seconds": [1.5],
                }
            )
            joined = join_search_latency(df, pack)
            self.assertAlmostEqual(float(joined.iloc[0]["search_latency_seconds"]), 0.42)
            out = annotate_mem0_latency(joined)
            self.assertAlmostEqual(float(out.iloc[0]["total_latency_seconds"]), 1.92)

    def test_annotate_thinking_reads_teacher_flag_from_catalog_run_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "exp"
            meta_dir = pack / "aggregate" / "by_run" / "cell-on"
            meta_dir.mkdir(parents=True)
            (meta_dir / "run_meta.json").write_text(
                json.dumps({"run_id": "cell-on", "teacher_thinking": True}),
                encoding="utf-8",
            )
            off_dir = pack / "aggregate" / "by_run" / "cell-off"
            off_dir.mkdir(parents=True)
            (off_dir / "run_meta.json").write_text(
                json.dumps({"run_id": "cell-off", "teacher_thinking": False}),
                encoding="utf-8",
            )
            df = pd.DataFrame(
                {
                    "run_id": ["cell-on", "cell-off", "cell-on"],
                    "model_family": ["OpenAI", "OpenAI", "OpenAI"],
                    "locomo_f1": [1.0, 0.0, 0.5],
                }
            )
            out = annotate_thinking(df, pack)
            self.assertEqual(list(out["thinking"]), ["on", "off", "on"])
            table = mean_table(out, ["model_family", "thinking"], ["locomo_f1"])
            self.assertEqual(len(table), 2)
            on = table[table["thinking"] == "on"].iloc[0]
            off = table[table["thinking"] == "off"].iloc[0]
            self.assertAlmostEqual(float(on["locomo_f1"]), 0.75)
            self.assertAlmostEqual(float(off["locomo_f1"]), 0.0)

    def test_thinking_within_family_plot_keeps_on_and_off_as_hue(self):
        spec = AnalysisSpec(
            id="thinking_within_reader",
            title="Thinking on vs off within reader",
            group_by=("model_family", "thinking"),
            metrics=("locomo_f1",),
            plots=(
                PlotSpec(
                    kind="grouped_bar",
                    x="model_family",
                    hue="thinking",
                    y="locomo_f1",
                ),
            ),
            source="examples",
        )
        df = pd.DataFrame(
            {
                "model_family": ["OpenAI", "OpenAI", "DeepSeek", "DeepSeek"],
                "thinking": ["off", "on", "off", "on"],
                "locomo_f1": [0.2, 0.8, 0.3, 0.9],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp))
            self.assertIsNone(result.skipped)
            self.assertEqual(len(result.table), 4)
            openai = result.table[result.table["model_family"] == "OpenAI"]
            self.assertEqual(set(openai["thinking"]), {"off", "on"})
            self.assertAlmostEqual(
                float(openai[openai["thinking"] == "off"]["locomo_f1"].iloc[0]), 0.2
            )
            self.assertAlmostEqual(
                float(openai[openai["thinking"] == "on"]["locomo_f1"].iloc[0]), 0.8
            )
            self.assertTrue(result.plot_paths)
            self.assertTrue(result.plot_paths[0].is_file())

    def test_plot_stem_keeps_bar_and_line_filenames_distinct(self):
        bar = PlotSpec(kind="grouped_bar", x="generation", y="locomo_f1")
        line = PlotSpec(kind="line", x="generation", y="locomo_f1")
        self.assertNotEqual(_plot_stem("reader_live", bar), _plot_stem("reader_live", line))
        self.assertIn("line", _plot_stem("reader_live", line))

    def test_line_plot_writes_year_series_png(self):
        spec = AnalysisSpec(
            id="reader_live_year_family",
            title="Live readers",
            group_by=("generation", "model_family", "memory_method", "thinking"),
            metrics=("locomo_f1",),
            plots=(PlotSpec(kind="line", x="generation", y="locomo_f1"),),
            source="examples",
        )
        df = pd.DataFrame(
            {
                "generation": ["2025", "2026", "2025", "2026"],
                "model_family": ["OpenAI", "OpenAI", "OpenAI", "OpenAI"],
                "memory_method": ["full_context", "full_context", "rag", "rag"],
                "thinking": ["off", "off", "off", "off"],
                "locomo_f1": [0.50, 0.62, 0.40, 0.41],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp))
            self.assertIsNone(result.skipped)
            self.assertTrue(result.plot_paths)
            self.assertTrue(result.plot_paths[0].is_file())
            self.assertGreater(result.plot_paths[0].stat().st_size, 0)



class TestDeterministicAnalysisReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.yaml_path = _write_synthetic_campaign(self.root)
        self.alpha_rows = [
            {
                "reader_provider": "openai",
                "reader_display_name": "GPT",
                "question_category": 2,
                "locomo_f1": 0.5,
            },
            {
                "reader_provider": "deepseek",
                "reader_display_name": "DeepSeek",
                "question_category": 1,
                "locomo_f1": 1.0,
            },
            {
                "reader_provider": "openai",
                "reader_display_name": "GPT",
                "question_category": 1,
                "locomo_f1": 0.0,
            },
        ]
        _write_synthetic_pack(self.root, "alpha", self.alpha_rows)
        _write_synthetic_pack(
            self.root,
            "beta",
            [
                {
                    "reader_provider": "anthropic",
                    "reader_display_name": "Claude",
                    "question_category": 5,
                    "locomo_f1": 1.0,
                }
            ],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_experiment_report_is_identical_after_input_row_reordering(self):
        cfg = load_campaign_yaml(self.yaml_path)
        first = render_experiment(cfg, "alpha", root=self.root)
        first_result = first.results[0]
        self.assertIsNone(first_result.skipped)
        self.assertTrue(first_result.csv_path.is_file())
        self.assertTrue(first_result.plot_paths[0].is_file())
        self.assertTrue((first.out_dir / "SUMMARY.md").is_file())
        csv_hash = hashlib.sha256(first_result.csv_path.read_bytes()).hexdigest()
        plot_hash = hashlib.sha256(first_result.plot_paths[0].read_bytes()).hexdigest()

        aggregate = self.root / "experiments" / "alpha" / "aggregate"
        pd.DataFrame(list(reversed(self.alpha_rows))).to_parquet(
            aggregate / "examples.parquet", index=False
        )
        second = render_experiment(cfg, "alpha", root=self.root)
        second_result = second.results[0]

        self.assertEqual(
            csv_hash,
            hashlib.sha256(second_result.csv_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            plot_hash,
            hashlib.sha256(second_result.plot_paths[0].read_bytes()).hexdigest(),
        )
        pd.testing.assert_frame_equal(first_result.table, second_result.table)

    def test_campaign_report_uses_only_recipe_selected_experiments(self):
        cfg = load_campaign_yaml(self.yaml_path)
        report = render_campaign(cfg, root=self.root)
        result = report.results[0]

        self.assertEqual(result.spec.id, "family_by_category")
        self.assertEqual(set(result.table["model_family"]), {"OpenAI", "DeepSeek"})
        self.assertNotIn("Anthropic", set(result.table["model_family"]))
        self.assertEqual(int(result.table["n"].sum()), len(self.alpha_rows))
        self.assertEqual(
            set(result.table["question_category"]),
            {"1 multi-hop", "2 temporal"},
        )


class TestLegendDoesNotCoverBars(unittest.TestCase):
    def test_legend_window_is_to_the_right_of_the_axes(self):
        from src.experiment_runner.analysis.plots import _pyplot

        plt = _pyplot()
        if plt is None:
            self.skipTest("matplotlib missing")
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        ax.bar([0], [1.0], label="OpenAI × full_context")
        ax.bar([0.4], [0.8], label="DeepSeek × full_context")
        ax.set_ylim(0, 1.05)
        _place_legend_outside(ax)
        fig.canvas.draw()
        ax_box = ax.get_window_extent()
        leg_box = ax.get_legend().get_window_extent()
        self.assertGreaterEqual(leg_box.x0, ax_box.x1 - 2)
        plt.close(fig)

    def test_reasoning_token_and_latency_metrics_are_not_score_scaled(self):
        from src.experiment_runner.analysis.plots import unbounded_metric

        self.assertTrue(unbounded_metric("agent_reasoning_tokens"))
        self.assertTrue(unbounded_metric("agent_input_tokens_mean"))
        self.assertTrue(unbounded_metric("total_latency_seconds_p95"))
        self.assertTrue(unbounded_metric("search_latency_seconds"))
        self.assertFalse(unbounded_metric("locomo_f1"))
        self.assertFalse(unbounded_metric("token_f1"))
        self.assertFalse(unbounded_metric("judge_score"))
        self.assertTrue(unbounded_metric("judge_score_per_usd"))
        self.assertTrue(unbounded_metric("locomo_f1_per_second"))
        self.assertTrue(unbounded_metric("n_web_search"))
        self.assertTrue(unbounded_metric("n_mcp"))
        self.assertTrue(unbounded_metric("n_retrieval_calls"))
        self.assertFalse(unbounded_metric("used_non_workspace_tools"))
        self.assertFalse(unbounded_metric("memory_recall"))
        self.assertTrue(unbounded_metric("n"))
        self.assertTrue(unbounded_metric("answer_n_words"))
        self.assertFalse(unbounded_metric("harness_failed_rate"))




class TestYearFamilyCampaign(unittest.TestCase):


    def test_annotate_generation_uses_catalog_year(self):
        df = pd.DataFrame(
            {
                "reader_model": ["gpt-4o-mini", "gpt-5", "deepseek-chat"],
                "reader_generation": ["2024", "2025", "2025"],
            }
        )
        readers = annotate_generation(df, "reader")
        self.assertEqual(list(readers["generation"]), ["2024", "2025", "2025"])
        writers = annotate_generation(
            pd.DataFrame({"writer_model": ["gpt-5", "deepseek-chat", "gpt-4o-mini"]}),
            "writer",
        )
        self.assertEqual(list(writers["generation"]), ["2025", "2025", "2024"])

    def test_annotate_reader_stack_keeps_persist_off_codex_and_year_model_only(self):
        df = pd.DataFrame(
            [
                {
                    "generation": "2024",
                    "memory_method": "full_context",
                    "reader_model": "gpt-4o-mini",
                    "model_family": "OpenAI",
                    "thinking": None,
                    "agent_persist": None,
                },
                {
                    "generation": "2024",
                    "memory_method": "workspace_files",
                    "reader_model": "gpt-4o-mini",
                    "model_family": "OpenAI",
                    "thinking": None,
                    "agent_persist": False,
                },
                {
                    "generation": "2024",
                    "memory_method": "workspace_files",
                    "reader_model": "gpt-4o-mini",
                    "model_family": "OpenAI",
                    "thinking": None,
                    "agent_persist": True,
                },
                {
                    "generation": "2025",
                    "memory_method": "full_context",
                    "reader_model": "gpt-5",
                    "model_family": "OpenAI",
                    "thinking": "off",
                    "agent_persist": None,
                },
                {
                    "generation": "2025",
                    "memory_method": "full_context",
                    "reader_model": "gpt-5",
                    "model_family": "OpenAI",
                    "thinking": "on",
                    "agent_persist": None,
                },
                {
                    "generation": "2026",
                    "memory_method": "full_context",
                    "reader_model": "gpt-5.6-terra",
                    "model_family": "OpenAI",
                    "thinking": "off",
                    "agent_persist": None,
                },
                {
                    "generation": "2025",
                    "memory_method": "full_context",
                    "reader_model": "deepseek-chat",
                    "model_family": "DeepSeek",
                    "thinking": "off",
                    "agent_persist": None,
                },
            ]
        )
        out = annotate_reader_stack(df)
        self.assertEqual(
            list(out["reader_stack"]),
            [
                "2024 mini model-only",
                "2024 mini + Codex",
                None,
                "2025 GPT-5 model-only",
                None,
                "2026 Terra model-only",
                None,
            ],
        )

    def test_annotate_paper_compare_keeps_live_model_codex_and_year_bars(self):
        df = pd.DataFrame(
            [
                {
                    "memory_method": "full_context",
                    "result_source": "live",
                    "agent_persist": None,
                    "writer_harness": None,
                },
                {
                    "memory_method": "workspace_files",
                    "result_source": "live",
                    "agent_persist": False,
                    "writer_harness": None,
                },
                {
                    "memory_method": "workspace_files",
                    "result_source": "live",
                    "agent_persist": True,
                    "writer_harness": None,
                },
                {
                    "memory_method": "agent_codex_mem0_facts",
                    "result_source": "live",
                    "agent_persist": None,
                    "writer_harness": "codex",
                },
                {
                    "memory_method": "graph",
                    "result_source": "live",
                    "agent_persist": None,
                    "writer_harness": "codex",
                    "experiment_name": "locomo-openai-codex-poc-writers",
                },
                {
                    "memory_method": "graph",
                    "result_source": "live",
                    "agent_persist": None,
                    "writer_harness": "chat_completions",
                },
                {
                    "memory_method": "full_context",
                    "result_source": "paper",
                    "agent_persist": None,
                    "writer_harness": None,
                },
                {
                    "memory_method": "mem0",
                    "result_source": "local_clone",
                    "agent_persist": None,
                    "writer_harness": None,
                },
            ]
        )
        out = annotate_paper_compare(df)
        self.assertEqual(
            list(out["paper_method"]),
            [
                "full_context",
                "full_context",
                "workspace_files",
                "mem0",
                "mem0g",
                "graph",
                "full_context",
                "mem0",
            ],
        )
        self.assertEqual(
            list(out["compare_source"]),
            [
                "live model",
                COMPARE_SOURCE_CODEX,
                None,
                COMPARE_SOURCE_CODEX,
                COMPARE_SOURCE_CODEX,
                None,
                "paper",
                "local clone",
            ],
        )
        self.assertEqual(
            list(out["live_source"]),
            [
                "live model",
                COMPARE_SOURCE_CODEX,
                None,
                COMPARE_SOURCE_CODEX,
                COMPARE_SOURCE_CODEX,
                None,
                "paper",
                "live model",
            ],
        )
        years = annotate_paper_compare(
            pd.DataFrame(
                [
                    {
                        "memory_method": "full_context",
                        "result_source": "live",
                        "generation": "2025",
                        "reader_model": "gpt-5",
                        "model_family": "OpenAI",
                        "thinking": "off",
                    },
                    {
                        "memory_method": "full_context",
                        "result_source": "live",
                        "generation": "2025",
                        "reader_model": "gpt-5",
                        "model_family": "OpenAI",
                        "thinking": "on",
                    },
                    {
                        "memory_method": "full_context",
                        "result_source": "live",
                        "generation": "2026",
                        "reader_model": "gpt-5.6-terra",
                        "model_family": "OpenAI",
                        "thinking": "off",
                    },
                    {
                        "memory_method": "full_context",
                        "result_source": "live",
                        "generation": "2025",
                        "reader_model": "deepseek-chat",
                        "model_family": "DeepSeek",
                        "thinking": "off",
                    },
                ]
            )
        )
        self.assertEqual(
            list(years["compare_source"]),
            [
                "2025 GPT-5 model-only",
                None,
                "2026 Terra model-only",
                None,
            ],
        )

    def test_paper_compare_keeps_live_model_and_does_not_average_methods(self):
        spec = AnalysisSpec(
            id="methods_vs_paper_j",
            title="Memory methods J",
            group_by=("paper_method", "live_source"),
            metrics=("judge_score",),
            plots=(
                PlotSpec(kind="grouped_bar", x="paper_method", hue="live_source", y="judge_score"),
                PlotSpec(
                    kind="grouped_bar",
                    x="live_source",
                    y="judge_score",
                    split_by="paper_method",
                ),
            ),
            source="examples",
            where=(
                ("paper_method", ("full_context", "mem0")),
                ("live_source", ("paper", "live model", COMPARE_SOURCE_CODEX)),
            ),
            include_pins=True,
        )
        df = pd.DataFrame(
            [
                {
                    "memory_method": "full_context",
                    "result_source": "live",
                    "judge_score": 0.74,
                    "question_category": 1,
                    "generation": "2024",
                    "model_family": "OpenAI",
                },
                {
                    "memory_method": "workspace_files",
                    "result_source": "live",
                    "agent_persist": False,
                    "judge_score": 0.67,
                    "question_category": 1,
                    "generation": "2024",
                    "model_family": "OpenAI",
                },
                {
                    "memory_method": "agent_codex_mem0_facts",
                    "result_source": "live",
                    "writer_harness": "codex",
                    "judge_score": 0.50,
                    "question_category": 1,
                    "generation": "2024",
                    "model_family": "OpenAI",
                },
            ]
        )
        pins = [
            {
                "memory_method": "full_context",
                "result_source": "paper",
                "judge_score": 0.729,
                "n": 1540,
                "generation": "2024",
                "model_family": "OpenAI",
            },
            {
                "memory_method": "full_context",
                "result_source": "local_clone",
                "judge_score": 0.7468,
                "n": 1540,
                "generation": "2024",
                "model_family": "OpenAI",
            },
            {
                "memory_method": "mem0",
                "result_source": "paper",
                "judge_score": 0.6688,
                "n": 1540,
                "generation": "2024",
                "model_family": "OpenAI",
            },
            {
                "memory_method": "mem0",
                "result_source": "local_clone",
                "judge_score": 0.4838,
                "n": 1540,
                "generation": "2024",
                "model_family": "OpenAI",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp), pins=pins)
        self.assertIn("live_source", result.table.columns)
        self.assertIn("paper_method", result.table.columns)
        sources = {
            (str(row.paper_method), str(row.live_source)): (
                float(row.judge_score),
                int(row.n),
            )
            for row in result.table.itertuples()
        }
        self.assertEqual(sources[("full_context", "live model")][0], 0.74)
        self.assertEqual(sources[("full_context", "live model")][1], 1)
        self.assertAlmostEqual(sources[("full_context", COMPARE_SOURCE_CODEX)][0], 0.67)
        self.assertAlmostEqual(sources[("full_context", "paper")][0], 0.729)
        self.assertNotIn(("full_context", "local clone"), sources)
        self.assertAlmostEqual(sources[("mem0", "live model")][0], 0.4838)
        self.assertEqual(int(result.table["n"].max()), 1540)

    def test_mean_table_reannotates_missing_live_source_instead_of_averaging(self):
        df = pd.DataFrame(
            [
                {
                    "memory_method": "full_context",
                    "result_source": "live",
                    "judge_score": 0.74,
                    "generation": "2024",
                    "model_family": "OpenAI",
                },
                {
                    "memory_method": "workspace_files",
                    "result_source": "live",
                    "agent_persist": False,
                    "judge_score": 0.67,
                    "generation": "2024",
                    "model_family": "OpenAI",
                },
            ]
        )
        table = mean_table(
            df, ["generation", "model_family", "paper_method", "live_source"], ["judge_score"]
        )
        self.assertIn("paper_method", table.columns)
        self.assertIn("live_source", table.columns)
        self.assertEqual(len(table), 2)
        self.assertEqual(int(table["n"].max()), 1)

    def test_write_plot_skips_when_requested_x_is_missing(self):
        spec = AnalysisSpec(
            id="methods_vs_paper_j",
            title="Memory methods J",
            group_by=("generation", "model_family"),
            metrics=("judge_score",),
            plots=(
                PlotSpec(
                    kind="grouped_bar",
                    x="paper_method",
                    hue="live_source",
                    y="judge_score",
                ),
            ),
            source="examples",
        )
        table = pd.DataFrame(
            {
                "generation": ["2024"],
                "model_family": ["OpenAI"],
                "n": [9240],
                "judge_score": [0.5705],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            written = _write_plot(
                spec.plots[0], spec, table, Path(tmp) / "methods_vs_paper_j.png"
            )
        self.assertIsNone(written)

    def test_concat_without_live_source_still_groups_methods_after_prepare(self):
        spec = AnalysisSpec(
            id="methods_vs_paper_j",
            title="Memory methods J",
            group_by=("paper_method", "live_source"),
            metrics=("judge_score",),
            plots=(
                PlotSpec(
                    kind="grouped_bar",
                    x="paper_method",
                    hue="live_source",
                    y="judge_score",
                ),
            ),
            source="examples",
            where=(
                ("paper_method", ("full_context", "mem0")),
                ("live_source", ("paper", "live model", COMPARE_SOURCE_CODEX)),
            ),
            include_pins=True,
        )
        readers = pd.DataFrame(
            [
                {
                    "memory_method": "full_context",
                    "result_source": "live",
                    "judge_score": 0.74,
                    "question_category": 1,
                    "generation": "2024",
                    "model_family": "OpenAI",
                },
                {
                    "memory_method": "workspace_files",
                    "result_source": "live",
                    "agent_persist": False,
                    "judge_score": 0.67,
                    "question_category": 1,
                    "generation": "2024",
                    "model_family": "OpenAI",
                },
            ]
        )
        writers = pd.DataFrame(
            [
                {
                    "memory_method": "agent_codex_mem0_facts",
                    "result_source": "live",
                    "writer_harness": "codex",
                    "judge_score": 0.50,
                    "question_category": 1,
                    "generation": "2024",
                    "model_family": "OpenAI",
                }
            ]
        )
        pins = [
            {
                "memory_method": "full_context",
                "result_source": "paper",
                "judge_score": 0.729,
                "n": 1540,
                "generation": "2024",
                "model_family": "OpenAI",
            },
            {
                "memory_method": "mem0",
                "result_source": "local_clone",
                "judge_score": 0.4838,
                "n": 1540,
                "generation": "2024",
                "model_family": "OpenAI",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(
                spec, pd.concat([readers, writers], ignore_index=True), Path(tmp), pins=pins
            )
        self.assertIn("paper_method", result.table.columns)
        self.assertIn("live_source", result.table.columns)
        keys = {
            (str(row.paper_method), str(row.live_source))
            for row in result.table.itertuples()
        }
        self.assertIn(("full_context", "live model"), keys)
        self.assertIn(("full_context", COMPARE_SOURCE_CODEX), keys)
        self.assertIn(("mem0", COMPARE_SOURCE_CODEX), keys)
        self.assertEqual(int(result.table["n"].max()), 1540)

    def test_pins_are_not_averaged_into_live_rows_and_cat5_is_dropped(self):
        spec = AnalysisSpec(
            id="j_full_context",
            title="Full-context J",
            group_by=("generation", "model_family", "result_source"),
            metrics=("judge_score",),
            plots=(PlotSpec(kind="grouped_bar", x="generation", y="judge_score"),),
            source="examples",
            where=(("memory_method", ("full_context",)),),
            exclude_question_categories=(5,),
            include_pins=True,
        )
        df = pd.DataFrame(
            [
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "result_source": "live",
                    "memory_method": "full_context",
                    "question_category": 1,
                    "judge_score": 1.0,
                },
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "result_source": "live",
                    "memory_method": "full_context",
                    "question_category": 5,
                    "judge_score": 0.0,
                },
                {
                    "generation": "2025",
                    "model_family": "OpenAI",
                    "result_source": "live",
                    "memory_method": "rag",
                    "question_category": 1,
                    "judge_score": 0.0,
                },
            ]
        )
        pins = [
            {
                "generation": "2024",
                "model_family": "OpenAI",
                "memory_method": "full_context",
                "result_source": "paper",
                "judge_score": 0.729,
                "n": 1540,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = render_analysis(spec, df, Path(tmp), pins=pins)
        self.assertEqual(len(result.table), 2)
        live = result.table[result.table["result_source"] == "live"].iloc[0]
        paper = result.table[result.table["result_source"] == "paper"].iloc[0]
        self.assertAlmostEqual(float(live["judge_score"]), 1.0)
        self.assertEqual(int(live["n"]), 1)
        self.assertAlmostEqual(float(paper["judge_score"]), 0.729)
        self.assertEqual(int(paper["n"]), 1540)
        self.assertEqual(list(result.table["generation"]), ["2024", "2025"])


class TestOpenAIAgentCampaigns(unittest.TestCase):
    def test_poc_campaign_has_audit_metrics_and_paper_pins(self):
        cfg = load_campaign_yaml(ROOT / "configs" / "analysis" / "campaign_openai_codex_poc.yaml")
        self.assertEqual(cfg.id, "openai_codex_poc")
        self.assertGreaterEqual(len(cfg.pins), 8)
        sources = {str(row.get("result_source")) for row in cfg.pins}
        self.assertEqual(sources, {"paper", "local_clone"})
        methods = {str(row.get("memory_method")) for row in cfg.pins}
        self.assertTrue({"full_context", "rag", "openai_memory", "mem0", "mem0g", "session_summaries"} <= methods)
        summary_pin = next(
            row
            for row in cfg.pins
            if row.get("memory_method") == "session_summaries" and row.get("result_source") == "paper"
        )
        self.assertIsNone(summary_pin.get("judge_score"))
        self.assertAlmostEqual(float(summary_pin["locomo_f1"]), LOCOMO_2024_SUMMARY_RAG_F1)
        self.assertEqual(int(summary_pin["n"]), LOCOMO_2024_SUMMARY_RAG_N)
        ids = [spec.id for spec in cfg.campaign_analyses]
        self.assertIn("reader_audit", ids)
        self.assertIn("reader_failure_modes", ids)
        self.assertIn("reader_j_vs_f1", ids)
        self.assertIn("reader_recall_bins", ids)
        self.assertIn("reader_failure_modes_judge", ids)
        self.assertIn("reader_hop_to_evidence", ids)
        self.assertIn("reader_qidx_by_category", ids)
        hop = next(spec for spec in cfg.campaign_analyses if spec.id == "reader_hop_to_evidence")
        self.assertIn("hop_bin", hop.group_by)
        self.assertIn("reader_vs_paper_j", ids)
        self.assertIn("methods_vs_paper_j", ids)
        self.assertIn("reader_stack_vs_paper_j", ids)
        vs_paper = next(spec for spec in cfg.campaign_analyses if spec.id == "reader_vs_paper_j")
        self.assertEqual(vs_paper.source, "examples")
        self.assertEqual(set(vs_paper.experiments), {"readers", "year_2025", "year_2026"})
        self.assertIn("compare_source", vs_paper.group_by)
        self.assertEqual(vs_paper.plots[0].x, "compare_source")
        methods = next(spec for spec in cfg.campaign_analyses if spec.id == "methods_vs_paper_j")
        self.assertEqual(set(methods.experiments), {"readers", "writers"})
        self.assertIn("live_source", methods.group_by)
        self.assertIn("locomo_f1", methods.metrics)
        self.assertTrue(any(p.split_by == "paper_method" for p in methods.plots))
        self.assertTrue(any(p.y == "locomo_f1" for p in methods.plots))
        stack = next(spec for spec in cfg.campaign_analyses if spec.id == "reader_stack_vs_paper_j")
        self.assertEqual(set(stack.experiments), {"readers", "year_2025", "year_2026"})
        self.assertIn("reader_stack", stack.group_by)
        self.assertTrue(stack.include_pins)
        audit = next(spec for spec in cfg.campaign_analyses if spec.id == "reader_audit")
        self.assertIn("n_web_search", audit.metrics)
        self.assertIn("n_mcp", audit.metrics)
        self.assertIn("harness_failed_rate", audit.metrics)
        self.assertNotIn("comparison_status", audit.group_by)
        controls = next(spec for spec in cfg.campaign_analyses if spec.id == "reader_controls")
        self.assertNotIn("comparison_status", controls.group_by)
        modes = next(spec for spec in cfg.campaign_analyses if spec.id == "reader_failure_modes")
        self.assertEqual(modes.source, "examples")
        self.assertIn("failure_mode", modes.group_by)
        self.assertEqual(set(cfg.experiments), {"readers", "writers", "year_2025", "year_2026"})
        insight_ids = [item.id for item in cfg.insights]
        self.assertIn("pin_vs_poc_full_context", insight_ids)
        self.assertIn("workspace_j_minus_f1", insight_ids)
        pin = next(item for item in cfg.insights if item.id == "pin_vs_poc_full_context")
        self.assertEqual(pin.live_generation, "2024")
        takeaway_ids = [item.id for item in cfg.takeaways]
        self.assertIn("harness_reader_vs_chat_reader", takeaway_ids)
        self.assertIn("harness_writer_vs_chat_reader", takeaway_ids)
        self.assertIn("layers_are_different_claims", takeaway_ids)

    def test_agents_campaign_has_year_axis_audit_and_chat_writers(self):
        cfg = load_campaign_yaml(ROOT / "configs" / "analysis" / "campaign_openai_agents.yaml")
        self.assertEqual(cfg.id, "openai_agents")
        self.assertEqual(
            set(cfg.experiments),
            {"readers", "writers", "codex_readers", "codex_writers", "codex_end_to_end"},
        )
        ids = [spec.id for spec in cfg.campaign_analyses]
        self.assertIn("chat_readers_vs_paper_j", ids)
        self.assertIn("chat_readers_year", ids)
        self.assertIn("sandwich_writers_year", ids)
        self.assertIn("codex_readers_audit", ids)
        self.assertIn("codex_readers_failure_modes", ids)
        self.assertIn("codex_readers_j_vs_f1", ids)
        self.assertIn("codex_readers_recall_bins", ids)
        self.assertIn("codex_end_to_end_audit", ids)
        self.assertIn("codex_end_to_end_failure_modes", ids)
        self.assertIn("codex_end_to_end_hop", ids)
        self.assertIn("codex_end_to_end_qidx", ids)
        year = next(spec for spec in cfg.campaign_analyses if spec.id == "codex_readers_year")
        self.assertIn("generation", year.group_by)
        audit = next(spec for spec in cfg.campaign_analyses if spec.id == "codex_readers_audit")
        self.assertNotIn("comparison_status", audit.group_by)
        self.assertIn("harness_failed_rate", audit.metrics)
        insight_gens = {item.live_generation for item in cfg.insights if item.kind == "pin_gaps"}
        self.assertEqual(insight_gens, {"2024", "2025", "2026"})

    def test_persist_memory_campaign_has_notes_only_vs_summaries_takeaway(self):
        cfg = load_campaign_yaml(
            ROOT / "configs" / "analysis" / "campaign_openai_codex_persist_memory.yaml"
        )
        self.assertEqual(cfg.id, "openai_codex_persist_memory")
        self.assertEqual(set(cfg.experiments), {"persist", "summaries"})
        ids = [spec.id for spec in cfg.campaign_analyses]
        self.assertIn("persist_hop_to_evidence", ids)
        self.assertIn("persist_qidx_by_category", ids)
        self.assertIn("persist_notes_size", ids)
        takeaway_ids = [item.id for item in cfg.takeaways]
        self.assertIn("notes_only_is_the_memory_method", takeaway_ids)
        self.assertIn("notes_only_vs_summaries", takeaway_ids)
        hop = next(spec for spec in cfg.campaign_analyses if spec.id == "persist_hop_to_evidence")
        self.assertIn("hop_bin", hop.group_by)
        self.assertIn("hop_to_evidence", next(
            spec for spec in cfg.campaign_analyses if spec.id == "persist_audit"
        ).metrics)

    def test_mini_vs_codex_writers_campaign_groups_by_writer_harness(self):
        cfg = load_campaign_yaml(
            ROOT / "configs" / "analysis" / "campaign_openai_mini_vs_codex_writers.yaml"
        )
        self.assertEqual(cfg.id, "openai_mini_vs_codex_writers")
        self.assertEqual(set(cfg.experiments), {"readers", "chat", "codex"})
        writers = next(spec for spec in cfg.campaign_analyses if spec.id == "structured_writers")
        self.assertEqual(set(writers.experiments), {"chat", "codex"})
        self.assertIn("writer_harness", writers.group_by)
        self.assertIn("memory_method", writers.group_by)
        self.assertTrue(
            any(p.x == "memory_method" and p.hue == "writer_harness" for p in writers.plots)
        )
        takeaway_ids = [item.id for item in cfg.takeaways]
        self.assertIn("summaries_chat_vs_codex", takeaway_ids)
        self.assertIn("graph_chat_vs_codex", takeaway_ids)
        ceiling = next(
            spec for spec in cfg.campaign_analyses if spec.id == "methods_vs_full_context"
        )
        self.assertEqual(set(ceiling.experiments), {"readers", "chat", "codex"})
        self.assertEqual(ceiling.group_by, ("memory_lane", "system_harness"))
        self.assertTrue(
            any(p.x == "memory_lane" and p.hue == "system_harness" for p in ceiling.plots)
        )
        self.assertIn("methods_vs_full_context_ceiling", takeaway_ids)

    def test_model_harness_gaps_campaign_has_category_j_charts(self):
        cfg = load_campaign_yaml(
            ROOT / "configs" / "analysis" / "campaign_openai_model_harness_gaps.yaml"
        )
        self.assertEqual(cfg.id, "openai_model_harness_gaps")
        fc = next(spec for spec in cfg.campaign_analyses if spec.id == "fc_category_j")
        summaries = next(
            spec for spec in cfg.campaign_analyses if spec.id == "summaries_category_j"
        )
        year = next(spec for spec in cfg.campaign_analyses if spec.id == "year_fc_category_j")
        self.assertEqual(fc.plots[0].x, "question_category")
        self.assertEqual(fc.plots[0].hue, "mini_side")
        self.assertEqual(fc.plots[0].y, "judge_score")
        self.assertEqual(summaries.plots[0].hue, "mini_side")
        self.assertEqual(year.plots[0].hue, "gap_stack")
        self.assertIn("question_category", year.group_by)
        adversarial = next(
            spec for spec in cfg.campaign_analyses if spec.id == "year_fc_adversarial"
        )
        self.assertEqual(adversarial.group_by, ("gap_stack",))
        self.assertEqual(adversarial.exclude_question_categories, (1, 2, 3, 4))
        self.assertEqual(adversarial.metrics, ("locomo_f1", "judge_score"))
        self.assertEqual(adversarial.plots[0].y, "locomo_f1")
        self.assertEqual(adversarial.plots[0].x, "gap_stack")
        kinds = {item.kind for item in cfg.insights}
        self.assertEqual(kinds, {"series_gaps"})

    def test_notebook_posttest_accepts_run_report_list_without_using_it_as_id(self):
        cfg = load_campaign_yaml(
            ROOT / "configs" / "analysis" / "campaign_openai_codex_persist_memory.yaml"
        )
        fake = ReportResult(
            scope="campaign",
            out_dir=ROOT / "experiments" / "_campaign" / "openai_codex_persist_memory",
            results=[],
            missing_packs=["locomo-openai-codex-persist-memory"],
        )
        table = notebook_posttest(cfg, [fake], root=ROOT)
        self.assertFalse(table.empty)
        self.assertIn("pack", " ".join(table["check"].astype(str)))


if __name__ == "__main__":
    unittest.main()
