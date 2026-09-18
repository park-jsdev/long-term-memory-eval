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

from src.memorybench.analysis.load_campaign import (
    AnalysisSpec,
    PlotSpec,
    load_campaign_yaml,
)
from src.memorybench.analysis.plots import _place_legend_outside
from src.memorybench.analysis.report import (
    _plot_title,
    annotate_model_family,
    mean_table,
    render_analysis,
    render_campaign,
    render_experiment,
    run_report,
)
from scripts.analysis.campaign_tables import (
    annotate_generation,
    annotate_mem0_latency,
    annotate_thinking,
    join_search_latency,
)

CAMPAIGN = ROOT / "configs" / "analysis" / "campaign_2025_live.yaml"
VALID_PLOT_KINDS = {"bar", "grouped_bar", "metrics_grouped_bar"}
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


class TestLoadCampaignYaml(unittest.TestCase):
    def test_2025_live_has_campaign_and_three_experiments(self):
        cfg = load_campaign_yaml(CAMPAIGN)
        self.assertEqual(cfg.id, "2025_live")
        self.assertEqual(set(cfg.experiments), {"smoke", "baseline", "writers"})
        self.assertTrue(cfg.campaign_analyses)
        self.assertEqual(cfg.experiments["smoke"].family_from, "reader")
        self.assertEqual(cfg.experiments["writers"].family_from, "writer")
        reader_family = [a for a in cfg.campaign_analyses if a.id == "reader_family_metrics"]
        self.assertEqual(reader_family[0].experiments, ("smoke", "baseline"))
        cat = [a for a in cfg.campaign_analyses if a.id == "reader_family_by_category"][0]
        self.assertEqual(cat.plots[0].kind, "grouped_bar")
        self.assertEqual(cat.plots[0].x, "question_category")
        self.assertEqual(cat.plots[0].hue, "model_family")
        self.assertEqual(cat.plots[0].y, "locomo_f1")
        writer_cat = [
            a for a in cfg.campaign_analyses if a.id == "writer_family_by_category"
        ][0]
        self.assertEqual(writer_cat.experiments, ("writers",))
        self.assertEqual(writer_cat.plots[0].x, "question_category")
        self.assertEqual(writer_cat.plots[0].hue, "model_family")
        self.assertEqual(writer_cat.plots[0].y, "locomo_f1")

    def test_2025_openai_deepseek_reuses_recipes_with_separate_packs(self):
        cfg = load_campaign_yaml(
            ROOT / "configs" / "analysis" / "campaign_2025_openai_deepseek.yaml"
        )
        self.assertEqual(cfg.id, "2025_openai_deepseek")
        self.assertEqual(set(cfg.experiments), {"smoke", "baseline", "writers"})
        self.assertEqual(
            cfg.experiments["smoke"].name,
            "locomo-2025-readers-openai-deepseek-smoke",
        )
        self.assertEqual(
            cfg.experiments["baseline"].name,
            "locomo-2025-readers-openai-deepseek",
        )
        self.assertEqual(
            cfg.experiments["writers"].name,
            "locomo-mem0-reader-2025-writers-openai-deepseek",
        )
        live = load_campaign_yaml(CAMPAIGN)
        self.assertIn("reader_thinking_tokens", [a.id for a in cfg.campaign_analyses])
        self.assertIn("writer_thinking_tokens", [a.id for a in cfg.campaign_analyses])
        writer_cat = [
            a for a in cfg.campaign_analyses if a.id == "writer_family_by_category"
        ][0]
        self.assertEqual(writer_cat.experiments, ("writers",))
        self.assertIn("question_category", writer_cat.group_by)
        self.assertIn("thinking", writer_cat.group_by)
        self.assertEqual(writer_cat.plots[0].x, "question_category")
        self.assertEqual(writer_cat.plots[0].hue, "thinking")
        self.assertEqual(writer_cat.plots[0].y, "locomo_f1")
        latency = [a for a in cfg.campaign_analyses if a.id == "reader_latency_mem0"][0]
        self.assertIn("memory_method", latency.group_by)
        self.assertEqual(
            [p.y for p in latency.plots],
            [
                "total_latency_seconds_p50",
                "total_latency_seconds_p95",
                "search_latency_seconds_p50",
                "search_latency_seconds_p95",
            ],
        )
        self.assertNotEqual(
            [a.id for a in cfg.campaign_analyses],
            [a.id for a in live.campaign_analyses],
        )
        self.assertNotEqual(cfg.experiments["smoke"].name, live.experiments["smoke"].name)
        baseline_cat = [
            a for a in cfg.experiments["baseline"].analyses if a.id == "by_category"
        ][0]
        self.assertIn("reader_display_name", baseline_cat.group_by)
        self.assertIn("memory_method", baseline_cat.group_by)
        self.assertIn("thinking", baseline_cat.group_by)
        tokens = [
            a for a in cfg.experiments["baseline"].analyses if a.id == "thinking_tokens"
        ][0]
        self.assertIn("agent_reasoning_tokens", tokens.metrics)
        writer_tokens = [
            a for a in cfg.experiments["writers"].analyses if a.id == "thinking_tokens"
        ][0]
        self.assertEqual(writer_tokens.source, "runs")
        self.assertIn("teacher_reasoning_tokens", writer_tokens.metrics)
        within = [
            a for a in cfg.experiments["baseline"].analyses if a.id == "thinking_within_reader"
        ][0]
        self.assertEqual(within.group_by, ("reader_display_name", "thinking"))
        self.assertEqual(within.plots[0].x, "reader_display_name")
        self.assertEqual(within.plots[0].hue, "thinking")
        writer_within = [
            a for a in cfg.experiments["writers"].analyses if a.id == "thinking_within_writer"
        ][0]
        self.assertEqual(writer_within.plots[0].x, "writer_model")
        self.assertEqual(writer_within.plots[0].hue, "thinking")
        self.assertEqual(baseline_cat.plots[0].hue, "thinking")

    def test_2026_openai_deepseek_reuses_recipes_with_separate_packs(self):
        cfg = load_campaign_yaml(
            ROOT / "configs" / "analysis" / "campaign_2026_openai_deepseek.yaml"
        )
        self.assertEqual(cfg.id, "2026_openai_deepseek")
        self.assertEqual(set(cfg.experiments), {"smoke", "baseline", "writers"})
        self.assertEqual(
            cfg.experiments["smoke"].name,
            "locomo-2026-readers-openai-deepseek-smoke",
        )
        self.assertEqual(
            cfg.experiments["baseline"].name,
            "locomo-2026-readers-openai-deepseek",
        )
        self.assertEqual(
            cfg.experiments["writers"].name,
            "locomo-mem0-reader-2026-writers-openai-deepseek",
        )
        live = load_campaign_yaml(CAMPAIGN)
        self.assertIn("reader_thinking_tokens", [a.id for a in cfg.campaign_analyses])
        self.assertIn("writer_thinking_tokens", [a.id for a in cfg.campaign_analyses])
        writer_cat = [
            a for a in cfg.campaign_analyses if a.id == "writer_family_by_category"
        ][0]
        self.assertEqual(writer_cat.experiments, ("writers",))
        self.assertIn("question_category", writer_cat.group_by)
        self.assertIn("thinking", writer_cat.group_by)
        self.assertEqual(writer_cat.plots[0].x, "question_category")
        self.assertEqual(writer_cat.plots[0].hue, "thinking")
        self.assertEqual(writer_cat.plots[0].y, "locomo_f1")
        latency = [a for a in cfg.campaign_analyses if a.id == "reader_latency_mem0"][0]
        self.assertIn("memory_method", latency.group_by)
        self.assertEqual(
            [p.y for p in latency.plots],
            [
                "total_latency_seconds_p50",
                "total_latency_seconds_p95",
                "search_latency_seconds_p50",
                "search_latency_seconds_p95",
            ],
        )
        self.assertNotEqual(
            [a.id for a in cfg.campaign_analyses],
            [a.id for a in live.campaign_analyses],
        )
        self.assertNotEqual(cfg.experiments["smoke"].name, live.experiments["smoke"].name)
        baseline_cat = [
            a for a in cfg.experiments["baseline"].analyses if a.id == "by_category"
        ][0]
        self.assertIn("reader_display_name", baseline_cat.group_by)
        self.assertIn("memory_method", baseline_cat.group_by)
        self.assertIn("thinking", baseline_cat.group_by)


class TestGroupedBarKeepsReaderAndMemory(unittest.TestCase):
    def test_baseline_by_category_groups_reader_memory_and_category(self):
        cfg = load_campaign_yaml(CAMPAIGN)
        spec = [a for a in cfg.experiments["baseline"].analyses if a.id == "by_category"][0]
        self.assertEqual(
            spec.group_by,
            ("reader_display_name", "memory_method", "question_category"),
        )
        self.assertIsNone(spec.plots[0].hue)

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


class TestAnalysisNotebookContract(unittest.TestCase):
    def test_configured_experiment_notebooks_are_thin_yaml_wrappers(self):
        cfg = load_campaign_yaml(CAMPAIGN)
        for ref in cfg.experiments.values():
            with self.subTest(experiment=ref.id):
                self.assertIsNotNone(ref.notebook)
                notebook = ROOT / str(ref.notebook)
                self.assertTrue(notebook.is_file())
                code = _notebook_code(notebook)
                self.assertIn("campaign_2025_live.yaml", code)
                self.assertIn(f'render_experiment(camp, "{ref.id}"', code)
                self.assertIn("notebook_pretest(camp,", code)
                self.assertIn("notebook_posttest(camp,", code)
                self.assertNotIn("matplotlib", code)
                self.assertNotIn("groupby(", code)

    def test_campaign_notebook_is_a_thin_yaml_wrapper(self):
        notebook = ROOT / "notebooks" / "06_2025_live_campaign_analysis.ipynb"
        code = _notebook_code(notebook)
        self.assertIn("campaign_2025_live.yaml", code)
        self.assertIn("render_campaign(camp, root=ROOT)", code)
        self.assertIn("notebook_pretest(camp, root=ROOT)", code)
        self.assertIn("notebook_posttest(camp,", code)
        self.assertNotIn("matplotlib", code)
        self.assertNotIn("groupby(", code)

    def test_openai_deepseek_notebooks_state_thinking_within_family(self):
        names = (
            "07_2025_readers_openai_deepseek_smoke_analysis.ipynb",
            "08_2025_readers_openai_deepseek_analysis.ipynb",
            "09_mem0_reader_2025_writers_openai_deepseek_analysis.ipynb",
            "10_2025_openai_deepseek_campaign_analysis.ipynb",
            "11_2026_readers_openai_deepseek_smoke_analysis.ipynb",
            "12_2026_readers_openai_deepseek_analysis.ipynb",
            "13_mem0_reader_2026_writers_openai_deepseek_analysis.ipynb",
            "14_2026_openai_deepseek_campaign_analysis.ipynb",
        )
        for name in names:
            notebook = ROOT / "notebooks" / name
            with self.subTest(notebook=name):
                self.assertTrue(notebook.is_file())
                payload = json.loads(notebook.read_text(encoding="utf-8"))
                md = "\n".join(
                    "".join(cell.get("source") or [])
                    for cell in payload.get("cells") or []
                    if cell.get("cell_type") == "markdown"
                )
                self.assertIn("within", md.lower())
                self.assertIn("hue=thinking", md.lower())


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


class TestRenderSkipsMissingPack(unittest.TestCase):
    def test_render_experiment_records_missing_pack_without_raising(self):
        cfg = load_campaign_yaml(CAMPAIGN)
        with tempfile.TemporaryDirectory() as tmp:
            report = render_experiment(cfg, "baseline", root=Path(tmp))
            self.assertEqual(report.missing_packs, ["locomo-2025-readers-full-context"])
            self.assertEqual(report.results, [])


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
        from src.memorybench.analysis.plots import _pyplot

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
        from src.memorybench.analysis.plots import unbounded_metric

        self.assertTrue(unbounded_metric("agent_reasoning_tokens"))
        self.assertTrue(unbounded_metric("total_latency_seconds_p95"))
        self.assertTrue(unbounded_metric("search_latency_seconds"))
        self.assertFalse(unbounded_metric("locomo_f1"))
        self.assertFalse(unbounded_metric("token_f1"))
        self.assertFalse(unbounded_metric("judge_score"))


class TestSmokeReportWhenPackPresent(unittest.TestCase):
    def test_smoke_report_writes_family_table_when_aggregate_exists(self):
        pack = ROOT / "experiments" / "locomo-2025-readers-full-context-smoke" / "aggregate"
        if not (pack / "examples.parquet").is_file():
            self.skipTest("smoke aggregate not checked out")
        reports = run_report(CAMPAIGN, experiment_id="smoke", root=ROOT)
        self.assertEqual(len(reports), 1)
        ids = [item.spec.id for item in reports[0].results if not item.skipped]
        self.assertIn("by_reader", ids)
        self.assertIn("reader_family_metrics", ids)
        family = next(r for r in reports[0].results if r.spec.id == "reader_family_metrics")
        self.assertGreaterEqual(len(family.table), 3)
        self.assertTrue((reports[0].out_dir / "tables" / "by_reader.csv").is_file())


class TestYearFamilyCampaign(unittest.TestCase):
    YAML = ROOT / "configs" / "analysis" / "campaign_year_family.yaml"

    def test_year_family_yaml_has_pins_and_2026_axis_note(self):
        cfg = load_campaign_yaml(self.YAML)
        self.assertEqual(cfg.id, "year_family")
        self.assertEqual(set(cfg.experiments), {"readers", "writers", "readers_2026", "writers_2026"})
        self.assertGreaterEqual(len(cfg.pins), 7)
        generations = {str(row.get("generation")) for row in cfg.pins}
        self.assertEqual(generations, {"2024"})
        sources = {str(row.get("result_source")) for row in cfg.pins}
        self.assertEqual(sources, {"paper", "local_clone"})
        ids = [spec.id for spec in cfg.campaign_analyses]
        self.assertIn("j_full_context_by_year_family", ids)
        self.assertIn("j_rag_by_year_family", ids)
        self.assertIn("j_memory_write_by_year_family", ids)
        self.assertIn("year_family_by_thinking", ids)
        self.assertIn("thinking_tokens_by_year_family", ids)
        self.assertIn("writer_thinking_tokens_by_year_family", ids)
        fc = [a for a in cfg.campaign_analyses if a.id == "j_full_context_by_year_family"][0]
        self.assertTrue(fc.include_pins)
        self.assertEqual(fc.exclude_question_categories, (5,))
        self.assertEqual(fc.where, (("memory_method", ("full_context",)),))

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


if __name__ == "__main__":
    unittest.main()
