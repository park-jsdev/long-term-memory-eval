"""Campaign cost estimates vs priced pack actuals (no API)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.locomo_eval.pricing import estimate_usd, load_pricing
from src.experiment_runner.analysis.cost import render_cost
from src.experiment_runner.analysis.load_campaign import load_campaign_yaml
from src.experiment_runner.analysis.notebook_protocol import _posttest_rows
from src.experiment_runner.analysis.report import ReportResult


def _write_cell(
    pack: Path,
    run_id: str,
    *,
    memory: str,
    reader: str,
    teacher: str | None,
    prompt: int,
    completion: int,
) -> None:
    cell = pack / "aggregate" / "by_run" / run_id
    cell.mkdir(parents=True)
    n_pred = 1986 if prompt > 1000 else 5
    (cell / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "memory_type": memory,
                "reader_model": reader,
                "teacher_model": teacher,
                "n_predictions": n_pred,
            }
        ),
        encoding="utf-8",
    )
    reader_rec = {
        "n_calls": n_pred,
        "prompt_tokens": prompt if teacher is None else 100,
        "completion_tokens": completion if teacher is None else 10,
    }
    cost = {
        "reader": {
            **reader_rec,
            "by_model": {reader: dict(reader_rec)},
        },
        "teacher": {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}},
        "other": {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}},
    }
    if teacher:
        teacher_rec = {
            "n_calls": 272,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        }
        cost["teacher"] = {**teacher_rec, "by_model": {teacher: dict(teacher_rec)}}
    (cell / "cost.json").write_text(json.dumps(cost), encoding="utf-8")


class TestPricingYaml(unittest.TestCase):
    def test_terra_and_fable_are_priced(self):
        table = load_pricing()
        self.assertAlmostEqual(estimate_usd("gpt-5.6-terra", 1_000_000, 0, table=table), 2.0)
        self.assertAlmostEqual(estimate_usd("claude-fable-5-1", 1_000_000, 0, table=table), 10.0)
        self.assertAlmostEqual(
            estimate_usd("deepseek-v4-flash", 1_000_000, 0, scenario="peak", table=table),
            0.30,
        )


class TestCollectAndEstimate(unittest.TestCase):
    def test_map_gpt5_volume_to_terra_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vol = root / "experiments" / "prior"
            _write_cell(
                vol,
                "cell-fc",
                memory="full_context",
                reader="gpt-5",
                teacher=None,
                prompt=55_304_493,
                completion=27_920,
            )
            cfg_path = root / "campaign.yaml"
            pricing = (ROOT / "configs" / "models" / "pricing.yaml").as_posix()
            cfg_path.write_text(
                f"""
campaign:
  id: toy
cost:
  pricing: {pricing}
  volume_from:
    baseline: experiments/prior
  map_models:
    gpt-5: gpt-5.6-terra
experiments:
  baseline:
    name: toy-baseline
    pack: experiments/missing
    pretest:
      n_cells: 1
      n_questions: 1986
      scientific_claim: true
""".strip()
                + "\n",
                encoding="utf-8",
            )
            cfg = load_campaign_yaml(cfg_path)
            report = render_cost(cfg, root / "out", root=root, experiment_id="baseline")
            self.assertIsNotNone(report)
            cell = report.by_cell
            self.assertEqual(set(cell["model"]), {"gpt-5.6-terra"})
            usd = float(cell["usd_expected"].sum())
            self.assertAlmostEqual(usd, 55.304493 * 2 + 0.027920 * 12, places=2)

    def test_parked_fable_is_not_in_launched_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vol = root / "experiments" / "prior"
            _write_cell(
                vol,
                "cell-fc",
                memory="full_context",
                reader="gpt-5",
                teacher=None,
                prompt=1_000_000,
                completion=0,
            )
            cfg_path = root / "campaign.yaml"
            pricing = (ROOT / "configs" / "models" / "pricing.yaml").as_posix()
            cfg_path.write_text(
                f"""
campaign:
  id: toy
cost:
  pricing: {pricing}
  volume_from:
    baseline: experiments/prior
  map_models:
    gpt-5: gpt-5.6-terra
  parked:
    - id: claude_fable
      display_name: Claude Fable 5.1
      api_model_id: claude-fable-5-1
      copy_from: gpt-5.6-terra
experiments:
  baseline:
    name: toy-baseline
    pack: experiments/missing
    pretest:
      n_cells: 1
      n_questions: 1986
""".strip()
                + "\n",
                encoding="utf-8",
            )
            cfg = load_campaign_yaml(cfg_path)
            report = render_cost(cfg, root / "out", root=root, experiment_id="baseline")
            launched = float(report.by_cell["usd_expected"].sum())
            parked = float(report.parked["usd"].sum())
            self.assertAlmostEqual(launched, 2.0, places=4)
            self.assertAlmostEqual(parked, 10.0, places=4)
            self.assertTrue((report.parked["launched"] == False).all())

    def test_smoke_scales_five_over_full_locomo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vol = root / "experiments" / "prior"
            _write_cell(
                vol,
                "cell-fc",
                memory="full_context",
                reader="gpt-5",
                teacher=None,
                prompt=1_986_000,
                completion=0,
            )
            cfg_path = root / "campaign.yaml"
            pricing = (ROOT / "configs" / "models" / "pricing.yaml").as_posix()
            cfg_path.write_text(
                f"""
campaign:
  id: toy
cost:
  pricing: {pricing}
  volume_from:
    baseline: experiments/prior
  scale:
    smoke:
      from_experiment: baseline
      memory_methods: [full_context]
experiments:
  smoke:
    name: toy-smoke
    pack: experiments/missing-smoke
    pretest:
      n_cells: 2
      n_questions: 5
      scientific_claim: false
  baseline:
    name: toy-baseline
    pack: experiments/missing
    pretest:
      n_cells: 4
      n_questions: 1986
""".strip()
                + "\n",
                encoding="utf-8",
            )
            cfg = load_campaign_yaml(cfg_path)
            report = render_cost(cfg, root / "out", root=root, experiment_id="smoke")
            reader = report.by_cell[report.by_cell["role"] == "reader"].iloc[0]
            self.assertEqual(int(reader["n_questions"]), 5)
            self.assertEqual(int(reader["prompt_tokens"]), 5000)
            actual = report.by_stage.loc[
                report.by_stage["experiment"] == "smoke", "usd_actual"
            ]
            self.assertTrue(actual.isna().all())




class TestPosttestChecks(unittest.TestCase):
    def test_missing_pack_is_skip_not_fail(self):
        cfg = load_campaign_yaml(
            ROOT / "configs" / "analysis" / "campaign_openai_codex_poc.yaml"
        )
        report = ReportResult(
            scope="experiment:readers",
            out_dir=ROOT,
            results=[],
            missing_packs=[cfg.experiments["readers"].name],
        )
        rows = _posttest_rows(cfg, "readers", report, ROOT)
        results = {r["check"]: r["result"] for r in rows}
        self.assertEqual(results["readers.pack"], "SKIP")
        self.assertEqual(results["readers.scientific_claim"], "SKIP")
