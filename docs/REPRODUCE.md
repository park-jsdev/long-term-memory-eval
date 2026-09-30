# Reproduction runbook

**Audience:** a researcher or engineer who has this repository and wants to regenerate the results.
**Reading time:** 10 minutes. **Wall clock:** ~20 minutes offline, hours-to-days for the paid campaign.

You can stop after any stage. Stages 0–3 cost nothing and prove the pipeline works. Stage 4 onward
spends real API money.

| Stage | What you get | Cost | Needs keys |
|---|---|---|---|
| 0 | Working environment and dataset | none | no |
| 1 | Green deterministic verifiers | none | no |
| 2 | A complete mock run pack you can inspect | none | no |
| 3 | A mock multi-cell experiment, aggregated and reported | none | no |
| 4 | One live cell, verified cheaply | cents | yes |
| 5 | The three-experiment 2025 campaign | substantial | yes |
| 6 | Campaign tables, plots, and notebook review | none | no |
| 7 | Audit trail walked back from a published number | none | no |

Throughout: never commit `.env` or any key. Do not claim Mem0 paper Table 1–2 `J` from these OSS
clones — the judged `J` here is a protocol clone, not a Platform reproduction.

---

## Stage 0 — Environment and dataset

```bash
conda create -n distillation python=3.11 -y
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py
```

`fetch_locomo.py` downloads the pinned LoCoMo file to `data/raw/locomo10.json`. It is gitignored.

**Checkpoint.** The file exists and is a few megabytes. The pin recorded in
`configs/data/locomo10.yaml` is commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.

For live stages, create your key file:

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

Set whichever of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY` you need. The loader
reads this file automatically at run start.

---

## Stage 1 — Prove the verifiers are green

Every test is mock-only and offline. No key, no network, no cost.

```bash
python -m pytest tests/ -q
```

If an environment has `pytest-asyncio` installed, `pytest.ini` already disables it; these are plain
`unittest` classes. If you prefer `unittest`, use the **file-path** form so a site-packages module
named `tests` cannot shadow the folder:

```bash
python -m unittest tests/test_regressions.py tests/test_claim_audit.py
```

**Checkpoint.** All tests pass or skip. Skips are expected only for a missing dataset file, missing
`matplotlib`, missing `pyarrow`, or a not-yet-pulled aggregate pack.

What you have just locked: scoring, indexing, pooling and fusion behavior; the judge protocol
including the category-5 exclusion; the sandwich contracts (one YAML to one pack, replace-not-append,
gold answers never entering a prompt); YAML merge semantics; hashed run ids; and byte-identical
analysis output under row reordering.

---

## Stage 2 — One mock run, end to end

This exercises the whole read path with a fake model, and writes a real audit pack.

```bash
python -m src.locomo_eval.run --config configs/presets/mem0_baseline.yaml \
  --reader mock --max-questions 5 --run-id smoke_mock
```

**Checkpoint.** Open `experiments/smoke_mock/` and confirm you can answer these questions from
files alone:

| Question | File |
|---|---|
| What was asked and answered? | `predictions.csv` |
| How did it score, overall and by category? | `metrics.json`, `metrics_by_category.csv` |
| What exact memory text went into the prompt? | `memory/by_sample/*.txt` |
| What exact request did the model receive? | `reader/traces.jsonl` |
| Which config produced this, after merging and CLI overrides? | `config.source.yaml`, `config.resolved.yaml` |
| Which code and data produced this? | `run_meta.json` (git hash, data hash, `audit_pack.v2`) |
| What did it cost? | `cost.json` |

Try other memory methods offline. Each is one flag, no new code:

```bash
python -m src.locomo_eval.run --config configs/writers/raw_chunks.yaml --reader mock --max-questions 5 --run-id smoke_raw
python -m src.locomo_eval.run --config configs/writers/session_summaries.yaml --reader mock --max-questions 5 --run-id smoke_sess
python -m src.locomo_eval.run --config configs/writers/graph.yaml --reader mock --teacher mock --max-questions 3 --run-id smoke_graph
```

Compare two arms offline, with no API:

```bash
python scripts/compare_full_runs.py --runs experiments/smoke_raw experiments/smoke_sess \
  --out experiments/compare_raw_vs_sessions
```

**Checkpoint.** The compare report shows the two arms had *different* memory text. If
`fraction_same_memory_text` is near 1, you varied nothing and the comparison is meaningless.

---

## Stage 3 — One mock experiment through the staged pipeline

Now use the harness instead of a single run. `configs/experiments/poc.yaml` is a mock matrix.

```bash
python -m src.experiment_runner write-manifest configs/experiments/poc.yaml
python -m src.experiment_runner execute-qa        configs/experiments/poc.yaml --run-index 0
python -m src.experiment_runner execute-autorater configs/experiments/poc.yaml --run-index 0
python -m src.experiment_runner aggregate         configs/experiments/poc.yaml
python -m src.experiment_runner status            configs/experiments/poc.yaml
```

**Checkpoint.** `status` reports `qa_completed` and `autorater_completed` greater than zero, and
`experiments/<name>/aggregate/` now holds `runs.parquet`, `examples.parquet`, `cells.jsonl`,
`status.json`, and a `by_run/` catalog.

Re-run `execute-qa --run-index 0`. It should **skip**, because `_SUCCESS` exists. That skip is the
resume policy: interrupted campaigns are restarted with the same command and you are not re-billed
for finished cells. Pass `--force` only when you intend to pay again.

---

## Stage 4 — One live cell

Before spending on a full matrix, prove each provider answers. Smallest possible live check:

```bash
python -m src.locomo_eval.run --config configs/presets/mem0_baseline.yaml \
  --max-questions 3 --run-id smoke_live
```

Then verify the write-path providers you plan to use:

```bash
python -m src.locomo_eval.ping_writers --providers openai,anthropic,deepseek
```

**Checkpoint.** `reader/traces.jsonl` shows a real model id and real token usage, and `cost.json` is
non-zero. If a provider errors here, fix it now — a matrix will only multiply the failure. The two
failure modes this campaign already hit and pinned:

- Hosted `gpt-5` rejects `reasoning_effort=none`; the catalog pins `minimal`. GPT-5.6 keeps `none`.
- Anthropic SDK 1.0 removed `temperature` from `messages.create`; it now travels in `extra_body`.

---

## Stage 5 — The 2025 campaign

**Active (budget):** GPT-5 vs DeepSeek-V3. Anthropic cells are parked in the original three-family
YAMLs (`2025_readers_full_context*.yaml`, `mem0_reader_2025_writers.yaml`) until more funds.
Do not mix the two experiment-name prefixes.

Three experiments, run in order. Each is one sandwich claim.

| # | Claim | Frozen | Varies | Cells |
|---|---|---|---|---|
| 1 smoke | Do the two 2025 readers work at all? | LoCoMo, `qa_mem0_v1`, `full_context`, judge | reader: GPT-5, DeepSeek-V3 | 2 |
| 2 baseline | Full-context vs Mem0-paper RAG, per reader | prompt, judge, shared `rag_locomo10` dump | reader × `{full_context, rag}` | 4 |
| 3 writers | Which 2025 writer builds better memory? | **reader frozen** to `gpt-4o-mini` + `qa_mem0_v1`, judge | writer × `{session_summaries, graph}` | 4 |

Parked three-family (adds Claude Sonnet 4.5): 3 / 6 / 6 cells. Operator: [`docs/agent/RUNBOOK_2025_LIVE.md`](agent/RUNBOOK_2025_LIVE.md).

Experiment 3 varies only the *write* model. The reader is frozen, so a difference in score is a
claim about memory construction rather than about answering ability.

Note what experiment 3 is **not**: it does not run the Mem0 paper write path. Both memory methods
send the same prompt (`teacher_session_v1` or `teacher_graph_v1`) to both writers. LoCoMo's own
`session_summaries` are dataset text and are not generated here.

### Prerequisite: the shared RAG index

Experiment 2 needs a frozen RAG dump so that every reader retrieves from identical chunks. Build it
once — chunk size 256, k=2, `text-embedding-3-small`:

```bash
python -m src.locomo_eval.rag.run_index --config configs/writers/rag.yaml --run-id rag_locomo10
```

**Checkpoint.** `experiments/rag_locomo10/rag_index/` contains `schema.json` and `index.jsonl`. If
you run in the cloud, this directory must also exist under the bucket's `shared/` prefix, or RAG
cells will exit.

### Option A — Local

Each cell is one command. Run the indices in order; watch cost between cells.

```bash
# Experiment 1 (smoke, 4 cells: 2 readers × thinking)
python -m src.experiment_runner write-manifest configs/experiments/2025_readers_openai_deepseek_smoke.yaml
python -m src.experiment_runner execute-qa        configs/experiments/2025_readers_openai_deepseek_smoke.yaml --run-index 0
python -m src.experiment_runner execute-autorater configs/experiments/2025_readers_openai_deepseek_smoke.yaml --run-index 0
python -m src.experiment_runner aggregate         configs/experiments/2025_readers_openai_deepseek_smoke.yaml
```

Repeat `--run-index 1 … 7` for experiment 2 (8 cells) and experiment 3 (8 writer cells), using
`configs/experiments/2025_readers_openai_deepseek.yaml` and
`configs/experiments/mem0_reader_2025_writers_openai_deepseek.yaml`. Always finish **all** QA for an experiment
before starting its autorater; the judge fail-fasts on a missing QA `_SUCCESS` rather than scoring a
partial answer set.

Confirm which index is which cell at any time:

```bash
python -m src.experiment_runner write-manifest configs/experiments/2025_readers_openai_deepseek.yaml
```

### Option B — Cloud Run

Use the `_gcs.yaml` overlays, which change only the storage block. Full operator detail is in
[`docs/agent/RUNBOOK_2025_OPENAI_DEEPSEEK.md`](agent/RUNBOOK_2025_OPENAI_DEEPSEEK.md). Parked three-family
operator notes: [`docs/agent/RUNBOOK_2025_LIVE.md`](agent/RUNBOOK_2025_LIVE.md). One-time GCP bootstrap is in
[`docs/gcp.md`](gcp.md). The 2026 Terra vs DeepSeek-V4 campaign uses the same
waves with [`docs/agent/RUNBOOK_2026_OPENAI_DEEPSEEK.md`](agent/RUNBOOK_2026_OPENAI_DEEPSEEK.md).

The shape per experiment, three sequential waves:

```powershell
$env:EXPERIMENT_YAML = "configs/experiments/2025_readers_openai_deepseek_smoke_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1

gcloud.cmd run jobs execute memorybench-qa         --region=us-central1 --tasks=4 --async
# wait for 4 _SUCCESS under experiments/<name>/runs/**/_SUCCESS
gcloud.cmd run jobs execute memorybench-autorater  --region=us-central1 --tasks=4 --async
# wait for 4 autorater/_SUCCESS
gcloud.cmd run jobs execute memorybench-aggregate  --region=us-central1 --tasks=1 --async
```

Three rules that cause most operator errors:

1. **Redeploy whenever you change `EXPERIMENT_YAML`.** Job arguments are baked into the job spec, so
   an un-redeployed job silently runs the previous experiment. Verify with
   `gcloud.cmd run jobs describe memorybench-qa --region=us-central1 --format="value(spec.template.spec.template.spec.containers[0].args)"`.
2. **`--parallelism` is not an `execute` flag.** It is set at deploy time (`8` here). Passing it to
   `execute` fails.
3. **Set `--tasks` to the cell count**: 4 for the smoke, 8 for experiments 2 and 3, and 1 for the
   collector.

Pull each aggregate locally when its wave finishes:

```powershell
New-Item -ItemType Directory -Force -Path experiments\locomo-2025-readers-openai-deepseek-smoke | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/locomo-2025-readers-openai-deepseek-smoke/aggregate" experiments/locomo-2025-readers-openai-deepseek-smoke/
```

For deep audit — the full `memory/` and `reader/` dumps rather than the thin catalog — run
`memorybench-collect-full` with `--tasks=1` and copy the `collected/` prefix instead.

---

## Stage 6 — Analysis

One command regenerates every table and figure from the analysis recipe. It reads finished Parquet
and never calls an API.

```bash
# One experiment
python -m src.experiment_runner report configs/analysis/campaign_2025_openai_deepseek.yaml --experiment smoke

# Every experiment plus the cross-experiment campaign concat
python -m src.experiment_runner report configs/analysis/campaign_2025_openai_deepseek.yaml
```

Outputs:

```text
experiments/<experiment-name>/analysis/tables/*.csv
experiments/<experiment-name>/analysis/plots/*.png
experiments/<experiment-name>/analysis/SUMMARY.md
experiments/_campaign/2025_openai_deepseek/analysis/...
```

Missing packs are listed and skipped, so this works before every experiment has landed.

That command is the whole analysis step — every table and figure in the repository comes out of it,
so you never need a notebook to reproduce a published number.

Notebooks are an optional review layer on top. Each is a thin wrapper that selects a scope and
displays the result of the same YAML, with no plotting code of its own. `notebooks/` is gitignored
because notebooks accumulate local paths, cloud ids, and executed output; only the notebooks backing
a specific reported result are force-added. See `notebooks/README.md` for the naming convention and
the per-study index, and write your own wrapper the same way if you want one.

To add a comparison, edit `configs/analysis/campaign_2025_openai_deepseek.yaml` and re-run `report`. Do not fork
the plotting code; the engine is shared on purpose, which is why category axes read
`1 multi-hop` / `2 temporal` / `3 open-domain` and legends sit outside the bars everywhere without
per-campaign styling.

If you want a Mem0-style judged report over a single finished pack rather than a campaign table:

```bash
# offline plumbing check
python -m scripts.analysis.run_benchmark --run experiments/<run_id> --autorater mock

# live judge (this one costs money)
python -m scripts.analysis.run_benchmark --run experiments/<run_id>
```

The mock judge is a token-overlap plumbing check, logged as `mock_sanity_not_llm_judge`. It can mark
contradictory answers correct and is excluded from any literature column. Only the live command
produces a judged `J`, and category 5 is excluded from `J` as in the paper.

---

## Stage 7 — Walk one number back to its bytes

This is the reproducibility check that matters. Pick any cell in a published table and follow it
down:

1. **`analysis/tables/<analysis_id>.csv`** — the published mean and its `n`.
2. **`aggregate/examples.parquet`** — one row per question, with `run_id`, scores, judge verdict,
   tokens, and latency.
3. **`aggregate/by_run/<run_id>/`** — that cell's metrics, cost, and summary.
4. **`experiments/<run_id>/run_meta.json`** — the git hash, data hash, model pins, and layout
   version that produced it.
5. **`experiments/<run_id>/config.resolved.yaml`** — the merged YAML plus the CLI overrides actually
   applied.
6. **`experiments/<run_id>/reader/traces.jsonl`** — the exact request and response for that question.
7. **`experiments/<run_id>/memory/`** — the exact memory text in the prompt. Query-dependent methods
   such as `rag` and `mem0` write `by_question/<qid>.txt`; whole-conversation methods such as
   `session_summaries` and `full_context` write `by_sample/<sample_id>.txt`.
8. **`memory/lineage.jsonl`** — which memory item came from which teacher.
9. **`memory/teachers/calls.jsonl`** — the write-path call behind that item.
10. **`memory/retrieve_ranks.jsonl`** — for retrieval conditions, what was rejected as well as what
    won.
11. **`autorater/traces.jsonl`** — the judge's reasoning for that verdict.

**Checkpoint.** If step 7 shows an empty or identical memory string across two arms you were
comparing, the claim is invalid regardless of the score difference. If step 4 shows a different git
hash than you expect, you are reading an older run.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `status=to_confirm`, cell refuses to run | Model has an unpinned `model_snapshot` | Pin it in `configs/models/generation_catalog.yaml`, or pass `--allow-unconfirmed` and do not call the result reproducible |
| RAG cell exits immediately | Shared index missing | Build `rag_locomo10`, and in cloud upload it under the bucket's `shared/` prefix |
| Autorater fails instantly | QA `_SUCCESS` missing for that cell | Finish QA first; the judge does not poll |
| Re-run did nothing | `_SUCCESS` exists; this is the resume policy | Use `--force` only if you intend to pay again |
| Cloud job ran the wrong experiment | Forgot to redeploy after changing `EXPERIMENT_YAML` | Redeploy, then describe the job args |
| `execute --parallelism` unrecognized | Parallelism is a deploy-time field | Ignore; deploy already set it |
| Analysis plot or table missing | Pack not pulled yet | `report` lists missing packs and skips them; pull the aggregate |
| Numbers moved after a refactor | A verifier should have caught it | Run `python -m pytest tests/ -q` and check the regression and analysis tests |

Partial failures write `errors.jsonl` and no `_SUCCESS`. Re-running the same command finishes only
the incomplete cells.

---

## Where to go next

- Architecture, layer diagrams, and the LLM call inventory: [`../README.md`](../README.md)
- On-disk contracts: [`schemas/experiment_pack.md`](schemas/experiment_pack.md),
  [`schemas/analysis_campaign.md`](schemas/analysis_campaign.md)
- Freeze and extend rules: [`reports/engineering_notebook.md`](reports/engineering_notebook.md)
- Writer and fusion methodology: [`reports/multi_teacher_methodologies.md`](reports/multi_teacher_methodologies.md)
- Dataset and third-party prompt terms: [`NOTICE.md`](../NOTICE.md)
