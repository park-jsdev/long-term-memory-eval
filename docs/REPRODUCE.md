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
| 5 | The six-cell reader comparison | substantial | yes |
| 6 | Campaign tables, plots, and notebook review | none | no |
| 7 | Audit trail walked back from a published number | none | no |

Throughout: never commit `.env` or any key. Do not claim Mem0 paper Table 1–2 `J` from these OSS
clones — the judged `J` here is a protocol clone, not a Platform reproduction.

---

## Stage 0 — Environment and dataset

```bash
conda create -n <your-env-name> python=3.11 -y
conda activate <your-env-name>
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

What you have just locked: scoring and indexing; the judge protocol, including the
category-5 exclusion; the sandwich contracts (one YAML to one pack, replace-not-append,
gold answers never entering a prompt); YAML merge semantics; hashed run ids; and
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
| Which code and data produced this? | `run_meta.json` (git hash, data hash, `audit_pack.v3`) |
| What did it cost? | `cost.json` |

Try other memory methods offline. Each is one flag, no new code:

```bash
python -m src.locomo_eval.run --config configs/writers/raw_chunks.yaml --reader mock --max-questions 5 --run-id smoke_raw
python -m src.locomo_eval.run --config configs/writers/session_summaries.yaml --reader mock --max-questions 5 --run-id smoke_sess
python -m src.locomo_eval.run --config configs/writers/graph.yaml --reader mock --writer mock --max-questions 3 --run-id smoke_graph
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

## Stage 5 — Reader comparison

Six cells. Both sides use GPT-4o-mini. Chat Completions receives the released `qa_mem0_v1`
prompt as one request. Codex receives that same rendered payload. The two texts are stuffed
`full_context` and the dataset `session_summaries` field. No writer model runs. Codex is
included with persist off and persist on. Persist-on may add notes after the first question,
so that pair is a trajectory condition, not a byte-identical request.

| Text | Chat Completions | Codex, persist off | Codex, persist on |
|---|---|---|---|
| `full_context` | 1 | 1 | 1 |
| `session_summaries` | 1 | 1 | 1 |

YAML: `configs/experiments/openai_mini_codex_readers_analysis.yaml`. Operator steps:
[`runbook_mini_vs_codex_readers.md`](runbook_mini_vs_codex_readers.md).

A separate sandwich freezes the GPT-4o-mini reader and compares a GPT-4o-mini writer with a
Codex writer on `session_summaries` and `graph` (2 cells). Those prompts are
`prompts/writers/session_summary_v1.txt` and `prompts/writers/graph_v1.txt`. Operator steps:
[`runbook_mini_vs_codex_writers.md`](runbook_mini_vs_codex_writers.md). Dataset
`session_summaries` with no writer is the reader cell above, not that sandwich.

### Option A — Local

`write-manifest` prints `wrote 6 runs`. Finish every QA index before the autorater. The judge
stops if a cell has no QA `_SUCCESS`.

```bash
python -m src.experiment_runner write-manifest configs/experiments/openai_mini_codex_readers_analysis.yaml
python -m src.experiment_runner execute-qa        configs/experiments/openai_mini_codex_readers_analysis.yaml --run-index 0
python -m src.experiment_runner execute-autorater configs/experiments/openai_mini_codex_readers_analysis.yaml --run-index 0
python -m src.experiment_runner aggregate         configs/experiments/openai_mini_codex_readers_analysis.yaml
```

Repeat `--run-index` for `1` through `5` on both QA and the autorater.

### Option B — Cloud Run

Set `PROJECT_ID`, `REGION`, and `BUCKET` in the environment. Do not write them into a tracked
file. One-time bootstrap is in [`docs/gcp.md`](gcp.md). The `_gcs.yaml` overlay changes only
storage. Line-by-line steps are in the reader runbook.

```powershell
$env:EXPERIMENT_YAML = "configs/experiments/openai_mini_codex_readers_analysis_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1

gcloud.cmd run jobs execute memorybench-qa         --region=$env:REGION --tasks=6 --async
# wait for 6 _SUCCESS under experiments/locomo-openai-mini-codex-readers-analysis-v2/runs/**/_SUCCESS
gcloud.cmd run jobs execute memorybench-autorater  --region=$env:REGION --tasks=6 --async
# wait for 6 autorater/_SUCCESS
gcloud.cmd run jobs execute memorybench-aggregate  --region=$env:REGION --tasks=1 --async
gcloud.cmd run jobs execute memorybench-collect-full --region=$env:REGION --tasks=1 --async
```

Three rules that cause most operator errors:

1. **Redeploy whenever you change `EXPERIMENT_YAML`.** Job arguments are baked into the job spec, so
   an un-redeployed job silently runs the previous experiment. Verify with
   `gcloud.cmd run jobs describe memorybench-qa --region=$env:REGION --format="value(spec.template.spec.template.spec.containers[0].args)"`.
2. **`--parallelism` is not an `execute` flag.** It is set at deploy time. Passing it to
   `execute` fails.
3. **Set `--tasks` to the cell count**: 6 for QA and the autorater, and 1 for each collector.

Pull the aggregate when the collector finishes:

```powershell
New-Item -ItemType Directory -Force -Path experiments\locomo-openai-mini-codex-readers-analysis-v2 | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/locomo-openai-mini-codex-readers-analysis-v2/aggregate" experiments/locomo-openai-mini-codex-readers-analysis-v2/
```

For the full `memory/`, `reader/`, and `agent/` dumps, copy the `collected/` prefix as well.

---

## Stage 6 — Analysis

One command regenerates every table and figure from the analysis recipe. It reads finished Parquet
and never calls an API.

```bash
python -m src.experiment_runner report configs/analysis/campaign_openai_mini_codex_readers_analysis.yaml
```

Outputs:

```text
experiments/_campaign/openai_mini_codex_readers_analysis/analysis/tables/*.csv
experiments/_campaign/openai_mini_codex_readers_analysis/analysis/plots/*.png
experiments/_campaign/openai_mini_codex_readers_analysis/analysis/SUMMARY.md
```

Missing packs are listed and skipped, so this works before every experiment has landed.

That command is the whole analysis step — every table and figure in the repository comes out of it,
so you never need a notebook to reproduce a published number.

Notebooks are an optional local review layer. They are gitignored. `report` is the
reproduction step.

To add a comparison, edit `configs/analysis/campaign_openai_mini_codex_readers_analysis.yaml` and re-run `report`. Do not fork
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
8. **`memory/lineage.jsonl`** — which memory item came from which writer.
9. **`memory/writer/calls.jsonl`** — the write-path call behind that item.
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

- Architecture, layer diagrams, and the LLM call inventory: [`architecture.md`](architecture.md)
- On-disk contracts: [`schemas/experiment_pack.md`](schemas/experiment_pack.md),
  [`schemas/analysis_campaign.md`](schemas/analysis_campaign.md)
- One question, from config to score: [`loop.md`](loop.md)
- Dataset and third-party prompt terms: [`NOTICE.md`](../NOTICE.md)
