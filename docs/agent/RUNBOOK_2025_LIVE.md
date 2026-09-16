# 2025 live campaign runbook

**Completed 2025 (budget):** GPT-5 vs DeepSeek-V3 — [`docs/agent/RUNBOOK_2025_OPENAI_DEEPSEEK.md`](RUNBOOK_2025_OPENAI_DEEPSEEK.md).  
**Next 2026 (budget):** GPT-5.6 Terra vs DeepSeek-V4 — [`docs/agent/RUNBOOK_2026_OPENAI_DEEPSEEK.md`](RUNBOOK_2026_OPENAI_DEEPSEEK.md).  
**Parked:** this three-family sequence (adds Claude Sonnet 4.5) until Anthropic spend is available again. Do not mix GCS prefixes.

**Audience:** operator replaying this exact three-experiment Cloud Run sequence.  
**Date started:** 2026-09-15.  
**Not this file:** one-time GCP bootstrap (`docs/agent/GCP_RUNBOOK.md`), resource inventory (`infra/gcp/README.md`), full scientific matrix (`docs/agent/EXPERIMENT_MATRIX_v1.md`).

Do **not** claim Mem0 paper Table 1–2 J from these OSS clones. Do **not** launch `mem0_reader_2024_writers_*` (Claude 3.5 Sonnet returns API `404`).

Windows: repo root, PowerShell, `gcloud.cmd` (not `gcloud.ps1`). Never commit `.env` or keys.

---

## What this campaign is

Three sequential Cloud Run experiments. Redeploy between them so job args point at the right YAML. Within an experiment: **all QA → then autorater → then aggregate**. Tasks inside a wave may run in parallel (job `--parallelism=3`); cells do not share artifacts.

| # | Claim | Cloud YAML | Cells | `--tasks` | GCS experiment name | Notebooks |
|---|--------|------------|-------|-----------|---------------------|-----------|
| 1 smoke | 2025 readers × `full_context`, 1 conversation / 5 questions | `configs/experiments/2025_readers_full_context_smoke_gcs.yaml` | 3 | 3 | `locomo-2025-readers-full-context-smoke` | `03_*` |
| 2 baseline | Same readers × `{full_context, rag}`, **full LoCoMo** | `configs/experiments/2025_readers_full_context_gcs.yaml` | 6 | 6 | `locomo-2025-readers-full-context` | `04_*` |
| 3 writers | Frozen Mem0-parity reader × 2025 teachers × `{summaries, graph}`, full LoCoMo | `configs/experiments/mem0_reader_2025_writers_gcs.yaml` | 6 | 6 | `locomo-mem0-reader-2025-writers` | `05_*` |

Optional cheap check of experiment 3 (different `run_id`s; not required if experiment 1 already proved the three APIs): `configs/experiments/mem0_reader_2025_writers_smoke_gcs.yaml` (`--tasks=6`, name `locomo-mem0-reader-2025-writers-smoke`).

---

## Freeze / vary (sandwich)

| Experiment | Frozen | Varies |
|------------|--------|--------|
| 1 | LoCoMo + `qa_mem0_v1` + `full_context` + GPT-4o-mini judge | Answer model: GPT-5, Claude Sonnet 4.5, DeepSeek-V3 |
| 2 | Same prompt + judge + **shared** `rag_locomo10` dump | Answer model × `{full_context, rag}` |
| 3 | LoCoMo + `qa_mem0_v1` + **GPT-4o-mini reader** + GPT-4o-mini judge | Writer × `{teacher_session_summaries, teacher_graph}` |

### Session summaries vs Mem0 extract

Experiment 3 **does not** run the Mem0 paper write path (`mem0_extract_v1` / `mem0_update_v1`).

| Id | What it is |
|----|------------|
| `teacher_session_summaries` | Same prompt `prompts/teachers/teacher_session_v1.txt` to all three writers |
| `teacher_graph` | Same prompt `prompts/teachers/teacher_graph_v1.txt` → locked Mem0g JSON, three writers |
| LoCoMo `session_summaries` | Dataset-released text; **not generated** in this campaign |
| `mem0` / `mem0_locomo10` | Frozen gpt-4o-mini extract dump; **not** in this campaign |

---

## Pins (copy these into the lab notebook)

| Pin | Value |
|-----|--------|
| GCP project | `$env:PROJECT_ID` (set per shell; see `GCP_RUNBOOK.md`) |
| Region | `us-central1` |
| Bucket | `gs://$env:BUCKET` (defaults to `$env:PROJECT_ID-memorybench`) |
| Jobs | `memorybench-qa`, `memorybench-autorater`, `memorybench-aggregate`, `memorybench-collect-full` |
| Runner SA | `memorybench-runner@$env:PROJECT_ID.iam.gserviceaccount.com` |
| Dataset blob | `gs://…/data/locomo10.json` |
| Answer prompt | `prompts/readers/qa_mem0_v1.txt` |
| Judge | `gpt-4o-mini` (Mem0-style CORRECT/WRONG; category 5 excluded from J) |
| GPT-5 | catalog `gpt-5`, API `gpt-5` (`model_snapshot` still `TO_CONFIRM`) |
| Claude | catalog `claude-sonnet-4-5`, API `claude-sonnet-4-5-20250929` |
| DeepSeek | catalog `deepseek-v3`, API `deepseek-chat` (`model_snapshot` still `TO_CONFIRM`) |
| Frozen reader (exp 3) | catalog `gpt-4o-mini`, API `gpt-4o-mini` |
| RAG | `configs/writers/rag.yaml`: chunk **256**, **k=2**, `cl100k_base`, `text-embedding-3-small` |
| RAG dump | `experiments/rag_locomo10/rag_index/` uploaded to `gs://…/shared/rag_locomo10/` |
| Job parallelism | `3` (set at deploy; **not** an `execute` flag) |
| Task timeout | 12h; max retries 2; memory 4Gi |
| Seed | 1 |

`run_id` is `<experiment-slug>-<8 hex>` hashed from QA identity (reader, memory, writer, indexes, subset, seed). Same YAML cell always gets the same id. Changing `experiment.name` or subset mints new ids.

Image tag is `git rev-parse --short HEAD` at deploy. Record it.

---

## Cell order (`CLOUD_RUN_TASK_INDEX`)

Confirm anytime with:

```powershell
python -m src.memorybench write-manifest configs/experiments/<that_gcs.yaml>
```

**Experiment 1** (smoke overlay replaces `memory_method` with `[full_context]` only):

| Index | Reader | Memory |
|-------|--------|--------|
| 0 | `gpt-5` | `full_context` |
| 1 | `claude-sonnet-4-5-20250929` | `full_context` |
| 2 | `deepseek-chat` | `full_context` |

**Experiment 2** (reader × memory):

| Index | Reader | Memory |
|-------|--------|--------|
| 0 | `gpt-5` | `full_context` |
| 1 | `gpt-5` | `rag` |
| 2 | `claude-sonnet-4-5-20250929` | `full_context` |
| 3 | `claude-sonnet-4-5-20250929` | `rag` |
| 4 | `deepseek-chat` | `full_context` |
| 5 | `deepseek-chat` | `rag` |

**Experiment 3** (memory × writer; reader always `gpt-4o-mini`):

| Index | Memory | Writer |
|-------|--------|--------|
| 0 | `teacher_session_summaries` | `gpt-5` |
| 1 | `teacher_session_summaries` | `claude-sonnet-4-5-20250929` |
| 2 | `teacher_session_summaries` | `deepseek-chat` |
| 3 | `teacher_graph` | `gpt-5` |
| 4 | `teacher_graph` | `claude-sonnet-4-5-20250929` |
| 5 | `teacher_graph` | `deepseek-chat` |

---

## Operator rules

1. Redeploy whenever `EXPERIMENT_YAML` changes. Job args are baked into the image/job spec.
2. Confirm args before execute:
   `spec.template.spec.template.spec.containers[0].args`
3. Do **not** pass `--parallelism` to `gcloud run jobs execute` (unrecognized). Parallelism is `deploy_gcp.ps1` / `gcloud run jobs update`.
4. Do **not** start autorater until that experiment’s QA `_SUCCESS` files exist. Autorater fail-fasts; it does not poll.
5. Aggregate `--tasks=1`. Collect-full is optional (`--tasks=1`) after aggregate.
6. Cloud Run retries skip a cell when `_SUCCESS` exists. Re-execute the same `--tasks=N` to finish failures without rebilling successes. `--force` regenerates (paid).
7. PowerShell: keep the logging filter in **one** quoted string. Do not split `gcloud logging read` across lines with a broken filter.
8. RAG cells download `gs://…/shared/rag_locomo10/` into `/tmp/memorybench-experiments/rag_locomo10/rag_index/`. If that prefix is empty, RAG tasks exit.

---

## Prerequisites (already done if you ran the PoC)

- Docker Desktop, `gcloud.cmd` auth, project `${env:PROJECT_ID}`
- Secrets `openai-api-key`, `anthropic-api-key`, `deepseek-api-key` mapped on the jobs
- `gs://${env:BUCKET}/data/locomo10.json`

If the RAG dump is **not** already in the bucket:

```powershell
conda activate distillation
python -m src.locomo_eval.rag.run_index --config configs/writers/rag.yaml --run-id rag_locomo10
gcloud.cmd storage cp -r experiments/rag_locomo10 gs://${env:BUCKET}/shared/
```

Verify (required before experiment 2):

```powershell
gcloud.cmd storage ls gs://${env:BUCKET}/shared/rag_locomo10/rag_index/
```

You should see `schema.json` and `index.jsonl` (and `by_sample/`). Skip `run_index` if this listing is already populated.

---

## Constants for every command below

```powershell
$env:PROJECT_ID = "${env:PROJECT_ID}"
$env:REGION = "us-central1"
gcloud.cmd config set project $env:PROJECT_ID
```

Helpers used after each deploy:

```powershell
function Show-QaArgs {
  gcloud.cmd run jobs describe memorybench-qa --region=us-central1 --format="value(spec.template.spec.template.spec.containers[0].args)"
}
function Show-QaExec {
  gcloud.cmd run jobs executions list --job=memorybench-qa --region=us-central1 --limit=5
}
function Show-QaLogs {
  gcloud.cmd logging read "resource.labels.job_name=memorybench-qa" --limit=40 --format="value(textPayload)"
}
```

---

## Experiment 1 — reader smoke

```powershell
$env:EXPERIMENT_YAML = "configs/experiments/2025_readers_full_context_smoke_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1
Show-QaArgs
# must contain: 2025_readers_full_context_smoke_gcs.yaml

gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=3 --async
```

Wait until **three** `_SUCCESS` files exist:

```powershell
Show-QaExec
gcloud.cmd storage ls gs://${env:BUCKET}/experiments/locomo-2025-readers-full-context-smoke/runs/**/_SUCCESS
Show-QaLogs
```

Success log line looks like `qa complete locomo-2025-readers-full-context-smoke-… uploaded=N blobs`.

Then:

```powershell
gcloud.cmd run jobs execute memorybench-autorater --region=us-central1 --tasks=3 --async
# after three …/autorater/_SUCCESS:
gcloud.cmd run jobs execute memorybench-aggregate --region=us-central1 --tasks=1 --async
```

Pull + review `notebooks/03_2025_readers_full_context_analysis.ipynb`:

```powershell
New-Item -ItemType Directory -Force -Path experiments\locomo-2025-readers-full-context-smoke | Out-Null
gcloud.cmd storage cp -r gs://${env:BUCKET}/experiments/locomo-2025-readers-full-context-smoke/aggregate experiments/locomo-2025-readers-full-context-smoke/
```

Do not start experiment 2 until these three cells are green.

After aggregate is local, regenerate analysis tables/plots (no LLM):

```powershell
python -m src.memorybench report configs/analysis/campaign_2025_live.yaml --experiment smoke
```

Notebooks: `03_2025_readers_full_context_analysis.ipynb` (this pack) and `06_2025_live_campaign_analysis.ipynb` (campaign concat). YAML: `configs/analysis/campaign_2025_live.yaml`.

---

## Experiment 2 — full-context + Mem0-paper RAG

```powershell
$env:EXPERIMENT_YAML = "configs/experiments/2025_readers_full_context_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1
Show-QaArgs
# must contain 2025_readers_full_context_gcs.yaml and must NOT contain smoke

gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=6 --async
```

This wave is expensive (three families × whole LoCoMo; RAG cells also retrieve). Wait for **six** `_SUCCESS`:

```powershell
gcloud.cmd storage ls gs://${env:BUCKET}/experiments/locomo-2025-readers-full-context/runs/**/_SUCCESS
```

Then:

```powershell
gcloud.cmd run jobs execute memorybench-autorater --region=us-central1 --tasks=6 --async
gcloud.cmd run jobs execute memorybench-aggregate --region=us-central1 --tasks=1 --async
```

```powershell
New-Item -ItemType Directory -Force -Path experiments\locomo-2025-readers-full-context | Out-Null
gcloud.cmd storage cp -r gs://${env:BUCKET}/experiments/locomo-2025-readers-full-context/aggregate experiments/locomo-2025-readers-full-context/
```

Review `notebooks/04_2025_readers_full_context_analysis.ipynb`.

---

## Experiment 3 — frozen gpt-4o-mini × 2025 writers

```powershell
$env:EXPERIMENT_YAML = "configs/experiments/mem0_reader_2025_writers_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1
Show-QaArgs
# must contain: mem0_reader_2025_writers_gcs.yaml

gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=6 --async
```

Wait for **six** `_SUCCESS`:

```powershell
gcloud.cmd storage ls gs://${env:BUCKET}/experiments/locomo-mem0-reader-2025-writers/runs/**/_SUCCESS
```

Then:

```powershell
gcloud.cmd run jobs execute memorybench-autorater --region=us-central1 --tasks=6 --async
gcloud.cmd run jobs execute memorybench-aggregate --region=us-central1 --tasks=1 --async
```

```powershell
New-Item -ItemType Directory -Force -Path experiments\locomo-mem0-reader-2025-writers | Out-Null
gcloud.cmd storage cp -r gs://${env:BUCKET}/experiments/locomo-mem0-reader-2025-writers/aggregate experiments/locomo-mem0-reader-2025-writers/
```

Review `notebooks/05_mem0_reader_2025_writers_analysis.ipynb`.

---

## Optional: full audit packs

After that experiment’s aggregate is done (jobs still pointed at that YAML):

```powershell
gcloud.cmd run jobs execute memorybench-collect-full --region=us-central1 --tasks=1 --async
```

```powershell
gcloud.cmd storage cp -r gs://${env:BUCKET}/experiments/<experiment-name>/collected experiments/<experiment-name>/
```

Local: `experiments/<experiment-name>/collected/runs/<run_id>/` including `memory/` and `reader/`.

---

## GCS layout

```text
gs://${env:BUCKET}/
  data/locomo10.json
  shared/rag_locomo10/rag_index/     # frozen Mem0-paper RAG dump
  experiments/<experiment-name>/
    runs/<run_id>/                   # QA pack; autorater writes autorater/
      _SUCCESS
      autorater/_SUCCESS
    aggregate/                       # after memorybench-aggregate
    collected/                       # after memorybench-collect-full
```

Console: [Cloud Run jobs](https://console.cloud.google.com/run/jobs) · [buckets](https://console.cloud.google.com/storage/browser) — both pick up whichever project is selected in the console header.

Durable output is the bucket. `/tmp/memorybench-experiments` on the VM is scratch.

---

## Failure modes

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `reasoning_effort` does not support `none` on `gpt-5` | Hosted gpt-5 only accepts `minimal`/`low`/`medium`/`high` | Fixed in catalog (`minimal`). Redeploy this image. |
| `Messages.create() unexpected keyword argument 'temperature'` | Anthropic SDK 1.0 dropped sampling kwargs | Fixed (`extra_body`). Redeploy this image. |
| `anthropic.NotFoundError` on `claude-3-5-sonnet-*` | Wrong YAML (2024 campaign) | Stop. This runbook’s Claude pin is `claude-sonnet-4-5-20250929`. |
| Job args still `poc_gcs.yaml` or the previous experiment | Forgot redeploy | Set `EXPERIMENT_YAML`, run `deploy_gcp.ps1`, `Show-QaArgs`. |
| RAG task: missing `rag_index` / empty `shared/rag_locomo10` | Dump not uploaded or wrong prefix | `storage ls` the prefix; re-upload `experiments/rag_locomo10` under `shared/`. |
| Autorater fails immediately | QA `_SUCCESS` missing for that `run_id` | Finish QA first; same YAML so hashed ids match. |
| `execute --parallelism=…` unrecognized | Parallelism is a job update field | Ignore; deploy already sets `--parallelism=3`. |
| Logging filter syntax error in PowerShell | Split filter / `.ps1` parser | `gcloud.cmd logging read "resource.labels.job_name=memorybench-qa"` as one string. |
| Dataset download fail | Missing `data/locomo10.json` in bucket | Upload once (see `GCP_RUNBOOK.md`). |
| DeepSeek 404 | Hosted `deepseek-chat` alias moved | Pin a current id in `generation_catalog.yaml`, redeploy, new `run_id`s if API id changes. |

Partial failures write `errors.jsonl` and **do not** write `_SUCCESS`. Re-execute the same `--tasks=N`; finished cells skip.

---

## Local replay (no Cloud Run)

Same matrices, laptop `experiments/` (needs keys in repo-root `.env` except mock):

```powershell
conda activate distillation
python -m src.memorybench write-manifest configs/experiments/2025_readers_full_context.yaml
python -m src.memorybench execute-qa configs/experiments/2025_readers_full_context.yaml --run-index 0
# … --run-index 1..5 for experiment 2
python -m src.memorybench execute-autorater configs/experiments/2025_readers_full_context.yaml --run-index 0
python -m src.memorybench aggregate configs/experiments/2025_readers_full_context.yaml
```

Smoke overlay and writers YAML work the same way. RAG still needs a local `experiments/rag_locomo10/rag_index/`.

---

## Record sheet (fill as you go)

| Field | Value |
|-------|--------|
| Git SHA at last deploy | |
| Image tag (`memorybench:<sha>`) | |
| RAG dump already in GCS? | yes / rebuilt on ____ |
| Exp 1 QA execution id | |
| Exp 1 autorater execution id | |
| Exp 2 QA execution id | |
| Exp 2 autorater execution id | |
| Exp 3 QA execution id | |
| Exp 3 autorater execution id | |

Execution id: `gcloud.cmd run jobs executions list --job=memorybench-qa --region=us-central1 --limit=3`.
