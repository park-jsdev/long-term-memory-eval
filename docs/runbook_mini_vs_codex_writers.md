# GPT-4o-mini writers vs Codex writers

Operator steps for the two Chat Completions writer cells, then the campaign report that compares them to the Codex writer packs already in the bucket. Run this from the repository root in PowerShell, after `gcloud auth login`.

The two new cells call OpenAI. They do not call Codex. The Codex packs are downloaded, not rerun. The bucket is private: a clone can run these commands only with access to this project.

## What each file is for

Change a file only for the reason in the last column. The GCS file should stay a storage overlay.

| File | What it fixes | Why an operator touches it |
|---|---|---|
| `configs/experiments/openai_mini_writers_structured.yaml` | The claim: frozen `gpt-4o-mini` reader, writer `gpt-4o-mini`, methods `session_summaries` and `graph`, thinking off, seed 1. Experiment name `locomo-openai-mini-writers-structured`. That name is the bucket prefix. | This is the matrix. Two cells. Change `matrix.writer` to swap the writer model. Change `memory_method` only if the comparison should cover a different method. |
| `configs/experiments/openai_mini_writers_structured_gcs.yaml` | Includes the file above and sets `storage.backend: gcs`. Dataset object `data/locomo10.json`. Scratch dir `/tmp/memorybench-experiments` on the VM. The bucket is not in the file. | Point Cloud Run at this file, not the local one. `scripts/deploy_gcp.ps1` reads `$env:EXPERIMENT_YAML` and `$env:BUCKET`. |
| `configs/stacks/qa_default.yaml` | Frozen reader stack: LoCoMo pin, `qa_mem0_v1`, `gpt-4o-mini`. | Leave it. A reader change is a different experiment. |
| `configs/writers/openai_mini.yaml` | Default writer block: provider `openai`, model `gpt-4o-mini`, summary prompt and graph prompt. | The matrix above overrides the model for these two cells. Edit this file when a one-off `graph.yaml` run should use another model. |
| `configs/writers/graph.yaml` | Includes the stack and `openai_mini.yaml`. `pipeline.memory: graph`. | The runner loads this because the matrix cell says `graph`. |
| `configs/writers/session_summaries.yaml` | Dataset summaries when no writer is set. | This matrix sets a writer, so the model writes the summaries. The dataset text is not what gets scored. |
| `configs/models/generation_catalog.yaml` | `gpt-4o-mini` resolves to the API id sent on the wire. | Edit when the catalog id and the provider id diverge. |
| `configs/analysis/campaign_openai_mini_vs_codex_writers.yaml` | Which local packs to plot. `chat` is the pack this runbook builds. `codex` is `locomo-openai-codex-poc-writers`. `readers` is `locomo-openai-codex-poc-readers` (stuffed full context and persist-off Codex workspace). | The report reads `experiments/<name>/aggregate/`. A missing pack is skipped, and the figure that needed it is empty. |
| `scripts/deploy_gcp.ps1` | Builds the image, pushes `memorybench:<git-sha>`, and sets the four Cloud Run jobs to the YAML in `$env:EXPERIMENT_YAML`. Mounts Secret Manager keys. Sets `MEMORYBENCH_BUCKET` from `$env:BUCKET`. | Run it after any code or YAML change the jobs should pick up. Job names stay `memorybench-qa`, `memorybench-autorater`, `memorybench-aggregate`, `memorybench-collect-full`. |

`write-manifest` on the GCS YAML prints `wrote 2 runs`. On a laptop that file lands under `\tmp\memorybench-experiments\...` because the YAML scratch path is absolute. The jobs expand the same YAML themselves. The local manifest is the check that the matrix is two cells before you pay for them.

## Setup

Docker Desktop has to be running. Secret Manager in this project already has `openai-api-key`, `codex-api-key`, `anthropic-api-key`, and `deepseek-api-key`. Deploy fails if one of those four is missing, even though these two cells only call OpenAI.

```powershell
conda create -n distillation python=3.11 -y
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py

$env:PROJECT_ID = "agent-platform-508416"
$env:REGION = "us-central1"
$env:BUCKET = "agent-platform-508416-memorybench"
gcloud.cmd config set project $env:PROJECT_ID

gcloud.cmd storage cp data/raw/locomo10.json "gs://$env:BUCKET/data/locomo10.json"
```

`fetch_locomo.py` writes `data/raw/locomo10.json`. That file is gitignored. Re-upload it only when the LoCoMo file changes. Jobs fail if `gs://$env:BUCKET/data/locomo10.json` is missing.

## QA, judge, collect

```powershell
python -m src.experiment_runner write-manifest configs/experiments/openai_mini_writers_structured_gcs.yaml

$env:EXPERIMENT_YAML = "configs/experiments/openai_mini_writers_structured_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1

gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=2 --async
```

Wait until both QA cells have `_SUCCESS`. `--tasks=2` is the matrix size (summaries, then graph).

```powershell
gcloud.cmd storage ls -r "gs://$env:BUCKET/experiments/locomo-openai-mini-writers-structured/runs/**/_SUCCESS"
```

Then the judge. It stops immediately if a cell has no QA `_SUCCESS`.

```powershell
gcloud.cmd run jobs execute memorybench-autorater --region=us-central1 --tasks=2 --async
```

Wait until both autorater packs exist:

```powershell
gcloud.cmd storage ls -r "gs://$env:BUCKET/experiments/locomo-openai-mini-writers-structured/runs/**/autorater/_SUCCESS"
```

Then one collector for the thin catalog, and one for the full packs:

```powershell
gcloud.cmd run jobs execute memorybench-aggregate --region=us-central1 --tasks=1 --async
gcloud.cmd run jobs execute memorybench-collect-full --region=us-central1 --tasks=1 --async
```

Aggregate can start after the autorater `_SUCCESS` files exist. Collect-full copies `memory/` and is the pack you audit. The report only needs `aggregate/`.

A cell that already has `_SUCCESS` is skipped. Add a redeploy and, on a rerun you intend to pay for again, you would pass a fresh experiment name or delete that cell's prefix. These jobs do not take `--force` on the `gcloud` command line. The container args are the YAML only.

## Pull

```powershell
$name = "locomo-openai-mini-writers-structured"
New-Item -ItemType Directory -Force -Path "experiments/$name" | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$name/aggregate" "experiments/$name/"
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$name/collected" "experiments/$name/"

$writers = "locomo-openai-codex-poc-writers"
New-Item -ItemType Directory -Force -Path "experiments/$writers" | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$writers/aggregate" "experiments/$writers/"

$readers = "locomo-openai-codex-poc-readers"
New-Item -ItemType Directory -Force -Path "experiments/$readers" | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$readers/aggregate" "experiments/$readers/"
```

`codex` in the campaign YAML is the writer pack (summaries, facts, graph). `readers` is the ceiling overlay (stuffed Chat Completions full context and persist-off Codex workspace). Without `readers`, `methods_vs_full_context` is skipped. `structured_writers` still runs from the chat pack plus the Codex writer pack.

## Report

No API calls. Tables and plots land in `experiments/_campaign/openai_mini_vs_codex_writers/analysis/`.

```powershell
python -m src.experiment_runner report configs/analysis/campaign_openai_mini_vs_codex_writers.yaml
```
