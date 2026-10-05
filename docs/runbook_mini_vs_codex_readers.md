# GPT-4o-mini and Codex reader analysis

This six-run-spec comparison uses the same GPT-4o-mini reader payload through Chat
Completions and Codex. Each representation has a model-only run spec, Codex with
persistence off, and Codex with persistence on.

| Representation | Model | Codex, persist off | Codex, persist on |
|---|---:|---:|---:|
| `full_context` | ✓ | ✓ | ✓ |
| LoCoMo `session_summaries` | ✓ | ✓ | ✓ |

All six run specs render the released `qa_mem0_v1` prompt with the same `{memory}` and
`{question}`. Codex receives that reader payload directly; it does not retrieve
raw session files. Persist-on adds a short Codex-only, append-only notes
instruction after the shared payload. Payload and final-task hashes are stored
separately, so persist-on is an auditable stateful-agent condition.

## Files

| File | Purpose |
|---|---|
| `configs/experiments/openai_mini_codex_readers_analysis.yaml` | Six-run-spec local matrix. |
| `configs/experiments/openai_mini_codex_readers_analysis_gcs.yaml` | GCS storage overlay. |
| `configs/analysis/campaign_openai_mini_codex_readers_analysis.yaml` | Tables, plots, and paired takeaways. |
| `notebooks/17_openai_mini_codex_readers_analysis.ipynb` | Thin report wrapper. |

## Run

Set your GCP project, region, private bucket, and upload the fetched LoCoMo
dataset to `gs://$env:BUCKET/data/locomo10.json`. Then:

```powershell
python -m src.experiment_runner write-manifest configs/experiments/openai_mini_codex_readers_analysis_gcs.yaml

$env:EXPERIMENT_YAML = "configs/experiments/openai_mini_codex_readers_analysis_gcs.yaml"
# Full-context Codex persistence needs this task memory. Three concurrent
# tasks keep the deployment within the default regional allocation quota.
$env:JOB_MEMORY = "8Gi"
$env:JOB_CPU = "2"
$env:JOB_PARALLELISM = "3"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1
gcloud.cmd run jobs execute memorybench-qa --region=$env:REGION --tasks=6 --async
```

`--async` returns as soon as the execution is submitted. The six tasks keep
running after the shell exits. Start the next wave only after the previous
one has finished:

```powershell
gcloud.cmd run jobs execute memorybench-autorater --region=$env:REGION --tasks=6 --async
gcloud.cmd run jobs execute memorybench-aggregate --region=$env:REGION --tasks=1 --async
gcloud.cmd run jobs execute memorybench-collect-full --region=$env:REGION --tasks=1 --async
```

Cells with `_SUCCESS` are skipped. Use a new experiment name for a paid
regeneration rather than overwriting an audited pack.

## Pull and report

```powershell
$name = "locomo-openai-mini-codex-readers-analysis-v2"
New-Item -ItemType Directory -Force -Path "experiments/$name" | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$name/aggregate" "experiments/$name/"
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$name/collected" "experiments/$name/"

python -m src.experiment_runner report configs/analysis/campaign_openai_mini_codex_readers_analysis.yaml
```

## Audit gate

For each Codex run spec, inspect `agent/events.jsonl`, `agent/traces.jsonl`,
`agent/trajectory.jsonl`, `agent/metrics.json`, `agent/COMPARISON.md`, and
`agent/workspaces/`. Require `n_web_search=0`, `n_mcp=0`, and
`used_non_workspace_tools=0`. Persist-on notes snapshots show any state that
could affect later questions. `prompt_injected` is expected for reader-prompt
run specs and is not a workspace retrieval failure. The pack preserves
CLI-emitted reasoning and usage only; it cannot expose hidden chain-of-thought.
