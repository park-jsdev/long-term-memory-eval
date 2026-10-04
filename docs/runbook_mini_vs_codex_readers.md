# GPT-4o-mini vs Codex with matched reader prompts

This six-cell run compares the same GPT-4o-mini reader payload through Chat
Completions and Codex. Each representation has a model-only cell, Codex with
persistence off, and Codex with persistence on.

| Representation | Model | Codex, persist off | Codex, persist on |
|---|---:|---:|---:|
| `full_context` | ✓ | ✓ | ✓ |
| LoCoMo `session_summaries` | ✓ | ✓ | ✓ |

All cells render the released `qa_mem0_v1` prompt with the same `{memory}` and
`{question}`. Codex receives that rendered reader payload directly; it does
not retrieve raw session files. Persist-on has an empty, auditable notes
workspace and may add state after the first question, so it is a best-agent
trajectory condition rather than exact end-to-end request parity.

## Files

| File | Purpose |
|---|---|
| `configs/experiments/openai_mini_codex_prompt_parity.yaml` | Six-cell local matrix. |
| `configs/experiments/openai_mini_codex_prompt_parity_gcs.yaml` | GCS storage overlay. |
| `configs/analysis/campaign_openai_mini_codex_prompt_parity.yaml` | Tables, plots, and paired takeaways. |
| `notebooks/17_openai_mini_codex_prompt_parity_analysis.ipynb` | Thin report wrapper. |

## Run

Set your GCP project, region, private bucket, and upload the fetched LoCoMo
dataset to `gs://$env:BUCKET/data/locomo10.json`. Then:

```powershell
python -m src.experiment_runner write-manifest configs/experiments/openai_mini_codex_prompt_parity_gcs.yaml

$env:EXPERIMENT_YAML = "configs/experiments/openai_mini_codex_prompt_parity_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1
gcloud.cmd run jobs execute memorybench-qa --region=$env:REGION --tasks=6 --async
```

Wait for six QA `_SUCCESS` markers, then run the separate judge and collectors:

```powershell
gcloud.cmd run jobs execute memorybench-autorater --region=$env:REGION --tasks=6 --async
gcloud.cmd run jobs execute memorybench-aggregate --region=$env:REGION --tasks=1 --async
gcloud.cmd run jobs execute memorybench-collect-full --region=$env:REGION --tasks=1 --async
```

Cells with `_SUCCESS` are skipped. Use a new experiment name for a paid
regeneration rather than overwriting an audited pack.

## Pull and report

```powershell
$name = "locomo-openai-mini-codex-prompt-parity"
New-Item -ItemType Directory -Force -Path "experiments/$name" | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$name/aggregate" "experiments/$name/"
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$name/collected" "experiments/$name/"

python -m src.experiment_runner report configs/analysis/campaign_openai_mini_codex_prompt_parity.yaml
```

## Audit gate

For each Codex cell, inspect `agent/events.jsonl`, `agent/traces.jsonl`,
`agent/trajectory.jsonl`, `agent/metrics.json`, `agent/COMPARISON.md`, and
`agent/workspaces/`. Require `n_web_search=0`, `n_mcp=0`, and
`used_non_workspace_tools=0`. Persist-on notes snapshots show any state that
could affect later questions. The pack preserves CLI-emitted reasoning and
usage only; it cannot expose hidden chain-of-thought.
