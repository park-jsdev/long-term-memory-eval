# Codex persist-as-memory (3 cells, GCS)

GPT-4o-mini backbone, thinking off. Native Codex stays audit-only unless
`comparison_status` is `comparable`. Needs `codex-api-key`.

| Cell | Persist | Sessions | Prompt | Claim |
|------|---------|----------|--------|-------|
| 0 | off | full | `qa_workspace_v1` | retrieval-only (prompt never mentions notes) |
| 1 | on | full | `qa_workspace_persist_v1` | write ablation; session haystack still on disk |
| 2 | on | notes_only | `qa_workspace_notes_only_v1` | **memory method**: ingest, then hide `sessions/` |

`notes_only` vs PoC `teacher_session_summaries` is a campaign takeaway, not a
fourth QA cell. New experiment name → new hashed ids; first wave does not need
`--force`.

Local mock (no Codex, no API):

```powershell
python -m src.memorybench write-manifest configs/experiments/openai_codex_persist_memory.yaml
python -m src.memorybench execute-qa configs/experiments/openai_codex_persist_memory.yaml --run-index 0
```

Use `--reader mock` only on `python -m src.locomo_eval.run` smokes, not on this
live Codex YAML.

---

## GCS (copy-paste)

From repo root, conda env `distillation`. Image must include this commit
(notes prompts, ingest/hide, ledger). Redeploy after pull.

```powershell
python -m src.memorybench write-manifest configs/experiments/openai_codex_persist_memory_gcs.yaml

$env:EXPERIMENT_YAML = "configs/experiments/openai_codex_persist_memory_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1

# QA: 3 cells. New run ids; skip --force on the first wave.
gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=3 --async
# wait until all three QA _SUCCESS exist

gcloud.cmd run jobs execute memorybench-autorater --region=us-central1 --tasks=3 --async
# wait until all three autorater packs exist

gcloud.cmd run jobs execute memorybench-aggregate --region=us-central1 --tasks=1 --async
gcloud.cmd run jobs execute memorybench-collect-full --region=us-central1 --tasks=1 --async
```

If a cell must be paid again after `_SUCCESS`:

```powershell
gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=3 `
  --args="execute-qa,configs/experiments/openai_codex_persist_memory_gcs.yaml,--force" --async
```

Pull (aggregate for tables; collected for hop/notes overlay):

```powershell
$name = "locomo-openai-codex-persist-memory"
New-Item -ItemType Directory -Force -Path "experiments/$name" | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$name/aggregate" "experiments/$name/"
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$name/collected" "experiments/$name/"
```

If the PoC writers pack is not already local, pull it for the summaries takeaway:

```powershell
$writers = "locomo-openai-codex-poc-writers"
New-Item -ItemType Directory -Force -Path "experiments/$writers" | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$writers/aggregate" "experiments/$writers/"
```

Report + notebook 17 wrapper (no LLM):

```powershell
python -m src.memorybench report configs/analysis/campaign_openai_codex_persist_memory.yaml
```

Notebook: `notebooks/17_openai_codex_persist_memory_analysis.ipynb`.
Hop/qidx also land on the PoC and agents notebooks after
`python -m src.memorybench report configs/analysis/campaign_openai_codex_poc.yaml`
and `campaign_openai_agents.yaml`.
