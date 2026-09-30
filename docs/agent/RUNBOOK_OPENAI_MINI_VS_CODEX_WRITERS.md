# GPT-4o-mini Chat Completions vs Codex sandwich writers (2 new cells)

Frozen GPT-4o-mini + `qa_mem0_v1` reader. New pack writes the two structured
artifacts Codex already wrote in the PoC: session summaries and Mem0g-shaped
`graph`. Codex facts stay out (no Chat Completions twin). Native Codex
is audit-only unless `comparison_status` is `comparable`.

| Cell | YAML | Writer | Method | Claim |
|------|------|--------|--------|-------|
| 0 | `openai_mini_writers_structured` | Chat Completions `gpt-4o-mini` | `session_summaries` | model-only summary writer |
| 1 | `openai_mini_writers_structured` | Chat Completions `gpt-4o-mini` | `graph` | model-only Mem0g-shaped graph writer |
| (existing) | `openai_codex_poc_writers` | Codex `gpt-4o-mini` | same two methods | model + harness |

Not dataset `session_summaries`. Not OSS `mem0g` extract/update. Needs
`OPENAI_API_KEY`. Does **not** need `codex` or `CODEX_API_KEY` (Codex pack
already exists).

Local mock (no API):

```powershell
python -m src.locomo_eval.run --config configs/writers/session_summaries.yaml --reader mock --teacher mock --max-questions 3 --run-id smoke_mini_summaries
python -m src.locomo_eval.run --config configs/writers/graph.yaml --reader mock --teacher mock --max-questions 3 --run-id smoke_mini_graph
python -m src.experiment_runner write-manifest configs/experiments/openai_mini_writers_structured.yaml
```

---

## Local live (copy-paste)

From repo root, conda env `distillation`. Two QA cells then two autorater cells.

```powershell
python -m src.experiment_runner write-manifest configs/experiments/openai_mini_writers_structured.yaml
python -m src.experiment_runner execute-qa configs/experiments/openai_mini_writers_structured.yaml --run-index 0
python -m src.experiment_runner execute-qa configs/experiments/openai_mini_writers_structured.yaml --run-index 1
python -m src.experiment_runner execute-autorater configs/experiments/openai_mini_writers_structured.yaml --run-index 0
python -m src.experiment_runner execute-autorater configs/experiments/openai_mini_writers_structured.yaml --run-index 1
python -m src.experiment_runner aggregate configs/experiments/openai_mini_writers_structured.yaml
python -m src.experiment_runner collect-full configs/experiments/openai_mini_writers_structured.yaml
```

If a cell must be paid again after `_SUCCESS`, add `--force`.

---

## GCS (copy-paste)

Image does not need the Codex binary. Redeploy after this commit.

```powershell
python -m src.experiment_runner write-manifest configs/experiments/openai_mini_writers_structured_gcs.yaml

$env:EXPERIMENT_YAML = "configs/experiments/openai_mini_writers_structured_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1

# QA: 2 cells. New run ids; skip --force on the first wave.
gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=2 --async
# wait until both QA _SUCCESS exist

gcloud.cmd run jobs execute memorybench-autorater --region=us-central1 --tasks=2 --async
# wait until both autorater packs exist

gcloud.cmd run jobs execute memorybench-aggregate --region=us-central1 --tasks=1 --async
gcloud.cmd run jobs execute memorybench-collect-full --region=us-central1 --tasks=1 --async
```

Pull both packs (new chat writers + existing Codex PoC writers). The ceiling overlay also needs the already-pulled PoC readers pack (`locomo-openai-codex-poc-readers`).

```powershell
$chat = "locomo-openai-mini-writers-structured"
$codex = "locomo-openai-codex-poc-writers"
New-Item -ItemType Directory -Force -Path "experiments/$chat" | Out-Null
New-Item -ItemType Directory -Force -Path "experiments/$codex" | Out-Null
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$chat/aggregate" "experiments/$chat/"
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/$codex/aggregate" "experiments/$codex/"
```

Report + notebook 17 wrapper (no LLM):

```powershell
python -m src.experiment_runner report configs/analysis/campaign_openai_mini_vs_codex_writers.yaml
```

Notebook: `notebooks/17_openai_mini_vs_codex_writers_analysis.ipynb`.
`methods_vs_full_context` is stuffed Chat Completions `full_context` vs persist-off
Codex workspace vs the two sandwich writers — a ceiling overlay, not a sandwich
ranking.
