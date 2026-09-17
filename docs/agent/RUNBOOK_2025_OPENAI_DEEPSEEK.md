# 2025 OpenAI vs DeepSeek runbook

**Audience:** operator running the budget 2025 campaign (GPT-5 and DeepSeek-V3 only).  
**Next:** 2026 Terra vs DeepSeek-V4 — [`docs/agent/RUNBOOK_2026_OPENAI_DEEPSEEK.md`](RUNBOOK_2026_OPENAI_DEEPSEEK.md).  
**Parked:** the three-family campaign (adds Claude Sonnet 4.5) stays in `configs/experiments/2025_readers_full_context*.yaml`, `mem0_reader_2025_writers*.yaml`, and `docs/agent/RUNBOOK_2025_LIVE.md` for when Anthropic spend is available again.

GCP bootstrap, secrets, RAG dump upload, logging filters, and failure modes: `docs/agent/RUNBOOK_2025_LIVE.md` and `docs/agent/GCP_RUNBOOK.md`. This file is only the YAML paths, cell counts, and index tables that differ.

Do **not** claim Mem0 paper Table 1–2 J. Do **not** mix these packs with the parked three-family prefixes — experiment names (and therefore `run_id`s) are different on purpose.

Windows: repo root, PowerShell, `gcloud.cmd`. Never commit `.env` or keys.

---

## What this campaign is

Same sandwich as the parked three-family campaign, **no Anthropic cells**, plus a matched **thinking on/off** axis for both families as readers and as writers. Completion-token headroom: readers off 256 / on 8192; teachers off 8192 / on 32768 (matrix overrides catalog `teacher:`). HTTP timeout for thinking-on calls is 600s. DeepSeek on-cells send `extra_body.thinking={type:enabled}` because `deepseek-chat` defaults thinking off (the V4-flash id defaults on).

| # | Claim | Cloud YAML | Cells | `--tasks` | GCS experiment name |
|---|--------|------------|-------|-----------|---------------------|
| 1 smoke | 2025 OpenAI vs DeepSeek × thinking × `full_context`, 1 conversation / 5 questions | `configs/experiments/2025_readers_openai_deepseek_smoke_gcs.yaml` | 4 | 4 | `locomo-2025-readers-openai-deepseek-smoke` |
| 2 baseline | Same readers × thinking × `{full_context, rag}`, **full LoCoMo** | `configs/experiments/2025_readers_openai_deepseek_gcs.yaml` | 8 | 8 | `locomo-2025-readers-openai-deepseek` |
| 3 writers | Frozen Mem0-parity reader × `{GPT-5, DeepSeek-V3}` × thinking × `{summaries, graph}` | `configs/experiments/mem0_reader_2025_writers_openai_deepseek_gcs.yaml` | 8 | 8 | `locomo-mem0-reader-2025-writers-openai-deepseek` |

Optional writer smoke: `configs/experiments/mem0_reader_2025_writers_openai_deepseek_smoke_gcs.yaml` (`--tasks=8`).

Local YAMLs (same cells, `storage.backend: local`): `2025_readers_openai_deepseek.yaml` and `mem0_reader_2025_writers_openai_deepseek.yaml`.

Analysis: `configs/analysis/campaign_2025_openai_deepseek.yaml` (thinking-axis recipes in `openai_deepseek_thinking_axis.yaml`). Reasoning tokens (`agent_reasoning_tokens` per question; `teacher_reasoning_tokens` on writer runs) sit next to latency so they can be compared.

Prior DeepSeek writer packs (no `teacher_thinking` in the hash) are **not** in this matrix. Do not `--force` them.

---

## Freeze / vary

| Experiment | Frozen | Varies |
|------------|--------|--------|
| 1 | LoCoMo + `qa_mem0_v1` + `full_context` + GPT-4o-mini judge | Answer model × thinking |
| 2 | Same prompt + judge + **shared** `rag_locomo10` dump | Answer model × thinking × `{full_context, rag}` |
| 3 | LoCoMo + `qa_mem0_v1` + **GPT-4o-mini reader** (thinking unset) + GPT-4o-mini judge | Writer × thinking × `{teacher_session_summaries, teacher_graph}` |

---

## Cell order (`CLOUD_RUN_TASK_INDEX`)

```powershell
python -m src.memorybench write-manifest configs/experiments/<that_gcs.yaml>
```

**Experiment 1** (smoke):

| Index | Reader | Thinking | Memory |
|-------|--------|----------|--------|
| 0 | `gpt-5` | off | `full_context` |
| 1 | `gpt-5` | on | `full_context` |
| 2 | `deepseek-chat` | off | `full_context` |
| 3 | `deepseek-chat` | on | `full_context` |

**Experiment 2** (reader × memory × thinking):

| Index | Reader | Memory | Thinking |
|-------|--------|--------|----------|
| 0 | `gpt-5` | `full_context` | off |
| 1 | `gpt-5` | `full_context` | on |
| 2 | `gpt-5` | `rag` | off |
| 3 | `gpt-5` | `rag` | on |
| 4 | `deepseek-chat` | `full_context` | off |
| 5 | `deepseek-chat` | `full_context` | on |
| 6 | `deepseek-chat` | `rag` | off |
| 7 | `deepseek-chat` | `rag` | on |

**Experiment 3** (memory × writer × thinking; reader always `gpt-4o-mini`):

| Index | Memory | Writer | Thinking |
|-------|--------|--------|----------|
| 0 | `teacher_session_summaries` | `gpt-5` | off |
| 1 | `teacher_session_summaries` | `gpt-5` | on |
| 2 | `teacher_session_summaries` | `deepseek-v3` | off |
| 3 | `teacher_session_summaries` | `deepseek-v3` | on |
| 4 | `teacher_graph` | `gpt-5` | off |
| 5 | `teacher_graph` | `gpt-5` | on |
| 6 | `teacher_graph` | `deepseek-v3` | off |
| 7 | `teacher_graph` | `deepseek-v3` | on |

Every cell hashes thinking and the matching max-token cap into `run_id`. Re-`write-manifest` before execute-qa.

---

## Cloud Run sequence

Redeploy whenever `EXPERIMENT_YAML` changes. `--tasks` equals the cell count (4 / 8 / 8). Parallelism is `8` at deploy (not an `execute` flag).

```powershell
$env:EXPERIMENT_YAML = "configs/experiments/2025_readers_openai_deepseek_smoke_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1

gcloud.cmd run jobs execute memorybench-qa         --region=us-central1 --tasks=4 --async
# wait for 4 _SUCCESS
gcloud.cmd run jobs execute memorybench-autorater  --region=us-central1 --tasks=4 --async
gcloud.cmd run jobs execute memorybench-aggregate  --region=us-central1 --tasks=1 --async
```

Then redeploy `2025_readers_openai_deepseek_gcs.yaml` with `--tasks=8`, then `mem0_reader_2025_writers_openai_deepseek_gcs.yaml` with `--tasks=8`.

Pull:

```powershell
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/locomo-2025-readers-openai-deepseek-smoke/aggregate" experiments/locomo-2025-readers-openai-deepseek-smoke/
```

Report (no LLM):

```bash
python -m src.memorybench report configs/analysis/campaign_2025_openai_deepseek.yaml
```

---

## Local

```bash
# Smoke (4 cells)
python -m src.memorybench write-manifest configs/experiments/2025_readers_openai_deepseek_smoke.yaml
python -m src.memorybench execute-qa        configs/experiments/2025_readers_openai_deepseek_smoke.yaml --run-index 0
python -m src.memorybench execute-autorater configs/experiments/2025_readers_openai_deepseek_smoke.yaml --run-index 0
python -m src.memorybench aggregate         configs/experiments/2025_readers_openai_deepseek_smoke.yaml

# Baseline (8 cells) — repeat --run-index 1..7
python -m src.memorybench write-manifest configs/experiments/2025_readers_openai_deepseek.yaml
python -m src.memorybench execute-qa        configs/experiments/2025_readers_openai_deepseek.yaml --run-index 0
```
