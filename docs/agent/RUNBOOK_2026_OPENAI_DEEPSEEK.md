# 2026 OpenAI vs DeepSeek runbook

**Audience:** operator running the 2026 campaign (GPT-5.6 Terra and DeepSeek-V4 only).  
**Not in this matrix:** Claude Fable 5.1. Costed below as a parked third family; do not launch Anthropic cells here.

GCP bootstrap, secrets, RAG dump upload, logging filters, and failure modes: `docs/agent/RUNBOOK_2025_LIVE.md` and `docs/agent/GCP_RUNBOOK.md`. This file is only the YAML paths, cell counts, index tables, and a token-based cost estimate.

Do **not** claim Mem0 paper Table 1–2 J. Do **not** mix these packs with the 2025 prefixes — experiment names (and therefore `run_id`s) are different on purpose.

Windows: repo root, PowerShell, `gcloud.cmd`. Never commit `.env` or keys.

---

## What this campaign is

Same sandwich as `docs/agent/RUNBOOK_2025_OPENAI_DEEPSEEK.md`, including matched **thinking on/off**. The 2026 pair is **GPT-5.6 Terra** vs **DeepSeek-V4** (catalog `deepseek-v4` → API `deepseek-v4-flash`, served as V4.1-Flash). Headroom: readers off 256 / on 8192; teachers off 8192 / on 32768. DeepSeek on/off still sends `extra_body.thinking` explicitly (V4-flash defaults on; the 2025 `deepseek-chat` alias defaults off).

| # | Claim | Cloud YAML | Cells | `--tasks` | GCS experiment name |
|---|--------|------------|-------|-----------|---------------------|
| 1 smoke | 2026 OpenAI vs DeepSeek × thinking × `full_context`, 1 conversation / 5 questions | `configs/experiments/2026_readers_openai_deepseek_smoke_gcs.yaml` | 4 | 4 | `locomo-2026-readers-openai-deepseek-smoke` |
| 2 baseline | Same readers × thinking × `{full_context, rag}`, **full LoCoMo** | `configs/experiments/2026_readers_openai_deepseek_gcs.yaml` | 8 | 8 | `locomo-2026-readers-openai-deepseek` |
| 3 writers | Frozen Mem0-parity reader × `{GPT-5.6 Terra, DeepSeek-V4}` × thinking × `{summaries, graph}` | `configs/experiments/mem0_reader_2026_writers_openai_deepseek_gcs.yaml` | 8 | 8 | `locomo-mem0-reader-2026-writers-openai-deepseek` |

Optional writer smoke: `configs/experiments/mem0_reader_2026_writers_openai_deepseek_smoke_gcs.yaml` (`--tasks=8`).

Local YAMLs (same cells, `storage.backend: local`): `2026_readers_openai_deepseek.yaml` and `mem0_reader_2026_writers_openai_deepseek.yaml`.

Analysis: `configs/analysis/campaign_2026_openai_deepseek.yaml` (thinking-axis recipes in `openai_deepseek_thinking_axis.yaml`).

Shared `rag_locomo10` dump does **not** need a re-upload.

---

## Freeze / vary

| Experiment | Frozen | Varies |
|------------|--------|--------|
| 1 | LoCoMo + `qa_mem0_v1` + `full_context` + GPT-4o-mini judge | Answer model × thinking |
| 2 | Same prompt + judge + **shared** `rag_locomo10` dump | Answer model × thinking × `{full_context, rag}` |
| 3 | LoCoMo + `qa_mem0_v1` + **GPT-4o-mini reader** + GPT-4o-mini judge | Writer × thinking × `{session_summaries, graph}` |

---

## Cell order (`CLOUD_RUN_TASK_INDEX`)

```powershell
python -m src.experiment_runner write-manifest configs/experiments/<that_gcs.yaml>
```

**Experiment 1** (smoke):

| Index | Reader (API id) | Thinking | Memory |
|-------|-----------------|----------|--------|
| 0 | `gpt-5.6-terra` | off | `full_context` |
| 1 | `gpt-5.6-terra` | on | `full_context` |
| 2 | `deepseek-v4-flash` | off | `full_context` |
| 3 | `deepseek-v4-flash` | on | `full_context` |

**Experiment 2** (reader × memory × thinking):

| Index | Reader (API id) | Memory | Thinking |
|-------|-----------------|--------|----------|
| 0 | `gpt-5.6-terra` | `full_context` | off |
| 1 | `gpt-5.6-terra` | `full_context` | on |
| 2 | `gpt-5.6-terra` | `rag` | off |
| 3 | `gpt-5.6-terra` | `rag` | on |
| 4 | `deepseek-v4-flash` | `full_context` | off |
| 5 | `deepseek-v4-flash` | `full_context` | on |
| 6 | `deepseek-v4-flash` | `rag` | off |
| 7 | `deepseek-v4-flash` | `rag` | on |

**Experiment 3** (memory × writer × thinking; reader always `gpt-4o-mini`):

| Index | Memory | Writer (API id) | Thinking |
|-------|--------|-----------------|----------|
| 0 | `session_summaries` | `gpt-5.6-terra` | off |
| 1 | `session_summaries` | `gpt-5.6-terra` | on |
| 2 | `session_summaries` | `deepseek-v4-flash` | off |
| 3 | `session_summaries` | `deepseek-v4-flash` | on |
| 4 | `graph` | `gpt-5.6-terra` | off |
| 5 | `graph` | `gpt-5.6-terra` | on |
| 6 | `graph` | `deepseek-v4-flash` | off |
| 7 | `graph` | `deepseek-v4-flash` | on |

Every cell hashes thinking and the matching max-token cap into `run_id`. Re-`write-manifest` before execute-qa.

---

## Cloud Run sequence

Redeploy whenever `EXPERIMENT_YAML` changes. `--tasks` equals the cell count (4 / 8 / 8). Parallelism is `8` at deploy (not an `execute` flag).

```powershell
$env:EXPERIMENT_YAML = "configs/experiments/2026_readers_openai_deepseek_smoke_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1

gcloud.cmd run jobs execute memorybench-qa         --region=us-central1 --tasks=4 --async
# wait for 4 _SUCCESS
gcloud.cmd run jobs execute memorybench-autorater  --region=us-central1 --tasks=4 --async
gcloud.cmd run jobs execute memorybench-aggregate  --region=us-central1 --tasks=1 --async
```

Then redeploy `2026_readers_openai_deepseek_gcs.yaml` with `--tasks=8`, then `mem0_reader_2026_writers_openai_deepseek_gcs.yaml` with `--tasks=8`.

Pull:

```powershell
gcloud.cmd storage cp -r "gs://$env:BUCKET/experiments/locomo-2026-readers-openai-deepseek-smoke/aggregate" experiments/locomo-2026-readers-openai-deepseek-smoke/
```

Report (no LLM):

```bash
python -m src.experiment_runner report configs/analysis/campaign_2026_openai_deepseek.yaml
```

---

## Local

```bash
# Smoke (4 cells)
python -m src.experiment_runner write-manifest configs/experiments/2026_readers_openai_deepseek_smoke.yaml
python -m src.experiment_runner execute-qa        configs/experiments/2026_readers_openai_deepseek_smoke.yaml --run-index 0
python -m src.experiment_runner execute-autorater configs/experiments/2026_readers_openai_deepseek_smoke.yaml --run-index 0
python -m src.experiment_runner aggregate         configs/experiments/2026_readers_openai_deepseek_smoke.yaml

# Baseline (8 cells) — repeat --run-index 1..7
python -m src.experiment_runner write-manifest configs/experiments/2026_readers_openai_deepseek.yaml
python -m src.experiment_runner execute-qa        configs/experiments/2026_readers_openai_deepseek.yaml --run-index 0
```

---

## Cost estimate (API tokens only)

Reusable: `configs/models/pricing.yaml` + `cost:` in `configs/analysis/campaign_2026_openai_deepseek.yaml`. The same tables appear in notebooks 11–14 (pre-test estimate, post-test actual).

```bash
python -m src.experiment_runner report configs/analysis/campaign_2026_openai_deepseek.yaml
```

Method: 2025 OpenAI vs DeepSeek `cost.json` token volumes (1986 questions / cell; 272 teacher calls / writer cell) × list prices as of 2026-09-16. Not an invoice. Cloud Run CPU/RAM is extra. The dollar figures below are **thinking-off style volumes**; thinking-on cells can spend up to the 8192 completion cap on reasoning, so re-price after smoke. Assumes **no prompt-cache hits**, Terra **short-context** rates (full-context mean prompt ≈ 28k ≪ 272k), DeepSeek **cache-miss**.

| Rate pin | Input / 1M | Output / 1M |
|----------|------------|-------------|
| GPT-5.6 Terra (standard, short) | $2.00 | $12.00 |
| DeepSeek-V4.1-Flash off-peak | $0.15 | $0.60 |
| DeepSeek-V4.1-Flash peak | $0.30 | $1.20 |
| GPT-4o-mini (frozen reader + judge) | $0.15 | $0.60 |
| Claude Fable 5.1 (parked; not launched) | $10.00 | $50.00 |

| Stage | Off-peak (thinking-off volume) | DeepSeek peak |
|-------|----------|---------------|
| 1 smoke (4 × full_context, 5 q) | **order ~$0.60** | order ~$0.64 |
| 2 baseline (8 cells, full LoCoMo) | **order ~$250** before thinking-on extra | higher |
| 3 writers (8 cells) | **order ~$24** before thinking-on extra | similar |
| Optional writer smoke | +~$1.70 | +~$1.70 |

Baseline is still almost entirely Terra × `full_context`. RAG cells are a few dollars. Writers stay cheap because the frozen `gpt-4o-mini` reader is small; teacher thinking-on is the new spend.

**Claude Fable 5.1 (do not add):** parked counterfactual in `cost.parked` is **~$632** at the previous four-cell volumes (includes judge + RAG + writer Terra cells at $10/$50). Adaptive thinking is always on (default effort `high`), unlike Terra `reasoning_effort=none` on off-cells.

Sources: [OpenAI pricing](https://developers.openai.com/api/docs/pricing), [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing), [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
