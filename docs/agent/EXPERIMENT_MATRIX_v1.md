# Experiment matrix (v1)

**Audience:** researchers filling snapshots and deciding which Cloud Run cells to launch.  
**Machine source of truth:** YAML under `configs/experiments/` plus `configs/models/generation_catalog.yaml`.  
**Harness:** `python -m src.memorybench` (wraps `locomo_eval`; does not replace it).

This document is the fillable scientific plan. SPEC_v2 is the execution architecture.

---

## Experiment types

The harness expands whatever `experiment.type` you set. Sandwich is one design, not the only one.

| Type | YAML | What varies | What stays frozen |
|------|------|-------------|-------------------|
| `calibration` | `configs/experiments/poc.yaml`, `calibration_c01_c03.yaml` | C01–C03 memory methods on 1 conversation | cheap reader |
| `sandwich` | `sandwich_memory.yaml` | memory method | reader, prompt, judge, data |
| `sweep` | `longitudinal_core.yaml`, `ceiling_frontier.yaml` | reader (year × family) × memory | prompt, judge, **shared Mem0/RAG indexes** |
| `ablation` | `teacher_ablation.yaml` | 2026 readers × five memory conditions | prompt, judge, shared indexes except `teacher_graph` |

Do not fold a reader sweep into a sandwich YAML. The expander rejects `type: sandwich` with more than one reader.

---

## Fillable model catalog

Edit `configs/models/generation_catalog.yaml`. Replace every `TO_CONFIRM` **before** a live cell is treated as reproducible. Store snapshot ids, not moving aliases (`gpt-5.6`, `deepseek-chat`, `claude-*-latest`).

| Year | OpenAI | Anthropic | DeepSeek | Catalog ids | Status |
|------|--------|-----------|----------|-------------|--------|
| 2024 | GPT-4o | Claude 3.5 Sonnet | DeepSeek-V2.5 | `gpt-4o`, `claude-3-5-sonnet`, `deepseek-v2.5` | pin snapshots |
| 2025 | GPT-5 | Claude Sonnet 4 / 4.5 | DeepSeek-V3 | `gpt-5`, `claude-sonnet-4`, `deepseek-v3` | pin snapshots |
| 2026 | GPT-5.6 Sol | Claude Sonnet 5 | DeepSeek-V4 Pro | `gpt-5.6-sol`, `claude-sonnet-5`, `deepseek-v4-pro` | pin snapshots |
| Frontier | GPT-6 Astra | Claude Fable / Opus 5 | — | `gpt-6-astra`, `claude-fable`, `claude-opus-5` | later |

`status: runnable` means execute-qa will run. `status: to_confirm` is listed in the manifest but refused until you pin `api_model_id` + `model_snapshot` (or pass `--allow-unconfirmed`).

Each persisted cell also stores:

```text
display_name, provider, family, generation, api_model_id, model_snapshot
```

---

## Core paper matrix (27)

$$
3\ \text{years} \times 3\ \text{families} \times \{\text{full_context},\ \text{rag},\ \text{mem0}\} = 27
$$

YAML: `configs/experiments/longitudinal_core.yaml`.

Writer is **frozen**: one `mem0_locomo10` dump (`gpt-4o-mini` extract) and one `rag_locomo10` dump (`text-embedding-3-small`). Reader models change; memory formation does not. That is the shared-artifact design. Matched-model extract is a later ablation, not this matrix.

Do **not** multiply by 3 seeds. Run one pass, then measure variance on a subset such as `{GPT-4o, GPT-5.6 Sol, Sonnet 5, DS V4} × {Full, Mem0}` if decoding noise looks real. Prefer bootstrap across LoCoMo questions over paying for repeated temp-0 inference.

---

## Calibration (C01–C03)

| ID | Reader | Memory | Index |
|----|--------|--------|-------|
| C01 | `gpt-4o-mini` | `full_context` | none |
| C02 | `gpt-4o-mini` | `rag` | shared `rag_locomo10` |
| C03 | `gpt-4o-mini` | `mem0` | shared `mem0_locomo10` |

PoC plumbing (`poc.yaml`) is C01 only, mock reader, 1 conversation, 3 questions.

---

## Add-on matrices (do not contaminate the 27)

| YAML | Cells | Notes |
|------|-------|-------|
| `teacher_ablation.yaml` | 15 | 2026 readers × `{full_context, rag, mem0, session_summaries, teacher_graph}` |
| `ceiling_frontier.yaml` | 3 | Opus 5 × `{full_context, rag, mem0}` |
| `sandwich_memory.yaml` | 3 | frozen `gpt-4o-mini` × `{raw_chunks, session_summaries, full_context}` |

---

## Pipeline stages (separate Cloud Run jobs)

```text
shared indexes (once) → execute-qa (N tasks) → execute-autorater (N tasks) → aggregate
```

QA does **not** call the judge. Autorater does **not** call the answer model. That matches the Mem0 evaluation split and keeps judge tokens out of the answer traces.

One Cloud Run task = **one full LoCoMo run** for that cell. Subset fields (`max_samples`, `max_questions`) are only for PoC/calibration YAMLs.

---

## Skip vs regenerate

| Invocation | Default | `--force` |
|------------|---------|-----------|
| `python -m src.locomo_eval.run` | Always wipe the run dir and regenerate | n/a |
| `memorybench execute-qa` | If `experiments/<run_id>/_SUCCESS` exists, **skip** (no LLM) | Delete marker, locomo_eval wipes audit+parquet, regenerate |
| `memorybench execute-autorater` | If `autorater/_SUCCESS` exists, skip | Re-judge stored predictions only |

**Why skip exists:** Cloud Run may retry a task. Without skip, a successful cell would be billed twice and the answers would change.

**When you must `--force` or change `experiment.name`:** code/prompt/data change that should invalidate answers. Run ids hash the scientific QA identity (reader, memory, indexes, subset, seed), not timestamps and not the judge. Changing only the judge: `--force` on execute-autorater.

Partial failures write `errors.jsonl` + `run.json` and **do not** write `_SUCCESS`. A retry starts QA from question one (locomo_eval still has no resume).

---

## Dual artifacts

| File | Audience |
|------|----------|
| `predictions.jsonl`, `reader/`, `memory/` | human audit (unchanged locomo_eval pack) |
| `examples.parquet`, `summary.parquet` | notebooks |
| `_SUCCESS` | orchestrator skip / aggregate inclusion |

`cost_usd` is **not** written by the runner. Price tables stay in analysis.

---

## Commands

```bash
conda activate distillation
pip install -r requirements.txt

python -m src.memorybench write-manifest configs/experiments/poc.yaml
python -m src.memorybench execute-qa configs/experiments/poc.yaml --run-index 0
python -m src.memorybench execute-autorater configs/experiments/poc.yaml --run-index 0
python -m src.memorybench status configs/experiments/poc.yaml
python -m src.memorybench aggregate configs/experiments/poc.yaml
```

GCP resource list: `infra/gcp/README.md`.
