# Trace: 2026-08-18 — before renaming preprocess modules

Snapshot of AGENTS.md and HUMANS.md. Rename ConversationIngestor / ingestor.py
to DataIngestor / data_ingestor.py, and ConversationPreprocessor / pipeline.py
to PreprocessingPipeline / preprocessing_pipeline.py.

---

# AGENTS.md (previous)

# AGENTS.md — agent operating notes (v0.1)

**Audience:** coding agents working in this repo.  
**Length target:** 2–3 pages.  
**Update rule:** change this file on every meaningful behavior or layout change; snapshot to `docs/agent/traces/` first.

---

## Mission

Research pipeline for long-term conversational memory on **LoCoMo**, eventually multi-teacher memory construction with a **sandwich design** (fixed data + fixed answer/eval; variable middle = memory method).

**Current phase:** end-to-end read path with **draft C0 vs C1** memory builders.  
HLD **(i) pre-processing** exists as `src/locomo_eval/preprocess/` (session blocks + persist helper) and is **not wired** into `run.py`.  
See `docs/reports/engineering_notebook.md` for freeze/extend rules.

Do **not** implement multi-teacher fusion or claim schema unless the human expands scope. `teacher_orchestrator.py` is a passthrough seam (one session block at a time, no LLM).

---

## North star (later)

```
Write: conversation → teachers → fusion/validate → memory store
Read:  question → retrieval → fixed answer LLM → LoCoMo evaluator
```

Conditions planned: C0 raw/chunk, C1 single teacher, C2 top-1 routing, C3 whole-memory aggregation, C4 claim-level fusion.

**Now:** `c0_raw` and `c1_session_summary` builders inject alternative `Memory.text` with frozen reader/metrics. C1 draft uses LoCoMo-provided summaries (not a live teacher API yet).

---

## Repo map (v0.1)

| Path | Role |
|------|------|
| `configs/baseline.yaml` | CLI default; currently C1. Standalone (not a parent of c0/c1) |
| `configs/c0_raw.yaml` | C0 raw dialog memory |
| `configs/c1_session_summary.yaml` | C1 session-summary memory (same condition as baseline.yaml) |
| `prompts/qa_v1.txt` | Fixed answer prompt |
| `docs/reports/engineering_notebook.md` | System map / extension points |
| `docs/schemas/memory_runtime.md` | Runtime `{memory}` audit |
| `docs/schemas/preprocess_runtime.md` | Session-block preprocess schema (`preprocess_io.v1`) |
| `src/locomo_eval/` | Baseline package |
| `src/metrics/locomo_qa.py` | Official LoCoMo category F1 |
| `data/raw/locomo10.json` | Dataset (gitignored; fetch) |
| `experiments/<run_id>/` | Human-auditable run pack |
| `docs/agent/SPEC_v1.md` | Phase 1 requirements |
| `docs/agent/HUMANS.md` | Human-facing brief |
| `docs/agent/traces/` | Doc version history |

### Package modules (`src/locomo_eval/`)

| File | Responsibility |
|------|----------------|
| `schemas.py` | Conversation, SessionBlock, Memory, Prediction |
| `dataset.py` | Load LoCoMo JSON → Conversation (C0/C1 read path) |
| `preprocess/` | HLD (i): ConversationIngestor + ConversationPreprocessor (unwired from run.py) |
| `teacher_orchestrator.py` | HLD (ii) stub: one SessionBlock at a time, passthrough, no LLM |
| `memory.py` | MemoryBuilder interface + C0/C1 |
| `prompts.py` | Load/render prompt text |
| `readers.py` | OpenAI + Mock readers, temp=0 |
| `utils/llm_response_cache.py` | LLM reply memo (implemented; **not wired** into run.py — future optimization) |
| `metrics.py` | EM, token F1, LoCoMo F1 |
| `report.py` | JSONL/CSV/plots |
| `run.py` | CLI: one memory YAML → one audit pack (`run_locomo_pipeline_with_memory_config`; compare is a separate script) |
| `offline_evaluate.py` | CLI: rescore stored predictions with string metrics only (no API, not an LLM autorater) |

---

## Commands agents should use

```bash
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py

# Offline smoke (no API key)
python -m src.locomo_eval.run --config configs/baseline.yaml --reader mock --max-questions 5 --run-id smoke_mock

# Live OpenAI (needs OPENAI_API_KEY in repo-root .env or shell)
python -m src.locomo_eval.run --config configs/baseline.yaml --max-questions 3 --run-id smoke_openai

# Full baseline (costly)
python -m src.locomo_eval.run --config configs/baseline.yaml --run-id baseline_session_summary

# Offline rescore (string metrics only; not an LLM autorater)
python -m src.locomo_eval.offline_evaluate --predictions experiments/<run_id>/predictions.jsonl

# Unit tests — preprocess (HLD i) + evaluation (HLD iv) + sandwich regression locks
python -m unittest tests/test_preprocessing_pipeline.py tests/test_evaluation_pipeline.py tests/test_regressions.py
```

Set API key via repo-root `.env` (`copy .env.example .env`) or shell `OPENAI_API_KEY`.  
Never commit keys or `.env`. `src/locomo_eval/env.py` loads `.env` at run start / OpenAI reader init.

---

## Run audit package (always write)

Each run under `experiments/<run_id>/` must include:

- `predictions.jsonl` — one row per question (incl. memory text)
- `predictions.csv` — spreadsheet-friendly + scores + memory preview
- `metrics.json` — overall + by-category
- `metrics_by_category.csv`
- `run_meta.json` — model, prompt, data hash, git hash, timestamp
- `plots/` — overall + category bars

Agents must not silently skip CSV/plots when code paths change.

---

## Design rules for agents

1. **Sandwich:** only change one middle variable per experimental claim later. v0.1 keeps reader prompt and metrics fixed.
2. **Orchestrator is software**, not one giant LLM call. `teacher_orchestrator.py` is a thin file (iterate session blocks). Do not grow a `write/` package until teachers generate memory.
3. **Prefer small pure functions** over frameworks.
4. **Keep metrics dual-reported:** SPEC token F1/EM *and* LoCoMo category F1.
5. **Do not wire `LlmResponseCache` yet.** Implementation lives in `src/locomo_eval/utils/llm_response_cache.py` (future optimization after E2E is trusted). Per-run resume is `predictions.jsonl`. Do not delete user caches unless asked.
6. **Plain YAML**, plain JSON loaders, local CSV — no Hydra/W&B required. Each config file is standalone; `pipeline.memory` is a builder id, not another YAML.
7. **Update docs:** after behavior change, copy previous AGENTS/HUMANS into `docs/agent/traces/YYYY-MM-DD_topic.md`, then edit live files.

---

## Code, tests, and comments

From review. Follow these when adding or renaming code.

**Names.** Unambiguous, justified, and kept current. If a name no longer matches the behavior, rename it (e.g. a generic orchestrator must not be called `run_baseline` when “baseline” is a config). Do not overload research terms across different mechanisms (`LlmResponseCache` vs JSONL resume; baseline YAML vs C0/C1).

**Comments.** Explain *why* and the surrounding context, not a restatement of the next line. Ambiguous helpers need a one-liner on what they pin for later reproduction (`_git_hash`, `_file_sha256`). Distinguish lookalike layers (`LlmResponseCache` vs JSONL resume; `offline_evaluate.py` vs `run.py`; later LLM autoraters vs this string scorer).

**Schemas / data-model classes.** The module docstring should map how types connect and which pipeline step uses them (load → memory → reader → prediction → score). Each class gets a short “what it is / who consumes it” note. Label gold answers as scorer-only (never in the reader prompt).

**Tests.**
- One unit test focuses on one function (`exact_match` tests stay separate from `token_f1` tests).
- Test names include the behavior **and** the expected outcome, e.g. `test_exact_match_returns_one_when_answers_match_after_normalization`.
- Group related cases in a `TestCase` per function or class; do not pile unrelated functions into one method.

**Experiments.** One YAML per `run_locomo_pipeline_with_memory_config` call (`python -m src.locomo_eval.run`). Compare C0 vs C1 with two runs, then `scripts/compare_runs.py` (no API). Do not fold A vs B into `run.py`.

---

## Out of scope (v0.1)

Multi-teacher, claim fusion, validator loop, retrieval budgets as experiments, training/distillation loop, web UI, event-summarization / multimodal tasks.

Stub remnants (`src/train.py`, `src/distill/`) are deferred KD; do not wire unless requested.

---

## Acceptance checklist for agent PRs

- [ ] Dataset loads without editing source JSON  
- [ ] One-question and full-run share the same command  
- [ ] Predictions JSONL deterministic fields  
- [ ] Metrics include EM, token F1, LoCoMo F1 by category  
- [ ] Memory builder swappable without changing reader/evaluator  
- [ ] Tests for parse, memory, preprocess session blocks, normalize; names = behavior + expected outcome; one function per unit test  
- [ ] `tests/test_regressions.py` still green (pipeline / sandwich contracts)  
- [ ] AGENTS.md + HUMANS.md updated + trace snapshot  

---

## Traceability

- Spec: `docs/agent/SPEC_v1.md`  
- LoCoMo pin: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`  
- Paper: Maharana et al., arXiv:2402.17753  
- Sandwich idea: Bowman et al. 2022 scalable oversight  

---

# HUMANS.md (previous)

# HUMANS.md — how to run and review (v0.1)

**Audience:** you (the researcher).  
**Length target:** 2–3 pages.  
**Update rule:** when workflows or outputs change, snap a copy into `docs/agent/traces/`, then edit this file.

---

## What this project is

You compare **how memory is built** for long multi-session chats (LoCoMo). The eventual design is an experimental **sandwich**:

| Layer | Fixed or variable? | Notes |
|-------|--------------------|--------|
| Top | Fixed | LoCoMo conversations + questions |
| Middle | Variable (later C0–C4) | How candidate memory is constructed / fused |
| Bottom | Fixed | Retrieval budget, answer LLM + prompt, metrics |

**v0.1 goal:** ship the **bottom + I/O skeleton** using LoCoMo’s own **session summaries** as memory, answered by one fixed OpenAI model, scored with LoCoMo-compatible metrics, and with **auditable logs/plots**.

You are not yet running multi-teacher fusion. **C0** (`c0_raw`) vs **C1** (`c1_session_summary`) compare how much structure helps under one fixed answer model. Full system map: `docs/reports/engineering_notebook.md`.

HLD **(i) pre-processing** can now emit ordered **session blocks** (stable `turn_id`s, speaker/time normalization) via `src/locomo_eval/preprocess/`. That path is **not** in `python -m src.locomo_eval.run` yet — C0/C1 still read `Conversation` from `dataset.py`. A thin `teacher_orchestrator.py` walks one session block at a time as a passthrough (no teacher LLM).

---

## Mental model of the data & task

### What LoCoMo gives you (source of truth)

[`data/raw/locomo10.json`](../../data/README.md) is the **official LoCoMo release** (fetched, not invented here). Each conversation sample roughly includes:

| Field | Role |
|-------|------|
| `conversation` | Multi-session dialog turns (speakers, text, times, optional image captions) |
| `session_summary` | Precomputed session-level summaries (also in the release) |
| `observation` | Precomputed observations (RAG-style material in the original paper) |
| `event_summary` | Event-graph style annotations (summarization task; not our Phase‑1 focus) |
| `qa` | **Questions + gold (reference) answers**, category, and often `evidence` dialog IDs |

So yes: the raw file holds **input dialog history**, **authoring aid fields** (summaries/observations), and the **correct answers** needed to score systems. It is a *benchmark package*, not “questions only.”

### What `qa_all` is (ours, processed)

`data/processed/qa_all.csv` / `.jsonl` come from **`scripts/prepare_data.py`**, which **flattens** LoCoMo for inspection. They are not a second official LoCoMo product. Rows include:

- question text, **gold answer**, category, evidence IDs  
- optional conversation context / previews for human browsing  

Treat the CSV as a **readable export**. Training-time “unmasking” is decided by the run pipeline, not by the CSV columns existing on disk.

### Two different “evaluators” (do not conflate)

```text
                    ┌─────────────────────────────────────┐
  gold answer ─────►│  SCORER (metrics.py / LoCoMo F1)    │  ← sees reference
  model answer ────►│  compares pred vs gold              │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
  Memory.text ─────►│  ANSWER LLM (OpenAI / Claude later) │  ← must NOT see gold
  question only ───►│  fixed prompt qa_v1                 │
                    └─────────────────────────────────────┘
```

1. **Answer model (what we usually mean by “running the eval”)**  
   Blind to the gold answer. In v0.1 the prompt only injects **`{memory}` + `{question}`** (`prompts/qa_v1.txt`). It does **not** currently receive the gold answer, category ID, or evidence list as separate fields. Dates appear only if they are **already inside** the memory string (e.g. session headers / summary prose).

2. **Scorer (after the model answers)**  
   *Does* see the reference answer (and category, so LoCoMo’s category-aware F1 can apply). That is not “cheating”; it is standard supervised scoring. Re-run it offline with `python -m src.locomo_eval.offline_evaluate`. A later **LLM autorater** (model grades the answer) would be a different module — do not put it here.

So we do not literally “mask columns on the dataset file.” We **construct a restricted prompt** from selected fields and never put `answer` into that prompt.

### What memory design is for

**Memory design = how we build the `Memory.text` (and later structure/schema) that is glued to the question for the answer model.**

```text
LoCoMo conversation  ──►  MemoryBuilder (C0 / C1 / …)  ──►  Memory.text
                                                                  │
question  ────────────────────────────────────────────────────────┤
                                                                  ▼
                                              fixed prompt → fixed LLM → pred
                                                                  │
gold answer ──────────────────────────────────────────────────────► scorer
```

- **C0** today: dump raw dialog as the memory string (little structure).  
- **C1** draft: dump released session summaries (structured-ish, still not a multi-teacher schema).  
- **Later (C2–C4):** teacher outputs, fusion, claim-level schema — still the same idea: **produce a better memory payload for the same fixed Q + fixed answer LLM.**

The experimental claim is almost always: *under a frozen answer model and prompt, does condition A’s memory make QA better than condition B’s?*

---

## One-time setup (conda)

```bash
conda create -n distillation python=3.11 -y
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py
```

Data lands at `data/raw/locomo10.json` (not committed; CC BY-NC 4.0).

For live API runs, put the key in a **repo-root `.env`** (gitignored):

```bash
copy .env.example .env
# edit .env → OPENAI_API_KEY=sk-...
```

The pipeline loads `.env` automatically on start. Shell export still works and wins if already set.

---

## Compare C0 vs C1

Each `python -m src.locomo_eval.run` call is **one** memory config (`run_locomo_pipeline_with_memory_config`). Freeze answer model + prompt; swap only the memory YAML; then compare the two run folders:

```bash
python -m src.locomo_eval.run --config configs/c0_raw.yaml --max-questions 20 --run-id cmp_c0_n20
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --max-questions 20 --run-id cmp_c1_n20
python scripts/compare_runs.py --runs experiments/cmp_c0_n20 experiments/cmp_c1_n20 --out experiments/compare_c0_c1
```

Open `experiments/compare_c0_c1/overall.csv` and the two `predictions.csv` files side by side (same `question_id`).

Evaluation-pipeline unit tests (string metrics / LoCoMo F1) live in `tests/test_evaluation_pipeline.py`. Preprocess session-block tests are `tests/test_preprocessing_pipeline.py`. Sandwich contracts that must not drift are locked in `tests/test_regressions.py` (mock only):

```bash
python -m unittest tests/test_preprocessing_pipeline.py tests/test_evaluation_pipeline.py tests/test_regressions.py
```

---

## Run the baseline

**Offline smoke (no money / no key):**

```bash
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --reader mock --max-questions 5 --run-id smoke_mock
```

**Small live check:**

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --max-questions 3 --run-id smoke_openai
```

**Full QA set** (many API calls; resume unfinished questions via `predictions.jsonl`):

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --run-id baseline_session_summary
```

Default model is `gpt-4.1-mini` in `configs/baseline.yaml`. For a stronger fixed answer model (your intended GPT-4.1 class):

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --model gpt-4.1 --run-id baseline_gpt41
```

Config knobs live only in YAML + CLI overrides — no hidden flags.

---

## Where to look after a run

Everything for one experiment is under `experiments/<run_id>/`:

| File | Why open it |
|------|-------------|
| `predictions.csv` | Spreadsheet audit: Q, gold, pred, scores, memory clip |
| `predictions.jsonl` | Full rows including full memory text |
| `metrics.json` | Overall exact match / token F1 / **LoCoMo F1** |
| `metrics_by_category.csv` | Category breakdown (single-hop, temporal, …) |
| `run_meta.json` | Model, prompt version, data SHA, git hash, time |
| `memory/` | Runtime memory audit (`schema.json`, full texts) — see [`docs/schemas/memory_runtime.md`](../schemas/memory_runtime.md) |
| `plots/*.png` | Quick visual of overall + by-category scores |

Two stores that are easy to mix up:

- **`predictions.jsonl`** — per-run resume (skip finished questions). This is what the live pipeline uses.
- **`experiments/cache/`** — `LlmResponseCache` (under `src/locomo_eval/utils/`). Implemented, **not wired** into `run.py` until E2E validation is done. Re-enable by passing a cache to `get_reader`.

Recompute string metrics without re-calling the API (not an LLM autorater):

```bash
python -m src.locomo_eval.offline_evaluate --predictions experiments/<run_id>/predictions.jsonl
```

---

## How to read the metrics

- **exact_match / token_f1** — simple string overlap (SPEC_v1). Good for debugging.  
- **locomo_f1** — category-aware F1 matching the released LoCoMo eval protocol (what you want for paper comparisons). Category 5 (adversarial) expects phrases like “not mentioned” / “no information available”.

If LoCoMo F1 is low but answers “feel” right, check:

1. session summaries incomplete vs evidence turns,  
2. model verbosity vs short gold,  
3. adversarial category instructions in prompt.

---

## Mental model of the code (reviewable path)

```
locomo10.json                 # official: dialog + summaries + gold QA
    → dataset.py              # Conversation / Question (C0/C1 read path; gold scorer-only)
    → preprocess/             # HLD (i): SessionBlock[] (not wired into run.py yet)
    → teacher_orchestrator.py # HLD (ii): one session block, passthrough, no LLM
    → memory.py               # experimental: Memory.text (gold never enters here for the LLM)
    → prompts/qa_v1.txt       # fixed: Memory + Question only
    → readers.py              # fixed answer LLM (no LlmResponseCache in this phase)
    → metrics + report        # scorer uses gold; reports for humans
```

Later swaps should only replace **memory builders** (and eventually teacher/fusion), not the answer prompt or scorer, if you want clean sandwich comparisons.

---

## Roadmap sketch (not in v0.1)

| ID | Middle layer | Question |
|----|--------------|----------|
| C0 | Raw / chunked dialog | Does structure help? |
| C1 | Single teacher memory | Standard pipeline strength |
| C2 | Top-1 of K teachers | Selection enough? |
| C3 | Aggregate whole memories | Synthesis enough? |
| C4 | Claim-level fusion + validation | Evidence-grounded fusion win? |

Teacher K∈{1,2,3} and utility U_K = Δscore / Δcost come **after** this baseline is trustworthy.

---

## What you should ask agents to do (and not)

**Do:** extend memory builders, improve reports, add Claude reader when you switch fixed answer model, fix bugs, keep docs current. Next slice can wire `write_preprocess_run_log` into `experiments/<run_id>/preprocess/`.

**Don’t (yet):** multi-LLM fusion, training loops, silent metric changes, deleting `experiments/cache/` or data, committing secrets.

---

## Doc versioning

Live: `docs/agent/AGENTS.md`, `docs/agent/HUMANS.md`, `docs/agent/SPEC_v1.md`  
History: `docs/agent/traces/` (dated snapshots when these change)  
