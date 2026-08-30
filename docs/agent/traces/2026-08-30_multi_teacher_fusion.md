# AGENTS.md — agent operating notes (v0.1)

**Audience:** coding agents working in this repo.  
**Length target:** 2–3 pages.  
**Update rule:** change this file on every meaningful behavior or layout change; snapshot to `docs/agent/traces/` first.

---

## Mission

Research pipeline for long-term conversational memory on **LoCoMo**, eventually multi-teacher memory construction with a **sandwich design** (fixed data + fixed answer/eval; variable middle = memory method).

**Current phase:** HLD (i) session-block preprocess plus a **Mem0 / Mem0g write-index** (`extract → ADD/UPDATE/DELETE/NONE`, optional in-memory graph). Read path still has `raw_chunks` / `session_summaries` / `teacher_session_summaries`, plus a Mem0-style **LLM autorater** over finished predictions. Do not claim paper J (66.88 / 68.44) from the write-index alone.
See `docs/reports/engineering_notebook.md` for freeze/extend rules.

Do **not** implement multi-teacher fusion unless the human expands scope. Later distilled memory is a new `GraphMemory` subclass (freeze extract when attributing the graph). `teacher_orchestrator.py` stays a passthrough seam.

---

## North star (later)

```
Write: conversation → teachers → fusion/validate → memory store
Read:  question → retrieval → fixed answer LLM → LoCoMo evaluator
```

Conditions planned: `raw_chunks`, `session_summaries`, `teacher_session_summaries`, `mem0`, `mem0g`, later `top1_teacher`, `whole_memory_aggregation`, `claim_fusion`. Use these ids in logs — do not number conditions C0, C1, …

**Now:** Default **reader** is Mem0-parity (`gpt-4o-mini` + `prompts/qa_mem0_v1.txt`) for `raw_chunks`, `session_summaries`, `mem0`, and `mem0g`. `mem0_baseline.yaml` pins Mem0's released answer/evaluation controls (GPT-4o-mini, Mem0 answer prompt, GPT-4o-mini judge); that YAML is evaluation parity over `session_summaries` memory, not a substitute for the write-index. Default **writer** for mem0 extract/update is `gpt-4o-mini`. Deterministic preprocess dump (`python -m src.locomo_eval.preprocess.run_index`) is the no-LLM write path for `raw_chunks` / `session_summaries`. `mem0` / `mem0g` load a write-index dump and cosine-retrieve top-k NL facts (`top_k=30`). Graph/store RQs freeze writer + reader + this retriever; swap only `GraphMemory`. `teacher_session_summaries` is a live single-teacher seam. Reader-model or `qa_v1` swaps remain a **separate** robustness axis.

---

## Repo map (v0.1)

| Path | Role |
|------|------|
| `configs/mem0_baseline.yaml` | CLI default; Mem0-parity reader controls over `session_summaries` memory |
| `configs/raw_chunks.yaml` | Raw dialog memory (`raw_chunks`) |
| `configs/session_summaries.yaml` | LoCoMo session-summary memory with original QA controls |
| `configs/session_summaries_reader_gpt56_luna.yaml` | `session_summaries` memory + GPT-5.6 Luna answer model |
| `configs/teacher_session_summaries.yaml` | Live single-teacher memory (`teacher_session_summaries`) |
| `configs/preprocess.yaml` | Deterministic HLD (i) write-index (no LLM) |
| `configs/mem0.yaml` | Mem0 vector write-index + load/retrieve seam |
| `configs/mem0g.yaml` | Mem0 vector + in-memory graph (`mem0g`) |
| `prompts/qa_mem0_v1.txt` | Pinned released Mem0 answer prompt for baseline parity |
| `prompts/qa_v1.txt` | Alternate short prompt; override only as a separate bottom-layer axis |
| `prompts/teacher_session_v1.txt` | Teacher session-summary prompt |
| `prompts/autorater_mem0_v1.txt` | Mem0 LLM-as-a-Judge prompt |
| `prompts/mem0_extract_v1.txt` / `mem0_update_v1.txt` | Mem0 fact extract + ADD/UPDATE/DELETE/NONE (pin: mem0 @ ece7ff6b) |
| `prompts/mem0g_*.txt` | Entity / relation / conflict (pin: mem0 graph @ 69a832dc) |
| `docs/reports/engineering_notebook.md` | System map / extension points |
| `docs/schemas/memory_runtime.md` | Runtime `{memory}` audit |
| `docs/schemas/preprocess_runtime.md` | Session-block preprocess schema (`preprocess_io.v1`) |
| `docs/schemas/mem0_index.md` | Write-index dump schema (`mem0_index.v1`) |
| `src/locomo_eval/` | Baseline package |
| `scripts/compare_full_runs.py` | Sandwich report for finished run packs; infers frozen prompt from run metadata |
| `scripts/analysis/` | Reusable analyses + plots (`run_benchmark` calls the autorater API unless mock) |
| `scripts/analysis/compare_predictions.py` | Two prediction JSONLs → paired LoCoMo F1 boxplot + histograms (kernel used by compare_full_runs) |
| `scripts/analysis/run_benchmark.py` | Finished prediction pack → Mem0 F1/BLEU-1/J + literature tables, histograms, boxplots, latency plots |
| `src/metrics/locomo_qa.py` | Official LoCoMo category F1 |
| `data/raw/locomo10.json` | Dataset (gitignored; fetch) |
| `experiments/<run_id>/` | Human-auditable run pack |
| `docs/agent/SPEC_v1.md` | Phase 1 requirements |
| `docs/agent/HUMANS.md` | Human-facing brief |
| `docs/agent/traces/` | Doc version history |

### Package modules (`src/locomo_eval/`)

| File | Responsibility |
|------|----------------|
| `schemas.py` | Conversation, Session, Turn, SessionBlock, Question, Memory, Prediction |
| `dataset.py` | Load LoCoMo JSON → Conversation (read path; preprocess is separate) |
| `preprocess/` | HLD (i): `DataIngestor` + `PreprocessingPipeline` + session-document join + deterministic `run_index` dump |
| `teacher_orchestrator.py` | HLD (ii) stub: one SessionBlock at a time, passthrough, no LLM |
| `memory.py` | MemoryBuilder interface + raw_chunks / session_summaries / teacher_session_summaries / mem0 / mem0g |
| `mem0/` | Write-index: ingest pairs, extract, update, vector store, `GraphMemory` ABC, dump, `run_index` CLI |
| `prompts.py` | Load/render prompt text |
| `readers.py` | OpenAI + Mock readers, temp=0 |
| `models.py` | Model ids / families / Chat Completions kwargs |
| `teachers.py` | Single teacher (session summaries) |
| `autorater.py` | Mem0-style CORRECT/WRONG LLM judge; GPT-4o + mock |
| `mem0_metrics.py` | Mem0 lexical F1/BLEU-1 and latency summaries |
| `mem0_baselines.py` | Published Mem0 Tables 1–2 values (literature pins, not re-runs) |
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
python -m src.locomo_eval.run --config configs/mem0_baseline.yaml --reader mock --max-questions 5 --run-id smoke_mock

# Live OpenAI (needs OPENAI_API_KEY in repo-root .env or shell)
python -m src.locomo_eval.run --config configs/mem0_baseline.yaml --max-questions 3 --run-id smoke_openai

# Full baseline (costly)
python -m src.locomo_eval.run --config configs/mem0_baseline.yaml --run-id mem0_baseline_session_summary

# Offline rescore (string metrics only; not an LLM autorater)
python -m src.locomo_eval.offline_evaluate --predictions experiments/<run_id>/predictions.jsonl

# Online Mem0 judge over a finished run (GPT-4o-mini; skips category 5)
python -m scripts.analysis.run_benchmark --run experiments/<run_id>

# Same component, offline plumbing smoke
python -m scripts.analysis.run_benchmark --run experiments/<run_id> --autorater mock

# Mem0 / Mem0g write-index (no QA). Mock smoke; live gpt-4o-mini is costly.
python -m src.locomo_eval.mem0.run_index --config configs/mem0.yaml --extractor mock --embedder mock --max-samples 1 --run-id smoke_mem0_index
python -m src.locomo_eval.mem0.run_index --config configs/mem0g.yaml --run-id mem0g_locomo10

# Deterministic preprocess write-index (no LLM). Full locomo10, then 10 reader calls per sanity memory:
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess --eval-questions 10
python -m src.locomo_eval.preprocess.run_index --eval-questions 10 --eval-reader mock --run-id smoke_preprocess

# Unit tests — preprocess (HLD i) + Mem0 index + evaluation (HLD iv) + sandwich regression locks
python -m pytest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_preprocess_index.py tests/test_mem0_index.py tests/test_evaluation_pipeline.py tests/test_regressions.py tests/test_autorater_sanity.py tests/test_integration_sanity.py tests/test_run_isolation.py -q
# or (file path avoids a site-packages module named `tests` shadowing this folder)
python -m unittest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_preprocess_index.py tests/test_mem0_index.py tests/test_evaluation_pipeline.py tests/test_regressions.py tests/test_autorater_sanity.py tests/test_integration_sanity.py tests/test_run_isolation.py

# Compare two prediction sets (offline; LoCoMo F1 boxplot + histograms)
python -m scripts.analysis.compare_predictions --a experiments/cmp_raw_chunks --b experiments/cmp_session_summaries --out experiments/compare_raw_chunks_session_summaries
python scripts/compare_full_runs.py --runs experiments/cmp_raw_chunks experiments/cmp_session_summaries --out experiments/compare_raw_chunks_session_summaries
python scripts/compare_cross_model.py --runs experiments/session_summaries_mini experiments/session_summaries_luna --axis reader --out experiments/compare_reader_mini_luna


# Session-document tables + dataset histograms (no API; needs locomo10.json)
python scripts/export_session_documents.py --data data/raw/locomo10.json --out data/processed
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
- `run_meta.json` — model, **teacher_model**, prompt, data hash, git hash, timestamp
- `plots/` — overall + category bars

Agents must not silently skip CSV/plots when code paths change.

An autorater pack under `experiments/<run_id>/autorater/` must include
`autorater_verdicts.jsonl`, `autorater_metrics.json`, `run_meta.json`,
`SUMMARY.md`, `tables/`, and `plots/`. It is a snapshot: every invocation
clears these generated artifacts and regenerates from one prediction file.
Never append autorater analyses.

A Mem0 write-index under `experiments/<run_id>/mem0_index/` must include
`run_meta.json`, `schema.json`, `index.jsonl`, and `by_sample/<id>/speaker_a.json`
+ `speaker_b.json` (and `graph.json` when `enable_graph`). Reusing a run id
clears that dump and regenerates every sample.

A deterministic preprocess dump under `experiments/<run_id>/preprocess/` must include
`run_meta.json`, `schema.json`, `index.jsonl`, and `by_sample/<id>/sessions.jsonl`
+ `documents.jsonl`. No LLM. Gold answers stay out of the dump.

---

## Design rules for agents

1. **Sandwich:** only change one middle variable per experimental claim. Default reader (`gpt-4o-mini` + `qa_mem0_v1`) and mem0 writer (`gpt-4o-mini` extract/update) stay frozen for graph/store RQs. Reader-model or `qa_v1` swaps are a **different** axis (`compare_cross_model.py` / prompt override).
2. **Orchestrator is software**, not one giant LLM call (future TeacherOrchestrator modules).
3. **Prefer small pure functions** over frameworks.
4. **Keep metrics dual-reported:** SPEC token F1/EM *and* LoCoMo category F1.
   Autorater reports additionally use Mem0 F1/BLEU-1/J and must label these
   separately; Mem0 category 5 is excluded from J.
   Mock autorater output is plumbing-only (`mock_sanity_not_llm_judge`), must
   not occupy the literature J column, and must never log a live model id.
5. **Runs are self-contained.** Every pipeline invocation (QA, autorater, Mem0
   index, preprocess index) clears its generated output and regenerates.
   `predictions.jsonl` is an audit artifact, not a checkpoint. Do not add
   response stores, JSONL skip, or per-sample index skip.
6. **Plain YAML**, plain JSON loaders, local CSV — no Hydra/W&B required. Each config file is standalone; `pipeline.memory` is a builder id, not another YAML.
7. **Update docs:** after behavior change, copy previous AGENTS/HUMANS into `docs/agent/traces/YYYY-MM-DD_topic.md`, then edit live files.

---

## Code, tests, and comments

From review. Follow these when adding or renaming code.

**Names.** Unambiguous, justified, and kept current. If a name no longer matches the behavior, rename it (e.g. a generic orchestrator must not be called `run_baseline` when “baseline” is a config). Do not overload research terms across different mechanisms (audit JSONL vs a live store; baseline YAML vs `session_summaries`). Condition ids are descriptive (`raw_chunks`), not numbered (`C0`).

**Comments.** Explain *why* and the surrounding context, not a restatement of the next line. Ambiguous helpers need a one-liner on what they pin for later reproduction (`_git_hash`, `_file_sha256`). Distinguish lookalike layers (`offline_evaluate.py` string scoring vs `autorater.py` online LLM judging).

**Schemas / data-model classes.** The module docstring should map how types connect and which pipeline step uses them (load → memory → reader → prediction → score). Each class gets a short “what it is / who consumes it” note. Label gold answers as scorer-only (never in the reader prompt).

**Tests.**
- One unit test focuses on one function (`exact_match` tests stay separate from `token_f1` tests).
- Test names include the behavior **and** the expected outcome, e.g. `test_exact_match_returns_one_when_answers_match_after_normalization`.
- Group related cases in a `TestCase` per function or class; do not pile unrelated functions into one method.

**Experiments.** One YAML per `run_locomo_pipeline_with_memory_config` call (`python -m src.locomo_eval.run`). Compare `raw_chunks` vs `session_summaries` with two runs, then `scripts/compare_full_runs.py` or `python -m scripts.analysis.compare_predictions` (no API). Full/cross-model comparisons infer the answer prompt from `run_meta.json` and reject mismatched prompts unless an explicit `--prompt` is supplied. Do not fold A vs B into `run.py`. Compare reader or teacher **models** with `scripts/compare_cross_model.py` (also no API); that is a different axis fed into the same two-pack compare.

---

## Out of scope (v0.1)

Multi-teacher fusion, Mem0 Platform / Neo4j / Qdrant, claiming paper Table 1–2 J from this OSS clone, mixing a reader-prompt swap into a graph/store claim, validator loop, retrieval budgets as experiments, training/distillation loop, web UI, event-summarization / multimodal tasks.

Stub remnants (`src/train.py`, `src/distill/`) are deferred KD; do not wire unless requested.

---

## Acceptance checklist for agent PRs

- [ ] Dataset loads without editing source JSON  
- [ ] Default config remains Mem0-parity (GPT-4o-mini reader/judge, pinned prompts, released message/request shape); overrides do not mutate defaults
- [ ] One-question and full-run share the same command  
- [ ] Predictions JSONL deterministic fields  
- [ ] Metrics include EM, token F1, LoCoMo F1 by category  
- [ ] Autorater includes Mem0 F1/BLEU-1/J, category-5 exclusion, fresh tables, and fresh plots (never appends)
- [ ] Memory builder swappable without changing reader/evaluator  
- [ ] Tests for parse, memory, preprocess session blocks, session-document join, Mem0 index (mock), normalize  
- [ ] Model-integration sanity (`test_integration_sanity`: reader swap, teacher family swap, LoCoMo vs SPEC scorer)  
- [ ] `tests/test_autorater_sanity.py` stays green (mock only; no API)
- [ ] `tests/test_mem0_index.py`, `tests/test_regressions.py`, and `tests/test_run_isolation.py` stay green (mock only; no API)
- [ ] AGENTS.md + HUMANS.md updated + trace snapshot  

---

## Traceability

- Spec: `docs/agent/SPEC_v1.md`  
- LoCoMo pin: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`  
- Paper: Maharana et al., arXiv:2402.17753  
- Mem0 protocol: Chhikara et al., arXiv:2504.19413 (architecture clone; not Platform v2 numbers)
- Autorater protocol/baselines: Chhikara et al., arXiv:2504.19413
- Autorater prompt source: [Mem0 `ACCURACY_PROMPT` (pinned code)](https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py), also reproduced in paper Appendix A
- Sandwich idea: Bowman et al. 2022 scalable oversight  


---

# HUMANS.md snapshot


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
| Middle | Variable (`raw_chunks`, `session_summaries`, `mem0`, `mem0g`, later teacher/fusion) | How candidate memory is constructed / fused |
| Bottom | Fixed | Retrieval budget, answer LLM + prompt, metrics |

**v0.1 goal:** ship the **bottom + I/O skeleton** using LoCoMo’s own **session summaries** as memory, answered by one fixed OpenAI model, scored with LoCoMo-compatible metrics, and with **auditable logs/plots**.

You are not yet running multi-teacher fusion. **`raw_chunks`** vs **`session_summaries`** compare how much structure helps under one fixed answer model. Full system map: `docs/reports/engineering_notebook.md`.

HLD **(i) pre-processing** emits ordered **session blocks** (stable `turn_id`s) via `DataIngestor` and `PreprocessingPipeline`. Dump them with `python -m src.locomo_eval.preprocess.run_index` (no LLM). `raw_chunks` / `session_summaries` can load that dump (`--preprocess-index-run-id`) or still read `Conversation` from `dataset.py`. A **Mem0 write-index** walks those blocks as message pairs (extract + update; optional graph) and dumps JSON under `experiments/<run_id>/mem0_index/`. Full LoCoMo QA for Mem0/Mem0g is a **later** command — this slice only indexes and can load the dump into `Memory.text`.

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

### Three different model/scoring roles (do not conflate)

```text
                    ┌─────────────────────────────────────┐
  gold answer ─────►│  SCORER (metrics.py / LoCoMo F1)    │  ← sees reference
  model answer ────►│  compares pred vs gold              │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
  Memory.text ─────►│  ANSWER LLM (OpenAI / Claude later) │  ← must NOT see gold
  question only ───►│  fixed prompt selected by config    │
                    └─────────────────────────────────────┘
```

1. **Answer model (what we usually mean by “running the eval”)**  
   Blind to the gold answer. Prompts inject only **`{memory}` + `{question}`**.
   The default baseline and memory-condition YAMLs freeze pinned
   `prompts/qa_mem0_v1.txt` as Mem0's sole system message. Neither receives
   the gold answer, category ID, or evidence list. Override `pipeline.prompt_path`
   (e.g. `qa_v1.txt`) only as a separate robustness axis.

2. **Scorer (after the model answers)**  
   *Does* see the reference answer (and category, so LoCoMo’s category-aware
   F1 can apply). It is deterministic and offline. Re-run it with
   `python -m src.locomo_eval.offline_evaluate`.

3. **Autorater (after the model answers)**
   Also sees question + reference + predicted answer, but asks a separate LLM
   for a binary `CORRECT` / `WRONG` judgment. `autorater.py` implements the
   Mem0 LLM-as-a-Judge prompt; `scripts.analysis.run_benchmark` defaults to
   the released GPT-4o-mini judge. This path does not see memory and does not implement Mem0
   extraction/update. It skips category 5, matching the Mem0 paper.

So we do not literally “mask columns on the dataset file.” We **construct a restricted prompt** from selected fields and never put `answer` into that prompt.

### What memory design is for

**Memory design = how we build the `Memory.text` (and later structure/schema) that is glued to the question for the answer model.**

```text
LoCoMo conversation  ──►  MemoryBuilder (raw_chunks / session_summaries / …)  ──►  Memory.text
                                                                  │
question  ────────────────────────────────────────────────────────┤
                                                                  ▼
                                              fixed prompt → fixed LLM → pred
                                                                  │
gold answer ──────────────────────────────────────────────────────► scorer
```

- **`raw_chunks`:** dump raw dialog as the memory string (little structure).  
- **`session_summaries`:** dump released session summaries (structured-ish, still not a multi-teacher schema).  
- **`teacher_session_summaries`:** one LLM summarizes each session; swap `teacher.model` within a family.  
- **`mem0` / `mem0g`:** load a Mem0 write-index dump and retrieve top-k facts (graph relations for `mem0g`). Does not re-extract.  
- **Later:** `top1_teacher`, `whole_memory_aggregation`, `claim_fusion`, distilled `GraphMemory` — still the same idea: **produce a better memory payload for the same fixed Q + fixed answer LLM.**

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

## Compare raw_chunks vs session_summaries

Each `python -m src.locomo_eval.run` call is **one** memory config (`run_locomo_pipeline_with_memory_config`). Freeze answer model + prompt; swap only the memory YAML; then compare the two run folders:

```bash
python -m src.locomo_eval.run --config configs/raw_chunks.yaml --max-questions 20 --run-id cmp_raw_chunks_n20
python -m src.locomo_eval.run --config configs/session_summaries.yaml --max-questions 20 --run-id cmp_session_summaries_n20
python scripts/compare_full_runs.py --runs experiments/cmp_raw_chunks_n20 experiments/cmp_session_summaries_n20 --out experiments/compare_raw_chunks_session_summaries
```

`compare_full_runs.py` is the **sandwich report** (needs two finished run folders: metrics, run_meta, predictions). `compare_predictions` is the **two-JSONL LoCoMo F1 plotter** it uses internally; you can call it alone if you only want boxplot + histograms:

```bash
python -m scripts.analysis.compare_predictions --a experiments/cmp_raw_chunks_n20 --b experiments/cmp_session_summaries_n20 --out experiments/compare_raw_chunks_session_summaries
```

Both need those `--run-id` folders to exist first (the error names the `run` command if they do not).

Open `experiments/compare_raw_chunks_session_summaries/overall.csv` (from `compare_full_runs.py`) and `plots/locomo_f1_boxplot.png` / `plots/locomo_f1_histograms.png`. Pairwise rows: `paired_questions.csv` or `paired_locomo_f1.csv`.

Evaluation-pipeline unit tests (string metrics / LoCoMo F1) live in `tests/test_evaluation_pipeline.py`. Preprocess session-block tests are `tests/test_preprocessing_pipeline.py`. Session-document join / naive retrieval checks are `tests/test_session_documents.py`. Sandwich contracts that must not drift are locked in `tests/test_regressions.py` (mock only):

```bash
python -m unittest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_preprocess_index.py tests/test_mem0_index.py tests/test_evaluation_pipeline.py tests/test_regressions.py
```

Flatten session documents + histograms (no API; gitignored under `data/processed/`):

```bash
python scripts/export_session_documents.py --data data/raw/locomo10.json --out data/processed
```

Open `data/processed/session_documents.csv` (session units) and `qa_joined.csv` (gold joined via evidence). Plots are in `data/processed/plots/`. JSON category ids are official LoCoMo eval ids (1=multi-hop, 4=single-hop), not the paper’s 1–5 prose list — see `data/README.md`.

---

## Deterministic preprocess write-index (no LLM)

Parse the whole release into session units you can later retrieve/format (raw turns, dataset summaries, later top-k or LLM compress). Gold stays out of the dump.

```bash
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess
```

Dumps land at `experiments/locomo_preprocess/preprocess/` (`sessions.jsonl` + `documents.jsonl` per sample). Then 10 frozen-reader LLM calls per sanity memory, **round-robin across conversations** (with 10 LoCoMo samples that is one question from each), from that dump:

```bash
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess --eval-questions 10
```

That is **20** reader calls (`raw_chunks` + `session_summaries` × 10). Mock:

```bash
python -m src.locomo_eval.preprocess.run_index --eval-questions 10 --eval-reader mock --run-id smoke_preprocess
```

QA packs: `experiments/locomo_preprocess_raw_chunks_n10/` and `experiments/locomo_preprocess_session_summaries_n10/`. You can also point a later `run.py` at the dump with `--preprocess-index-run-id locomo_preprocess`.

---

## Mem0 / Mem0g write-index (no QA in this slice)

This clones the Mem0 **write path** (message pairs, dual speaker indexes, extract, ADD/UPDATE/DELETE/NONE, optional in-memory graph). It is **not** the closed Mem0 Platform run that produced paper J 66.88 / 68.44. Schema: [`docs/schemas/mem0_index.md`](../schemas/mem0_index.md).

Offline plumbing:

```bash
python -m src.locomo_eval.mem0.run_index --config configs/mem0.yaml --extractor mock --embedder mock --max-samples 1 --run-id smoke_mem0_index
```

Live index (costly; default extract model `gpt-4o-mini`):

```bash
python -m src.locomo_eval.mem0.run_index --config configs/mem0.yaml --run-id mem0_locomo10
python -m src.locomo_eval.mem0.run_index --config configs/mem0g.yaml --run-id mem0g_locomo10
```

Dumps land at `experiments/<run_id>/mem0_index/`. Retrieve is cosine **top-k NL facts** (default 30) per speaker; mem0g also appends valid graph relations into the same `Memory.text`. A later graph/store condition should reuse this retriever. QA: `python -m src.locomo_eval.run --config configs/mem0.yaml` with `mem0.index_run_id` pointing at the dump. Default reader is `gpt-4o-mini` + `qa_mem0_v1` (same as raw_chunks / session_summaries).

---

## Cross-model robustness (reader or teacher)

This is **not** a memory-condition comparison. Freeze memory (or freeze the reader) and swap one model slot.

**Answer / eval model** (`gpt-4.1-mini` vs `gpt-5.6-luna`):

```bash
python -m src.locomo_eval.run --config configs/session_summaries.yaml --max-questions 5 --run-id cmp_reader_mini_n5
python -m src.locomo_eval.run --config configs/session_summaries_reader_gpt56_luna.yaml --max-questions 5 --run-id cmp_reader_luna_n5
python scripts/compare_cross_model.py --runs experiments/cmp_reader_mini_n5 experiments/cmp_reader_luna_n5 --axis reader --out experiments/compare_reader_mini_luna
```

**Teacher model within GPT-5.6** (live `teacher_session_summaries`; freeze the reader):

```bash
python -m src.locomo_eval.run --config configs/teacher_session_summaries.yaml --teacher-model gpt-5.6-luna --max-questions 3 --run-id cmp_teacher_luna
python -m src.locomo_eval.run --config configs/teacher_session_summaries.yaml --teacher-model gpt-5.6-terra --max-questions 3 --run-id cmp_teacher_terra
python scripts/compare_cross_model.py --runs experiments/cmp_teacher_luna experiments/cmp_teacher_terra --axis teacher --out experiments/compare_teacher_family
```

Offline teacher smoke (no API):

```bash
python -m src.locomo_eval.run --config configs/teacher_session_summaries.yaml --reader mock --teacher mock --teacher-model gpt-5.6-luna --max-questions 3 --run-id smoke_teacher_luna
```

`compare_cross_model.py` writes `overall.csv`, `paired_questions.csv`, `SUMMARY.md`, `compare.json`. Check `run_meta.json` for `reader_model` / `teacher_model` / `*_family`. Distinctness is `fraction_same_memory_text` and `fraction_same_answer` (inspect traces if you need to confirm the filled prompt changed).

Unit tests for these seams (no API): `python -m unittest tests/test_integration_sanity.py`.
Isolation (no store, self-contained run ids): `python -m unittest tests/test_run_isolation.py tests/test_regressions.py`.

---

## Run the baseline

**Offline smoke (no money / no key):**

```bash
python -m src.locomo_eval.run --config configs/session_summaries.yaml --reader mock --max-questions 5 --run-id smoke_mock
```

**Small live check:**

```bash
python -m src.locomo_eval.run --config configs/mem0_baseline.yaml --max-questions 3 --run-id smoke_openai
```

**Full QA set** (many API calls; each invocation starts from question one):

```bash
python -m src.locomo_eval.run --config configs/mem0_baseline.yaml --run-id mem0_baseline_session_summary
```

The baseline pins Mem0's released answer controls: `gpt-4o-mini`,
`prompts/qa_mem0_v1.txt` as the sole system message, temperature 0, and no
explicit completion-token limit. Its memory remains dataset-provided
`session_summaries`; it does not implement Mem0 extraction/update. For GPT-5.6
Luna as the **answer** model (robustness axis, not a memory claim):

```bash
python -m src.locomo_eval.run --config configs/session_summaries_reader_gpt56_luna.yaml --max-questions 5 --run-id cmp_reader_luna_n5
python scripts/compare_cross_model.py --runs experiments/smoke_openai experiments/cmp_reader_luna_n5 --axis reader --out experiments/compare_reader_mini_luna
```

Config knobs live only in YAML + CLI overrides — no hidden flags.
For a deliberate non-parity run, override without editing the default:

```bash
python -m src.locomo_eval.run --config configs/mem0_baseline.yaml \
  --model gpt-4.1-mini --prompt prompts/qa_v1.txt \
  --max-tokens 64 --message-layout default_system_user --run-id ablation
```

`tests/test_regressions.py` locks the on-disk defaults and verifies that CLI
overrides affect only the effective run metadata.

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

`predictions.jsonl` is an audit artifact. Reusing a run id clears generated
artifacts and rebuilds from question one.

Recompute string metrics without re-calling the API (not an LLM autorater):

```bash
python -m src.locomo_eval.offline_evaluate --predictions experiments/<run_id>/predictions.jsonl
```

## Run the Mem0-style autorater benchmark

Generate answers first with the normal pipeline, then grade the same stored
prediction/reference pairs:

```bash
# Offline component smoke (no judge API)
python -m scripts.analysis.run_benchmark \
  --run experiments/<run_id> --autorater mock

# Live released Mem0 GPT-4o-mini judge
python -m scripts.analysis.run_benchmark \
  --run experiments/<run_id>
```

`--autorater mock` is plumbing-only. It marks an answer correct when it shares
a substantive token with the gold answer. It is deliberately cheap and can
overrate contradictions (for example, `2 July 2023` vs `3 July 2023`).
Reports label it `mock_sanity_not_llm_judge`, omit it from the literature J
column, and never identify it as GPT-4o. Use the live command for actual J.
Every invocation removes prior generated autorater artifacts in that output
directory and regenerates from one prediction file. Autorater results never
append, so mock/live or different source runs cannot overlap.

The default `configs/autorater.yaml` judge is the released Mem0
`gpt-4o-mini` configuration. The
paper reports the mean ± standard deviation of 10 full judge runs; one local
autorater pack is one run, so repeat it under distinct output directories for
paper-level uncertainty estimates.
Judge model, prompt, temperature, and token limit can be overridden with
`--model`, `--prompt`, `--temperature`, and `--max-tokens`.

Prompt provenance:

- Pinned Mem0 implementation:
  [`evaluation/metrics/llm_judge.py`](https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py)
  (`ACCURACY_PROMPT`)
- Paper:
  [Chhikara et al., arXiv:2504.19413](https://arxiv.org/abs/2504.19413),
  Appendix A, “Prompt Template for LLM as a Judge”
- Local adaptation: `prompts/autorater_mem0_v1.txt`. It preserves the Mem0
  correctness/date-matching instructions and JSON label contract.

Each pipeline invocation is self-contained: the answer pipeline clears prior
generated artifacts under the selected run id and regenerates from question one.
Autorater rewrites all of its outputs. It never modifies source
`predictions.jsonl`, `metrics.json`, or `run_meta.json`.

Outputs under `experiments/<run_id>/autorater/`:

| Artifact | Contents |
|----------|----------|
| `autorater_verdicts.jsonl` | Fresh per-question F1, BLEU-1, J label, usage, reader/judge latency; overwritten each invocation |
| `autorater_metrics.json` | Overall and category metrics; category 5 excluded from J |
| `tables/overall.csv` | This run's F1/BLEU-1/J, token usage, p50/p95 latency |
| `tables/vs_literature.csv` | This run next to published Mem0 Table 2 J/latency values |
| `tables/vs_literature_by_category.csv` | Published Mem0 Table 1 category values |
| `plots/` | Correct/wrong verdict bar, continuous-score histograms/boxplots, category bars, and latency comparisons |
| `SUMMARY.md` | Compact interpretation and paper citation |

Published rows are **literature pins**, not local re-runs of Mem0. Scores use
the Mem0 paper's percentage scale in comparison tables. Mem0 F1 is distinct
from this repository's LoCoMo F1 and SPEC token F1.

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
    → dataset.py              # Conversation / Question (read path; gold scorer-only)
    → data_ingestor.py        # HLD (i): wrap dataset.py, do not rewrite source JSON
    → preprocessing_pipeline.py  # HLD (i): SessionBlock[]
    → mem0/run_index.py       # write-index dumps (mem0 / mem0g); not QA
    → teacher_orchestrator.py # HLD (ii): one session block, passthrough, no LLM
    → memory.py               # experimental: Memory.text (raw_chunks / session_summaries / teacher_session_summaries / mem0 / mem0g)
    → prompts/qa_mem0_v1.txt  # pinned Mem0-parity answer prompt (override qa_v1 as a separate axis)
    → readers.py              # answer LLM (swap only for robustness, not a memory claim)
    → metrics + report        # scorer uses gold; reports for humans
```

Later swaps should only replace **memory builders** (and eventually teacher/fusion), not the answer prompt or scorer, if you want clean sandwich comparisons.

---

## Roadmap sketch (not in v0.1)

| Condition | Middle layer | Question |
|-----------|--------------|----------|
| `raw_chunks` | Raw / chunked dialog | Does structure help? |
| `session_summaries` | LoCoMo-provided session summaries | How strong is a structured session-memory bank? |
| `teacher_session_summaries` | Live single-teacher session summaries | Does a live teacher beat released summaries? |
| `mem0` | Mem0 vector extract+update dump + top-k retrieve | Does the paper write path help vs session summaries? |
| `mem0g` | mem0 + in-memory graph relations | Does graph structure add anything (same extract)? |
| `top1_teacher` | Top-1 of K teachers (later) | Selection enough? |
| `whole_memory_aggregation` | Aggregate whole memories (later) | Synthesis enough? |
| `claim_fusion` | Claim-level fusion + validation (later) | Evidence-grounded fusion win? |

Teacher K∈{1,2,3} and utility U_K = Δscore / Δcost come **after** this baseline is trustworthy.

---

## What you should ask agents to do (and not)

**Do:** extend memory builders or `GraphMemory`, swap reader/teacher models for robustness checks, improve reports, run Mem0 indexes, keep docs current.

**Don’t (yet):** claim paper Table 1–2 J from this OSS clone, call Mem0 Platform / Neo4j, multi-LLM fusion, training loops, silent metric changes, deleting experiment data, committing secrets.

---

## Doc versioning

Live: `docs/agent/AGENTS.md`, `docs/agent/HUMANS.md`, `docs/agent/SPEC_v1.md`  
History: `docs/agent/traces/` (dated snapshots when these change)  
