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

HLD **(i) pre-processing** emits ordered **session blocks** (stable `turn_id`s) via `DataIngestor` and `PreprocessingPipeline`. `raw_chunks` / `session_summaries` still read `Conversation` from `dataset.py`. A **Mem0 write-index** walks those blocks as message pairs (extract + update; optional graph) and dumps JSON under `experiments/<run_id>/mem0_index/`. Full LoCoMo QA for Mem0/Mem0g is a **later** command — this slice only indexes and can load the dump into `Memory.text`.

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
python -m unittest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_mem0_index.py tests/test_evaluation_pipeline.py tests/test_regressions.py
```

Flatten session documents + histograms (no API; gitignored under `data/processed/`):

```bash
python scripts/export_session_documents.py --data data/raw/locomo10.json --out data/processed
```

Open `data/processed/session_documents.csv` (session units) and `qa_joined.csv` (gold joined via evidence). Plots are in `data/processed/plots/`. JSON category ids are official LoCoMo eval ids (1=multi-hop, 4=single-hop), not the paper’s 1–5 prose list — see `data/README.md`.

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

Dumps land at `experiments/<run_id>/mem0_index/` (`speaker_a.json` / `speaker_b.json`, `graph.json` for mem0g, `ingest_log.jsonl`). Already-complete samples are skipped unless `--overwrite`. A later eval command can `python -m src.locomo_eval.run --config configs/mem0.yaml` **without re-extracting** (`mem0.index_run_id` must point at that dump). Keep `prompts/qa_v1.txt` frozen if you want a memory-condition claim; swapping in Mem0’s answer prompt is a different bottom-layer setup.

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

`compare_cross_model.py` writes `overall.csv`, `paired_questions.csv`, `SUMMARY.md`, `compare.json`. Check `run_meta.json` for `reader_model` / `teacher_model` / `*_family`. Distinctness uses `fraction_same_llm_request_hash` (SHA-256 of the intended reader request; **not** a live store hit).

Unit tests for these seams (no API): `python -m unittest tests/test_integration_sanity.py`.

---

## Run the baseline

**Offline smoke (no money / no key):**

```bash
python -m src.locomo_eval.run --config configs/session_summaries.yaml --reader mock --max-questions 5 --run-id smoke_mock
```

**Small live check:**

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --max-questions 3 --run-id smoke_openai
```

**Full QA set** (many API calls; resume unfinished questions via `predictions.jsonl`):

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --run-id baseline_session_summary
```

Default model is `gpt-4.1-mini` in `configs/baseline.yaml`. For GPT-5.6 Luna as the **answer** model (robustness axis, not a memory claim):

```bash
python -m src.locomo_eval.run --config configs/session_summaries_reader_gpt56_luna.yaml --max-questions 5 --run-id cmp_reader_luna_n5
python scripts/compare_cross_model.py --runs experiments/smoke_openai experiments/cmp_reader_luna_n5 --axis reader --out experiments/compare_reader_mini_luna
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
- **`experiments/cache/`** — `LlmResponseHash` (under `src/locomo_eval/utils/`). Implemented, **not wired** into `run.py` until E2E validation is done. Re-enable by passing a store to `get_reader`.

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
    → dataset.py              # Conversation / Question (read path; gold scorer-only)
    → data_ingestor.py        # HLD (i): wrap dataset.py, do not rewrite source JSON
    → preprocessing_pipeline.py  # HLD (i): SessionBlock[]
    → mem0/run_index.py       # write-index dumps (mem0 / mem0g); not QA
    → teacher_orchestrator.py # HLD (ii): one session block, passthrough, no LLM
    → memory.py               # experimental: Memory.text (incl. load/retrieve from mem0 dumps)
    → prompts/qa_v1.txt       # frozen answer prompt
    → readers.py              # answer LLM (no LlmResponseHash in this phase) (swap only for robustness, not a memory claim)
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

**Don’t (yet):** claim paper Table 1–2 J from this OSS clone, call Mem0 Platform / Neo4j, multi-LLM fusion, training loops, silent metric changes, deleting `experiments/cache/` or data, committing secrets.

---

## Doc versioning

Live: `docs/agent/AGENTS.md`, `docs/agent/HUMANS.md`, `docs/agent/SPEC_v1.md`  
History: `docs/agent/traces/` (dated snapshots when these change)  
