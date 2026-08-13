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
   *Does* see the reference answer (and category, so LoCoMo’s category-aware F1 can apply). That is not “cheating”; it is standard supervised scoring.

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
- **C1 teacher (`c1_teacher`):** one LLM summarizes each session; swap `teacher.model` within a family.  
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

Freeze answer model + prompt; only swap memory configs:

```bash
python -m src.locomo_eval.run --config configs/c0_raw.yaml --max-questions 20 --run-id cmp_c0_n20
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --max-questions 20 --run-id cmp_c1_n20
python scripts/compare_runs.py --runs experiments/cmp_c0_n20 experiments/cmp_c1_n20 --out experiments/compare_c0_c1
```

Open `experiments/compare_c0_c1/overall.csv` and the two `predictions.csv` files side by side (same `question_id`).

---

## Cross-model robustness (reader or teacher)

This is **not** a C0 vs C1 memory comparison. Freeze memory (or freeze the reader) and swap one model slot.

**Answer / eval model** (`gpt-4.1-mini` vs `gpt-5.6-luna`):

```bash
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --max-questions 5 --run-id cmp_reader_mini_n5
python -m src.locomo_eval.run --config configs/c1_reader_gpt56_luna.yaml --max-questions 5 --run-id cmp_reader_luna_n5
python scripts/compare_cross_model.py --runs experiments/cmp_reader_mini_n5 experiments/cmp_reader_luna_n5 --axis reader --out experiments/compare_reader_mini_luna
```

**Teacher model within GPT-5.6** (live `c1_teacher`; freeze the reader):

```bash
python -m src.locomo_eval.run --config configs/c1_teacher.yaml --teacher-model gpt-5.6-luna --max-questions 3 --run-id cmp_teacher_luna
python -m src.locomo_eval.run --config configs/c1_teacher.yaml --teacher-model gpt-5.6-terra --max-questions 3 --run-id cmp_teacher_terra
python scripts/compare_cross_model.py --runs experiments/cmp_teacher_luna experiments/cmp_teacher_terra --axis teacher --out experiments/compare_teacher_family
```

Offline teacher smoke (no API):

```bash
python -m src.locomo_eval.run --config configs/c1_teacher.yaml --reader mock --teacher mock --teacher-model gpt-5.6-luna --max-questions 3 --run-id smoke_teacher_luna
```

`compare_cross_model.py` writes `overall.csv`, `paired_questions.csv`, `SUMMARY.md`, `compare.json`. Check `run_meta.json` for `reader_model` / `teacher_model` / `*_family`.

Unit tests for these seams (no API): `python -m unittest tests/test_integration_sanity.py`.

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

**Full QA set** (many API calls; cache resumes if interrupted):

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --run-id baseline_session_summary
```

Default model is `gpt-4.1-mini` in `configs/baseline.yaml`. For GPT-5.6 Luna as the **answer** model (robustness axis, not a memory claim):

```bash
python -m src.locomo_eval.run --config configs/c1_reader_gpt56_luna.yaml --max-questions 5 --run-id cmp_reader_luna_n5
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

Recompute metrics without re-calling the API:

```bash
python -m src.locomo_eval.evaluate --predictions experiments/<run_id>/predictions.jsonl
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
    → dataset.py              # Conversation / Question objects (gold kept for scorer only)
    → memory.py               # experimental: Memory.text (C0 / C1 / c1_teacher)
    → prompts/qa_v1.txt       # frozen answer prompt
    → readers.py              # answer LLM (swap only for robustness, not C0 vs C1)
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

**Do:** extend memory builders, swap reader/teacher models for robustness checks, improve reports, add Claude reader when you switch fixed answer model, fix bugs, keep docs current.

**Don’t (yet):** multi-LLM fusion, training loops, silent metric changes, deleting cache/data, committing secrets.

---

## Doc versioning

Live: `docs/agent/AGENTS.md`, `docs/agent/HUMANS.md`, `docs/agent/SPEC_v1.md`  
History: `docs/agent/traces/` (dated snapshots when these change)  
