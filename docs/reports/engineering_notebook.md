# Engineering notebook — system map & extension points

Living notes for humans reviewing the LoCoMo sandwich pipeline.  
**Update when** frozen/variable surfaces or condition IDs change.  
Related: `docs/agent/HUMANS.md`, `docs/agent/AGENTS.md`, `docs/agent/SPEC_v1.md`.

---

## 1. Experimental sandwich (what to freeze)

Borrowed from Bowman et al. (2022) “scalable oversight” sandwich: fix top + bottom; only vary the middle.

| Layer | Role | Status in this repo |
|-------|------|---------------------|
| **Top (fixed)** | LoCoMo conversations + questions | `dataset.py` + `data/raw/locomo10.json` |
| **Middle (variable)** | How “memory” text is produced (C0–C4) | `memory.py` (+ later write/fusion/store) |
| **Bottom (fixed)** | Answer prompt, answer LLM, metrics, reporting | `prompts/`, `readers.py`, `metrics.py`, `report.py` |

**Attribution rule:** if you change prompt *or* model *and* memory between two runs, you cannot cleanly attribute the score delta to memory design alone.

---

## 1b. External APIs & models (inventory)

Keep this list current when providers/models change.

### Active

| Provider | API surface | SDK / endpoint | Auth | Models we use | Where configured | Code |
|----------|-------------|----------------|------|---------------|------------------|------|
| OpenAI Platform | Chat Completions | `openai` Python package → `chat.completions.create` | `.env` → `OPENAI_API_KEY` | **`gpt-4.1-mini`** (default reader in c0/c1/baseline YAML); **`gpt-5.6-luna`** (`configs/c1_reader_gpt56_luna.yaml`); **`gpt-4.1`** / **`gpt-5.6-terra`** / **`gpt-5.6-sol`** when set | `reader.model`, `teacher.model`, CLI `--model` / `--teacher-model` | `OpenAIReader` / `OpenAITeacher` via `models.py` |

**Request shape (answer LLM):**

- System: `You answer questions using only the provided memory.`
- User: text from `prompts/qa_v1.txt` filled with memory + question
- GPT-4.1: `temperature=0.0`, `max_tokens=64` unless config overridden
- GPT-5.6 (Luna/Terra/Sol): `max_completion_tokens` (min 16), `reasoning_effort=none`, no temperature (see `src/locomo_eval/models.py`)

**Commands that call OpenAI:**

```bash
# Live answer generation (bills uncached Qs)
python -m src.locomo_eval.run --config configs/c0_raw.yaml --max-questions 20 --run-id cmp_c0_n20
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --max-questions 20 --run-id cmp_c1_n20
python -m src.locomo_eval.run --config configs/baseline.yaml --model gpt-4.1 --run-id ...
python -m src.locomo_eval.run --config configs/c1_reader_gpt56_luna.yaml --max-questions 5 --run-id ...
python -m src.locomo_eval.run --config configs/c1_teacher.yaml --teacher-model gpt-5.6-luna --max-questions 3 --run-id ...
```

**Commands that do *not* call OpenAI:**

```bash
python -m src.locomo_eval.run --reader mock ...
python -m src.locomo_eval.offline_evaluate --predictions experiments/<run>/predictions.jsonl
python scripts/compare_full_runs.py --runs ... --out ...
python scripts/compare_cross_model.py --runs ... --axis reader --out ...
python scripts/prepare_data.py --split all --no-jsonl
python scripts/fetch_locomo.py   # HTTP to GitHub raw only
```

### Considerations

| Concern | Practice |
|---------|----------|
| Rate limits / RPD | Low tiers (~50/day): small `max_questions`, same `--run-id` resume (`predictions.jsonl`) |
| Cost | Tokens ∝ memory string length (C0 >> C1 typically); requests ∝ unanswered Q count |
| Reproducibility | Log `reader_model`, temp, prompt version, data SHA in `run_meta.json` |
| Cache | `LlmResponseCache` in `utils/` is implemented but **not wired** (future optimization after E2E) |
| Security | Never commit `.env`; example only in `.env.example` |

### Planned (not implemented)

- Anthropic Claude as alternate fixed answer reader (or later judge — Phase‑1 has no LLM judge).
- Multi-teacher providers (GPT / Gemini / DeepSeek) on the **write** path only.

---

## 2. Runtime data path (injection)

```text
Conversation + Question
        │
        ▼
  MemoryBuilder.build()     ← EXPERIMENTAL (middle)
        │
        ▼
  Memory { memory_type, text, source_ids }
        │
        │   .text is the only payload the answer model sees
        ▼
  prompts/*.txt  {memory} + {question}   ← FREEZE for fair C comparisons
        │
        ▼
  Reader (OpenAI / mock, temp=0)         ← FREEZE for fair C comparisons
        │
        ▼
  predicted_answer
        │
        ▼
  metrics (EM, token F1, LoCoMo F1)      ← FREEZE always
        │
        ▼
  experiments/<run_id>/  CSV · JSONL · plots · run_meta
```

Single-config runner (`run_locomo_pipeline_with_memory_config` in `run.py`)
injects that condition into the answer LLM:

```text
reader.answer(memory.text, q.question, prompt_template)
```

A vs B is two of those calls (different YAML / `--run-id`), then
`scripts/compare_full_runs.py`. This function never compares.

The reader is intentionally dumb about teachers, fusion, and stores.

---

## 3. Conditions (draft set)

| ID | Name in code | What the answer LLM receives | Research question |
|----|--------------|------------------------------|-------------------|
| **C0** | `c0_raw` | Full raw dialog turns (chronological), optional char cap | Does structure help at all? |
| **C1** | `c1_session_summary` | LoCoMo-released session summaries (dataset “memory”) | How strong is a structured session-memory bank? |
| **C1 teacher** | `c1_teacher` | Per-session summaries from one teacher LLM | Does a live teacher beat released summaries? Swap teacher model within a family as a robustness check. |
| C2 | (future) | Top-1 of K teacher memories | Selection enough? |
| C3 | (future) | Aggregated whole memories | Synthesis enough? |
| C4 | (future) | Claim-level fused + validated store (+ retrieve) | Fine-grained fusion win? |

**C1 draft note:** default C1 still uses **provided** LoCoMo session summaries. `c1_teacher` is the live single-teacher replacement (same inject path). Multi-teacher fusion is still out of scope.

Aliases for convenience:

| Alias | Resolves to |
|-------|-------------|
| `raw` / `raw_dialog` | `c0_raw` |
| `teacher` | `c1_teacher` |

---

## 4. Where to change what

### 4.1 Answer prompt (bottom — freeze after lock-in)

| Item | Location |
|------|----------|
| Template text | `prompts/qa_v1.txt` (placeholders `{memory}`, `{question}`) |
| Loader | `src/locomo_eval/prompts.py` |
| Select in config | `pipeline.prompt_path` in YAML |
| CLI | `--prompt path/to/qa_v2.txt` |
| Logged as | `prompt_version` = filename stem |

Do **not** fork prompts per condition for the main table. If you ablate prompts, re-run **all** conditions under the new prompt and report the setup change.

### 4.2 Memory design (middle — primary experiment surface)

| Item | Location |
|------|----------|
| Interface | `MemoryBuilder.build(conversation, question) -> Memory` in `memory.py` |
| Registry | `get_memory_builder(name)` |
| Select in config | `pipeline.memory` |
| CLI | `--memory c0_raw` or `--memory c1_session_summary` |
| Logged as | `memory_type` on each prediction + `run_meta` |

**Add a new condition**

1. Implement a class with `name = "cX_..."`.
2. Register it in `get_memory_builder`.
3. Prefer writing long intermediates to `experiments/<run_id>/memories/` later; still end by filling `Memory.text`.
4. Run with the **same** `--prompt` / reader model as other C’s.

**Teacher (`c1_teacher`):** YAML `teacher.model` (or `--teacher-model`) selects the write-path LLM. Logged on `Memory.teacher_model`, `run_meta.json`, and each prediction row. Do not change `reader.model` in the same comparison if you want the delta attributed to the teacher.

**Reader-model robustness:** YAML `reader.model` / `--model` / `configs/c1_reader_gpt56_luna.yaml`. Compare with `scripts/compare_cross_model.py --axis reader`. Do not mix with a C0 vs C1 claim.

### 4.3 True multi-teacher write path (later, still middle)

Planned package (not required for C0/C1 draft):

```text
src/locomo_eval/write/   teachers, fusion, validator, store
src/locomo_eval/retrieve.py   fixed budget from store → Memory.text
```

Write vs read:

```text
WRITE (variable): conversation → teachers → fusion → validate → store
READ  (fixed policy): question → retriever(budget) → Memory.text → prompt → answer LLM → metrics
```

Orchestrator stays **software** (prompts, parallel IO, JSON checks), not one monolithic “teacher orchestrator LLM.”

### 4.4 Reader / metrics / reports (bottom — freeze)

| Item | Location | Notes |
|------|----------|--------|
| OpenAI / mock | `readers.py` | Optional `LlmResponseCache` hook (unwired from `run.py`; see `utils/llm_response_cache.py`) |
| Env / keys | `.env` + `env.py` | never commit secrets |
| Scoring | `metrics.py`, `src/metrics/locomo_qa.py` | dual: SPEC + LoCoMo F1 |
| Offline rescore | `python -m src.locomo_eval.offline_evaluate ...` | string metrics only; no API; not an LLM autorater |
| Two-run compare | `scripts/analysis/compare_predictions.py` | paired LoCoMo F1 boxplot + histograms |
| Audit pack | `report.py` → `experiments/<run_id>/` | CSV/JSON/plots |

---

## 5. How to compare C0 vs C1

Freeze bottom, vary middle only:

```bash
# Same max_questions, model, prompt; different memory + run_id
python -m src.locomo_eval.run --config configs/c0_raw.yaml --max-questions 20 --run-id cmp_c0_n20
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --max-questions 20 --run-id cmp_c1_n20

# Side-by-side metrics table
python scripts/compare_full_runs.py \
  --runs experiments/cmp_c0_n20 experiments/cmp_c1_n20 \
  --out experiments/compare_c0_c1
```

Writes `overall.csv`, `by_category.csv`, `paired_questions.csv`, `SUMMARY.md`, `plots/` (including LoCoMo F1 **boxplot** + side-by-side **histograms**), and cache/memory **sanity** (`fraction_same_cache_key` ≈ 0 means conditions are distinct).

Two-pack LoCoMo F1 plots alone (no sandwich SUMMARY):

```bash
python -m scripts.analysis.compare_predictions \
  --a experiments/cmp_c0_n20 --b experiments/cmp_c1_n20 \
  --out experiments/compare_c0_c1
```

### Inspect the flattened QA table

```bash
python scripts/prepare_data.py --split all --no-jsonl   # data/processed/qa_all.csv
# omit --no-jsonl to also refresh the large JSONL with full context
```

| Artifact | Why |
|----------|-----|
| `metrics.json` | overall LoCoMo F1 / EM |
| `metrics_by_category.csv` | category-wise |
| `predictions.csv` | qualitative: same `question_id`, different `memory_type` |
| `run_meta.json` | confirm model + prompt_version match |

**Validity checklist**

- [ ] Same `reader_model`, `temperature`, `prompt_version`
- [ ] Same question subset (`max_questions` / sample filter)
- [ ] Differ only in `memory_type` / builder
- [ ] Resume via same `--run-id` / `predictions.jsonl`. Do not wire `LlmResponseCache` until E2E is trusted.

---

## 6. Control panel (frozen vs changeable)

| Piece | Freeze for C0–C4 main table? | Experiment instead? |
|-------|------------------------------|---------------------|
| Dataset + question IDs | Yes | No |
| Answer prompt | Yes (after lock) | Only as a separate meta-setup |
| Answer model + decode | Yes | Model swap = new setup label |
| Metrics | Always | No |
| **MemoryBuilder / write / store** | No | **Yes — main axis** |
| Retriever + token budget | Policy fixed across C’s | Budget ablations as a second axis |
| Report columns | Prefer stable | Additive fields OK |

---

## 7. Suggested growth order

1. C0 vs C1 under frozen prompt/model (this draft).  
2. Lock short-answer prompt if needed; re-run C0/C1.  
3. Persist mid-layer JSON per sample.  
4. Real single-teacher C1 (API generation → `Memory.text`).  
5. C2–C4 + fixed retriever budget.  

---

## 8. File index (engineering)

```text
configs/baseline.yaml              # default (C1-compatible)
configs/c0_raw.yaml
configs/c1_session_summary.yaml
configs/c1_reader_gpt56_luna.yaml  # C1 + GPT-5.6 Luna reader
configs/c1_teacher.yaml            # live single teacher
prompts/qa_v1.txt
prompts/teacher_session_v1.txt
src/locomo_eval/models.py          # model catalog / API kwargs
src/locomo_eval/teachers.py        # Mock + OpenAI teacher
src/locomo_eval/memory.py          # C0/C1/c1_teacher builders + registry
src/locomo_eval/utils/llm_response_cache.py  # LLM reply memo (unwired; future optimization)
src/locomo_eval/run.py             # wires builder → reader → report
scripts/compare_full_runs.py            # C0 vs C1 metrics side-by-side
scripts/analysis/compare_predictions.py  # two-pack LoCoMo F1 boxplot + histograms
scripts/compare_cross_model.py     # reader/teacher model robustness
docs/reports/engineering_notebook.md  # this file
```
