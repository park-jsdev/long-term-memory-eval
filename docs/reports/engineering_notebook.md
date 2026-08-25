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
| **Middle (variable)** | How “memory” text is produced (`raw_chunks`, `session_summaries`, later teacher/fusion) | `memory.py` (+ later write/fusion/store) |
| **Bottom (fixed)** | Answer prompt, answer LLM, metrics, reporting | `prompts/`, `readers.py`, `metrics.py`, `report.py` |

**Attribution rule:** if you change prompt *or* model *and* memory between two runs, you cannot cleanly attribute the score delta to memory design alone.

---

## 1b. External APIs & models (inventory)

Keep this list current when providers/models change.

### Active

| Provider | API surface | SDK / endpoint | Auth | Models we use | Where configured | Code |
|----------|-------------|----------------|------|---------------|------------------|------|
| OpenAI Platform | Chat Completions | `openai` Python package → `chat.completions.create` | `.env` → `OPENAI_API_KEY` | **`gpt-4.1-mini`** default reader; **`gpt-4o`** default autorater; **`gpt-4o-mini`** Mem0-paper judge option; GPT-5.6 family for robustness | `reader.model`, `teacher.model`, `configs/autorater.yaml`, CLI model overrides | `OpenAIReader` / `OpenAITeacher` / `OpenAIAutorater` via `models.py` |

**Request shape (answer LLM):**

- System: `You answer questions using only the provided memory.`
- User: text from `prompts/qa_v1.txt` filled with memory + question
- GPT-4.1: `temperature=0.0`, `max_tokens=64` unless config overridden
- GPT-5.6 (Luna/Terra/Sol): `max_completion_tokens` (min 16), `reasoning_effort=none`, no temperature (see `src/locomo_eval/models.py`)

**Commands that call OpenAI:**

```bash
# Live answer generation (bills every Q on every invocation)
python -m src.locomo_eval.run --config configs/raw_chunks.yaml --max-questions 20 --run-id cmp_raw_chunks_n20
python -m src.locomo_eval.run --config configs/session_summaries.yaml --max-questions 20 --run-id cmp_session_summaries_n20
python -m src.locomo_eval.run --config configs/baseline.yaml --model gpt-4.1 --run-id ...
python -m src.locomo_eval.run --config configs/session_summaries_reader_gpt56_luna.yaml --max-questions 5 --run-id ...
python -m src.locomo_eval.run --config configs/teacher_session_summaries.yaml --teacher-model gpt-5.6-luna --max-questions 3 --run-id ...
python -m scripts.analysis.run_benchmark --run experiments/<run_id>  # GPT-4o judge
```

**Commands that do *not* call OpenAI:**

```bash
python -m src.locomo_eval.run --reader mock ...
python -m src.locomo_eval.offline_evaluate --predictions experiments/<run>/predictions.jsonl
python -m scripts.analysis.run_benchmark --run experiments/<run_id> --autorater mock
python scripts/compare_full_runs.py --runs ... --out ...
python scripts/compare_cross_model.py --runs ... --axis reader --out ...
python scripts/prepare_data.py --split all --no-jsonl
python scripts/fetch_locomo.py   # HTTP to GitHub raw only
```

Mock autorater output is a token-overlap plumbing check, not LLM-as-a-Judge:
it is logged as provider/model `mock`, excluded from literature J plots/tables,
and may overrate contradictory answers that share dates or topic words.
Autorater never resumes or appends analysis. Every invocation clears its known
generated files plus `plots/` and `tables/`, then rates one prediction file
from scratch.

### Considerations

| Concern | Practice |
|---------|----------|
| Rate limits / RPD | Low tiers (~50/day): use small `max_questions`; evaluation does not resume |
| Cost | Tokens ∝ memory string length (`raw_chunks` >> `session_summaries` typically); requests ∝ question count per invocation |
| Reproducibility | Log `reader_model`, temp, prompt version, data SHA in `run_meta.json` |
| Cache | Evaluation factories reject `LlmResponseHash`; JSONL is audit-only |
| Security | Never commit `.env`; example only in `.env.example` |

**Autorater prompt provenance:** local
`prompts/autorater_mem0_v1.txt` is adapted from Mem0's pinned
[`ACCURACY_PROMPT`](https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py)
and the paper's
[Appendix A](https://arxiv.org/abs/2504.19413) “Prompt Template for LLM as a
Judge.”

**No-cache invariant:** reader, teacher, and autorater factories reject
non-null `llm_response_hash`. `run.py` clears generated artifacts for a run id
and starts from question one; `predictions.jsonl` is audit-only. Autorater
clears and rewrites its pack and rejects source rows marked `cached=true`.

### Planned (not implemented)

- Anthropic Claude as alternate fixed answer reader or judge.
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
  prompts/*.txt  {memory} + {question}   ← FREEZE for fair memory-condition comparisons
        │
        ▼
  Reader (OpenAI / mock, temp=0)         ← FREEZE for fair memory-condition comparisons
        │
        ▼
  predicted_answer
        │
        ├──────────────► metrics (EM, token F1, LoCoMo F1)  ← deterministic
        │
 gold ──┴──────────────► Autorater (GPT-4o, Mem0 prompt)    ← online evaluator
        │
        ▼
  experiments/<run_id>/  CSV · JSONL · plots · run_meta
  experiments/<run_id>/autorater/  Mem0 F1/BLEU-1/J · tables · latency plots
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

| Condition id | What the answer LLM receives | Research question |
|--------------|------------------------------|-------------------|
| `raw_chunks` | Full raw dialog turns (chronological), optional char cap | Does structure help at all? |
| `session_summaries` | LoCoMo-released session summaries (dataset “memory”) | How strong is a structured session-memory bank? |
| `teacher_session_summaries` | Per-session summaries from one teacher LLM | Does a live teacher beat released summaries? Swap teacher model within a family as a robustness check. |
| `top1_teacher` (future) | Top-1 of K teacher memories | Selection enough? |
| `whole_memory_aggregation` (future) | Aggregated whole memories | Synthesis enough? |
| `claim_fusion` (future) | Claim-level fused + validated store (+ retrieve) | Fine-grained fusion win? |

Default memory still uses **provided** LoCoMo session summaries (`session_summaries`). `teacher_session_summaries` is the live single-teacher replacement (same inject path). Multi-teacher fusion is still out of scope.

Aliases for convenience (legacy numbered ids still resolve):

| Alias | Resolves to |
|-------|-------------|
| `raw` / `raw_dialog` / `c0_raw` | `raw_chunks` |
| `session_summary` / `c1_session_summary` | `session_summaries` |
| `teacher` / `c1_teacher` | `teacher_session_summaries` |

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
| CLI | `--memory raw_chunks` or `--memory session_summaries` |
| Logged as | `memory_type` on each prediction + `run_meta` |

**Add a new condition**

1. Implement a class with a descriptive `name` (e.g. `top1_teacher`), not a numbered code.
2. Register it in `get_memory_builder`.
3. Prefer writing long intermediates to `experiments/<run_id>/memories/` later; still end by filling `Memory.text`.
4. Run with the **same** `--prompt` / reader model as other memory conditions.

**Teacher (`teacher_session_summaries`):** YAML `teacher.model` (or `--teacher-model`) selects the write-path LLM. Logged on `Memory.teacher_model`, `run_meta.json`, and each prediction row. Do not change `reader.model` in the same comparison if you want the delta attributed to the teacher.

**Reader-model robustness:** YAML `reader.model` / `--model` / `configs/session_summaries_reader_gpt56_luna.yaml`. Compare with `scripts/compare_cross_model.py --axis reader`. Do not mix with a memory-condition claim.

### 4.3 True multi-teacher write path (later, still middle)

Planned package (not required for the current draft):

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
| OpenAI / mock | `readers.py` | Optional `LlmResponseHash` hook (unwired from `run.py`; see `utils/llm_response_hash.py`) |
| Env / keys | `.env` + `env.py` | never commit secrets |
| Scoring | `metrics.py`, `src/metrics/locomo_qa.py` | dual: SPEC + LoCoMo F1 |
| Offline rescore | `python -m src.locomo_eval.offline_evaluate ...` | string metrics only; no API; not an LLM autorater |
| Online autorater | `python -m scripts.analysis.run_benchmark --run ...` | Mem0 prompt, GPT-4o default, category 5 skipped; fresh non-appending report every invocation |
| Autorater literature pins | `mem0_baselines.py` | Mem0 paper Tables 1–2; comparison only, not local Mem0 re-runs |
| Two-run compare | `scripts/analysis/compare_predictions.py` | paired LoCoMo F1 boxplot + histograms |
| Audit pack | `report.py` → `experiments/<run_id>/` | CSV/JSON/plots |

---

## 5. How to compare raw_chunks vs session_summaries

Freeze bottom, vary middle only:

```bash
# Same max_questions, model, prompt; different memory + run_id
python -m src.locomo_eval.run --config configs/raw_chunks.yaml --max-questions 20 --run-id cmp_raw_chunks_n20
python -m src.locomo_eval.run --config configs/session_summaries.yaml --max-questions 20 --run-id cmp_session_summaries_n20

# Side-by-side metrics table
python scripts/compare_full_runs.py \
  --runs experiments/cmp_raw_chunks_n20 experiments/cmp_session_summaries_n20 \
  --out experiments/compare_raw_chunks_session_summaries
```

Writes `overall.csv`, `by_category.csv`, `paired_questions.csv`, `SUMMARY.md`, `plots/` (including LoCoMo F1 **boxplot** + side-by-side **histograms**), and memory / request-hash **sanity** (`fraction_same_llm_request_hash` ≈ 0 means conditions asked the reader different things; not a live store lookup).

Two-pack LoCoMo F1 plots alone (no sandwich SUMMARY):

```bash
python -m scripts.analysis.compare_predictions \
  --a experiments/cmp_raw_chunks_n20 --b experiments/cmp_session_summaries_n20 \
  --out experiments/compare_raw_chunks_session_summaries
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
- [ ] Both condition runs regenerated without cache/resume; all prediction rows have `cached=false`.

---

## 6. Control panel (frozen vs changeable)

| Piece | Freeze for the main memory-condition table? | Experiment instead? |
|-------|------------------------------|---------------------|
| Dataset + question IDs | Yes | No |
| Answer prompt | Yes (after lock) | Only as a separate meta-setup |
| Answer model + decode | Yes | Model swap = new setup label |
| Metrics | Always | No |
| **MemoryBuilder / write / store** | No | **Yes — main axis** |
| Retriever + token budget | Policy fixed across conditions | Budget ablations as a second axis |
| Report columns | Prefer stable | Additive fields OK |

---

## 7. Suggested growth order

1. `raw_chunks` vs `session_summaries` under frozen prompt/model (this draft).  
2. Lock short-answer prompt if needed; re-run both conditions.  
3. Persist mid-layer JSON per sample.  
4. Live `teacher_session_summaries` (API generation → `Memory.text`).  
5. Later `top1_teacher` / `whole_memory_aggregation` / `claim_fusion` + fixed retriever budget.  

---

## 8. File index (engineering)

```text
configs/baseline.yaml              # default (session_summaries)
configs/raw_chunks.yaml
configs/session_summaries.yaml
configs/session_summaries_reader_gpt56_luna.yaml  # session_summaries + GPT-5.6 Luna reader
configs/teacher_session_summaries.yaml            # live single teacher
prompts/qa_v1.txt
prompts/teacher_session_v1.txt
src/locomo_eval/models.py          # model catalog / API kwargs
src/locomo_eval/teachers.py        # Mock + OpenAI teacher
src/locomo_eval/memory.py          # raw_chunks / session_summaries / teacher_session_summaries
src/locomo_eval/utils/llm_request_hash.py   # SHA-256 of the intended LLM request (offline distinctness)
src/locomo_eval/utils/llm_response_hash.py  # LLM reply memo (unwired; future optimization)
src/locomo_eval/run.py             # wires builder → reader → report
scripts/compare_full_runs.py            # two memory conditions side-by-side
scripts/analysis/compare_predictions.py  # two-pack LoCoMo F1 boxplot + histograms
scripts/analysis/run_benchmark.py        # Mem0 F1/BLEU-1/J + tables/plots
src/locomo_eval/autorater.py             # GPT-4o / mock CORRECT-WRONG judge
src/locomo_eval/mem0_metrics.py          # Mem0 lexical metrics + latency
src/locomo_eval/mem0_baselines.py        # published Table 1–2 pins
scripts/compare_cross_model.py     # reader/teacher model robustness
docs/reports/engineering_notebook.md  # this file
```
