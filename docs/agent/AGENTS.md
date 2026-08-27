# AGENTS.md — agent operating notes (v0.1)

**Audience:** coding agents working in this repo.  
**Length target:** 2–3 pages.  
**Update rule:** change this file on every meaningful behavior or layout change; snapshot to `docs/agent/traces/` first.

---

## Mission

Research pipeline for long-term conversational memory on **LoCoMo**, eventually multi-teacher memory construction with a **sandwich design** (fixed data + fixed answer/eval; variable middle = memory method).

**Current phase:** HLD (i) session-block preprocess plus a **Mem0 / Mem0g write-index** (`extract → ADD/UPDATE/DELETE/NONE`, optional in-memory graph). Read path still has `raw_chunks` / `session_summaries` / `teacher_session_summaries`. Mem0 QA numbers are a later command — do not claim paper J (66.88 / 68.44).  
See `docs/reports/engineering_notebook.md` for freeze/extend rules.

Do **not** implement multi-teacher fusion unless the human expands scope. Later distilled memory is a new `GraphMemory` subclass (freeze extract when attributing the graph). `teacher_orchestrator.py` stays a passthrough seam.

---

## North star (later)

```
Write: conversation → teachers → fusion/validate → memory store
Read:  question → retrieval → fixed answer LLM → LoCoMo evaluator
```

Conditions planned: `raw_chunks`, `session_summaries`, `teacher_session_summaries`, `mem0`, `mem0g`, later `top1_teacher`, `whole_memory_aggregation`, `claim_fusion`. Use these ids in logs — do not number conditions C0, C1, …

**Now:** Default **reader** is Mem0-parity (`gpt-4o-mini` + `prompts/qa_mem0_v1.txt`) for `raw_chunks`, `session_summaries`, `mem0`, and `mem0g`. Default **writer** for mem0 extract/update is `gpt-4o-mini`. Deterministic preprocess dump (`python -m src.locomo_eval.preprocess.run_index`) is the no-LLM write path for `raw_chunks` / `session_summaries`. `mem0` / `mem0g` load a write-index dump and cosine-retrieve top-k NL facts (`top_k=30`). Graph/store RQs freeze writer + reader + this retriever; swap only `GraphMemory`. Override `reader.model` / `pipeline.prompt_path` for a separate axis (`qa_v1`, `gpt-4.1-mini`, Luna YAML).

---

## Repo map (v0.1)

| Path | Role |
|------|------|
| `configs/baseline.yaml` | CLI default; currently `session_summaries`. Standalone (not a parent of other configs) |
| `configs/raw_chunks.yaml` | Raw dialog memory (`raw_chunks`) |
| `configs/session_summaries.yaml` | LoCoMo session-summary memory (same condition as baseline.yaml) |
| `configs/session_summaries_reader_gpt56_luna.yaml` | `session_summaries` memory + GPT-5.6 Luna answer model |
| `configs/teacher_session_summaries.yaml` | Live single-teacher memory (`teacher_session_summaries`) |
| `configs/preprocess.yaml` | Deterministic HLD (i) write-index (no LLM) |
| `configs/mem0.yaml` | Mem0 vector write-index + load/retrieve seam |
| `configs/mem0g.yaml` | Mem0 vector + in-memory graph (`mem0g`) |
| `prompts/qa_mem0_v1.txt` | Default frozen answer prompt (Mem0 ANSWER_PROMPT, `{memory}`+`{question}`) |
| `prompts/qa_v1.txt` | Alternate short prompt; override only as a separate bottom-layer axis |
| `prompts/teacher_session_v1.txt` | Teacher session-summary prompt |
| `prompts/mem0_extract_v1.txt` / `mem0_update_v1.txt` | Mem0 fact extract + ADD/UPDATE/DELETE/NONE (pin: mem0 @ ece7ff6b) |
| `prompts/mem0g_*.txt` | Entity / relation / conflict (pin: mem0 graph @ 69a832dc) |
| `docs/reports/engineering_notebook.md` | System map / extension points |
| `docs/schemas/memory_runtime.md` | Runtime `{memory}` audit |
| `docs/schemas/preprocess_runtime.md` | Session-block preprocess schema (`preprocess_io.v1`) |
| `docs/schemas/mem0_index.md` | Write-index dump schema (`mem0_index.v1`) |
| `src/locomo_eval/` | Baseline package |
| `scripts/compare_full_runs.py` | Sandwich report for two **finished** run packs (tables, SUMMARY, distinctness + F1 plots) |
| `scripts/analysis/` | Reusable offline analyses + plots (no API) |
| `scripts/analysis/compare_predictions.py` | Two prediction JSONLs → paired LoCoMo F1 boxplot + histograms (kernel used by compare_full_runs) |
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
| `utils/llm_response_hash.py` | Disk memo of LLM replies, keyed by `llm_request_hash` (implemented; **not wired** into run.py — future optimization) |
| `utils/llm_request_hash.py` | SHA-256 of the intended LLM request; offline distinctness (not a live store lookup) |
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

# Mem0 / Mem0g write-index (no QA). Mock smoke; live gpt-4o-mini is costly.
python -m src.locomo_eval.mem0.run_index --config configs/mem0.yaml --extractor mock --embedder mock --max-samples 1 --run-id smoke_mem0_index
python -m src.locomo_eval.mem0.run_index --config configs/mem0g.yaml --run-id mem0g_locomo10

# Deterministic preprocess write-index (no LLM). Full locomo10, then 10 reader calls per sanity memory:
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess --eval-questions 10
python -m src.locomo_eval.preprocess.run_index --eval-questions 10 --eval-reader mock --run-id smoke_preprocess

# Unit tests — preprocess (HLD i) + Mem0 index + evaluation (HLD iv) + sandwich regression locks
python -m pytest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_preprocess_index.py tests/test_mem0_index.py tests/test_evaluation_pipeline.py tests/test_regressions.py -q
# or (file path avoids a site-packages module named `tests` shadowing this folder)
python -m unittest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_preprocess_index.py tests/test_mem0_index.py tests/test_evaluation_pipeline.py tests/test_regressions.py

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

A Mem0 write-index under `experiments/<run_id>/mem0_index/` must include
`run_meta.json`, `schema.json`, `index.jsonl`, and `by_sample/<id>/speaker_a.json`
+ `speaker_b.json` (and `graph.json` when `enable_graph`). Skip complete samples
unless `--overwrite`. This is **not** `LlmResponseHash`.

A deterministic preprocess dump under `experiments/<run_id>/preprocess/` must include
`run_meta.json`, `schema.json`, `index.jsonl`, and `by_sample/<id>/sessions.jsonl`
+ `documents.jsonl`. No LLM. Gold answers stay out of the dump.

---

## Design rules for agents

1. **Sandwich:** only change one middle variable per experimental claim. Default reader (`gpt-4o-mini` + `qa_mem0_v1`) and mem0 writer (`gpt-4o-mini` extract/update) stay frozen for graph/store RQs. Reader-model or `qa_v1` swaps are a **different** axis (`compare_cross_model.py` / prompt override).
2. **Orchestrator is software**, not one giant LLM call (future TeacherOrchestrator modules).
3. **Prefer small pure functions** over frameworks.
4. **Keep metrics dual-reported:** SPEC token F1/EM *and* LoCoMo category F1.
5. **Do not wire `LlmResponseHash` yet.** Implementation lives in `src/locomo_eval/utils/llm_response_hash.py`. Mem0 extract/update/embed factories reject a non-null store. Write-index resume is per-sample JSON under `mem0_index/`. QA resume remains `predictions.jsonl`.
6. **Plain YAML**, plain JSON loaders, local CSV — no Hydra/W&B required. Each config file is standalone; `pipeline.memory` is a builder id, not another YAML.
7. **Update docs:** after behavior change, copy previous AGENTS/HUMANS into `docs/agent/traces/YYYY-MM-DD_topic.md`, then edit live files.

---

## Code, tests, and comments

From review. Follow these when adding or renaming code.

**Names.** Unambiguous, justified, and kept current. If a name no longer matches the behavior, rename it (e.g. a generic orchestrator must not be called `run_baseline` when “baseline” is a config). Do not overload research terms across different mechanisms (`LlmResponseHash` vs JSONL resume vs `llm_request_hash`; baseline YAML vs `session_summaries`). Condition ids are descriptive (`raw_chunks`), not numbered (`C0`).

**Comments.** Explain *why* and the surrounding context, not a restatement of the next line. Ambiguous helpers need a one-liner on what they pin for later reproduction (`_git_hash`, `_file_sha256`). Distinguish lookalike layers (`LlmResponseHash` vs JSONL resume vs `llm_request_hash`; `offline_evaluate.py` vs `run.py`; later LLM autoraters vs this string scorer).

**Schemas / data-model classes.** The module docstring should map how types connect and which pipeline step uses them (load → memory → reader → prediction → score). Each class gets a short “what it is / who consumes it” note. Label gold answers as scorer-only (never in the reader prompt).

**Tests.**
- One unit test focuses on one function (`exact_match` tests stay separate from `token_f1` tests).
- Test names include the behavior **and** the expected outcome, e.g. `test_exact_match_returns_one_when_answers_match_after_normalization`.
- Group related cases in a `TestCase` per function or class; do not pile unrelated functions into one method.

**Experiments.** One YAML per `run_locomo_pipeline_with_memory_config` call (`python -m src.locomo_eval.run`). Compare `raw_chunks` vs `session_summaries` with two runs, then `scripts/compare_full_runs.py` or `python -m scripts.analysis.compare_predictions` (no API). Do not fold A vs B into `run.py`. Compare reader or teacher **models** with `scripts/compare_cross_model.py` (also no API); that is a different axis fed into the same two-pack compare.

---

## Out of scope (v0.1)

Multi-teacher fusion, Mem0 Platform / Neo4j / Qdrant, claiming paper Table 1–2 J from this OSS clone, mixing a reader-prompt swap into a graph/store claim, validator loop, training/distillation loop, web UI, event-summarization / multimodal tasks.

Stub remnants (`src/train.py`, `src/distill/`) are deferred KD; do not wire unless requested.

---

## Acceptance checklist for agent PRs

- [ ] Dataset loads without editing source JSON  
- [ ] One-question and full-run share the same command  
- [ ] Predictions JSONL deterministic fields  
- [ ] Metrics include EM, token F1, LoCoMo F1 by category  
- [ ] Memory builder swappable without changing reader/evaluator  
- [ ] Tests for parse, memory, preprocess session blocks, session-document join, Mem0 index (mock), normalize  
- [ ] Model-integration sanity (`test_integration_sanity`: reader swap, teacher family swap, LoCoMo vs SPEC scorer)  
- [ ] `tests/test_mem0_index.py` and `tests/test_regressions.py` stay green (mock only; no API)  
- [ ] AGENTS.md + HUMANS.md updated + trace snapshot  

---

## Traceability

- Spec: `docs/agent/SPEC_v1.md`  
- LoCoMo pin: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`  
- Paper: Maharana et al., arXiv:2402.17753  
- Mem0 protocol: Chhikara et al., arXiv:2504.19413 (architecture clone; not Platform v2 numbers)  
- Sandwich idea: Bowman et al. 2022 scalable oversight  
