# AGENTS.md — agent operating notes (v0.1)

**Audience:** coding agents working in this repo.  
**Length target:** 2–3 pages.  
**Update rule:** change this file on every meaningful behavior or layout change; snapshot to `docs/agent/traces/` first.

---

## Mission

Research pipeline for long-term conversational memory on **LoCoMo**, eventually multi-teacher memory construction with a **sandwich design** (fixed data + fixed answer/eval; variable middle = memory method).

**Current phase:** HLD (i) session-block preprocess, **Mem0 / Mem0g write-index**, and HLD (ii) **TeacherOrchestrator** (single teacher, naive pool, majority-vote fusion) writing the **locked Mem0 graph**. Read path: `raw_chunks` / `session_summaries` / `teacher_session_summaries` / `teacher_graph` / `pooled_teacher_graph` / `fused_teacher_graph` / `mem0` / `mem0g`, plus a Mem0-style **LLM autorater**. Do not claim paper J (66.88 / 68.44) from the write-index or teacher graphs.
See `docs/reports/engineering_notebook.md` for freeze/extend rules.

Teacher fusion is a `MemoryBuilder` + orchestrator claim. Distilled memory later is a new `GraphMemory` subclass (freeze extract when attributing the graph). `Mem0GraphMemory` is the locked graph schema whenever a condition uses a graph.

---

## North star (later)

```
Write: conversation → teachers → fusion/validate → memory store
Read:  question → retrieval → fixed answer LLM → LoCoMo evaluator
```

Conditions planned: `raw_chunks`, `session_summaries`, `teacher_session_summaries`, `teacher_graph`, `pooled_teacher_graph`, `fused_teacher_graph`, `mem0`, `mem0g`, later `top1_teacher`, `whole_memory_aggregation`, `claim_fusion`. Use these ids in logs — do not number conditions C0, C1, …

**Now:** Default **reader** is Mem0-parity (`gpt-4o-mini` + `prompts/qa_mem0_v1.txt`) for `raw_chunks`, `session_summaries`, `mem0`, `mem0g`, and teacher-graph conditions. `mem0_baseline.yaml` pins Mem0's released answer/evaluation controls (GPT-4o-mini, Mem0 answer prompt, GPT-4o-mini judge); that YAML is evaluation parity over `session_summaries` memory, not a substitute for the write-index. Default **writer** for mem0 extract/update is `gpt-4o-mini`. Cheap **teacher plumbing** models: `gpt-4o-mini`, `claude-haiku-4-5`, `deepseek-v4-flash`. Deterministic preprocess dump (`python -m src.locomo_eval.preprocess.run_index`) is the no-LLM write path for `raw_chunks` / `session_summaries`. `mem0` / `mem0g` load a write-index dump and cosine-retrieve top-k NL facts (`top_k=30`). Graph/store RQs freeze writer + reader + this retriever; swap only `GraphMemory`. `teacher_session_summaries` is a live single-teacher *summary* seam. `teacher_graph` is a live single-teacher *graph* seam (interchangeable provider/model). `pooled_teacher_graph` is naive K-teacher pool (equal_weight / random / round_robin). `fused_teacher_graph` is majority-vote fusion into the same Mem0 graph. Reader-model or `qa_v1` swaps remain a **separate** robustness axis.

---

## Repo map (v0.1)

| Path | Role |
|------|------|
| `configs/mem0_baseline.yaml` | CLI default; Mem0-parity reader controls over `session_summaries` memory |
| `configs/raw_chunks.yaml` | Raw dialog memory (`raw_chunks`) |
| `configs/session_summaries.yaml` | LoCoMo session-summary memory with original QA controls |
| `configs/session_summaries_reader_gpt56_luna.yaml` | `session_summaries` memory + GPT-5.6 Luna answer model |
| `configs/teacher_session_summaries.yaml` | Live single-teacher session summaries (`teacher_session_summaries`) |
| `configs/teacher_graph.yaml` | Single interchangeable teacher → locked Mem0 graph |
| `configs/pooled_teacher_graph.yaml` | K cheap teachers, naive pool (`equal_weight` / `random` / `round_robin`) |
| `configs/fused_teacher_graph.yaml` | K teachers, majority-vote fusion skeleton |
| `configs/preprocess.yaml` | Deterministic HLD (i) write-index (no LLM) |
| `configs/mem0.yaml` | Mem0 vector write-index + load/retrieve seam |
| `configs/mem0g.yaml` | Mem0 vector + in-memory graph (`mem0g`) |
| `prompts/qa_mem0_v1.txt` | Pinned released Mem0 answer prompt for baseline parity |
| `prompts/qa_v1.txt` | Alternate short prompt; override only as a separate bottom-layer axis |
| `prompts/teacher_session_v1.txt` | Teacher session-summary prompt |
| `prompts/teacher_graph_v1.txt` | Teacher → entity/relation JSON (Mem0g-shaped) |
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
| `teacher_orchestrator.py` | HLD (ii): session blocks → pool/fuse → `Mem0GraphMemory` (passthrough if no teachers) |
| `memory.py` | MemoryBuilder interface + raw_chunks / session_summaries / teacher_* / mem0 / mem0g |
| `mem0/` | Write-index: ingest pairs, extract, update, vector store, locked `GraphMemory`, dump, `run_index` CLI |
| `fusion.py` | Pool policies + majority-vote fusion (software) |
| `chat.py` | OpenAI / Anthropic / DeepSeek / mock chat callers (teachers) |
| `prompts.py` | Load/render prompt text |
| `readers.py` | OpenAI + Mock readers, temp=0 |
| `models.py` | Model ids / families / Chat Completions kwargs |
| `teachers.py` | Session-summary + graph teachers (openai / anthropic / deepseek / mock) |
| `ping_teachers.py` | Cheap-model plumbing ping (not QA) |
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
python -m pytest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_preprocess_index.py tests/test_mem0_index.py tests/test_evaluation_pipeline.py tests/test_regressions.py tests/test_autorater_sanity.py tests/test_integration_sanity.py tests/test_run_isolation.py tests/test_teacher_orchestrator.py -q
# or (file path avoids a site-packages module named `tests` shadowing this folder)
python -m unittest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_preprocess_index.py tests/test_mem0_index.py tests/test_evaluation_pipeline.py tests/test_regressions.py tests/test_autorater_sanity.py tests/test_integration_sanity.py tests/test_run_isolation.py tests/test_teacher_orchestrator.py

# Compare two prediction sets (offline; LoCoMo F1 boxplot + histograms)
python -m scripts.analysis.compare_predictions --a experiments/cmp_raw_chunks --b experiments/cmp_session_summaries --out experiments/compare_raw_chunks_session_summaries
python scripts/compare_full_runs.py --runs experiments/cmp_raw_chunks experiments/cmp_session_summaries --out experiments/compare_raw_chunks_session_summaries
python scripts/compare_cross_model.py --runs experiments/session_summaries_mini experiments/session_summaries_luna --axis reader --out experiments/compare_reader_mini_luna


# Session-document tables + dataset histograms (no API; needs locomo10.json)
python scripts/export_session_documents.py --data data/raw/locomo10.json --out data/processed
```

Set API keys via repo-root `.env` (`copy .env.example .env`) or shell:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`.
Never commit keys or `.env`. `src/locomo_eval/env.py` loads `.env` at run start / chat-caller init.

# Teacher plumbing ping (cheap models; needs keys unless --mock)
python -m src.locomo_eval.ping_teachers --mock
python -m src.locomo_eval.ping_teachers --providers openai,anthropic,deepseek

# Mock teacher-graph / pool / fusion (no API)
python -m src.locomo_eval.run --config configs/teacher_graph.yaml --reader mock --teacher mock --max-questions 3 --run-id smoke_teacher_graph
python -m src.locomo_eval.run --config configs/pooled_teacher_graph.yaml --reader mock --teacher mock --max-questions 3 --run-id smoke_pooled_teachers
python -m src.locomo_eval.run --config configs/fused_teacher_graph.yaml --reader mock --teacher mock --max-questions 3 --run-id smoke_fused_teachers

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
2. **Orchestrator is software**, not one giant LLM call (`TeacherOrchestrator` + `fusion.py`).
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

Mem0 Platform / Neo4j / Qdrant, claiming paper Table 1–2 J from this OSS clone, mixing a reader-prompt swap into a graph/store claim, validator loop, retrieval budgets as experiments, training/distillation loop, web UI, event-summarization / multimodal tasks. Weighted/learned fusion beyond majority_vote is a later adjustment of `fused_teacher_graph`, not a new schema.

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
- [ ] `tests/test_mem0_index.py`, `tests/test_regressions.py`, `tests/test_run_isolation.py`, and `tests/test_teacher_orchestrator.py` stay green (mock only; no API)
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
