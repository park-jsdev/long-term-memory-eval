# AGENTS.md — agent operating notes (v0.1)

**Audience:** coding agents working in this repo.  
**Length target:** 2–3 pages.  
**Update rule:** change this file on every meaningful behavior or layout change; snapshot to `docs/agent/traces/` first.

---

## Mission

Research pipeline for long-term conversational memory on **LoCoMo**, eventually multi-teacher memory construction with a **sandwich design** (fixed data + fixed answer/eval; variable middle = memory method).

**Current phase:** end-to-end read path with **draft C0 vs C1** memory builders.  
See `docs/reports/engineering_notebook.md` for freeze/extend rules.

Do **not** implement multi-teacher fusion or claim schema unless the human expands scope.

---

## North star (later)

```
Write: conversation → teachers → fusion/validate → memory store
Read:  question → retrieval → fixed answer LLM → LoCoMo evaluator
```

Conditions planned: C0 raw/chunk, C1 single teacher, C2 top-1 routing, C3 whole-memory aggregation, C4 claim-level fusion.

**Now:** `c0_raw` and `c1_session_summary` builders inject alternative `Memory.text` with frozen reader/metrics. `c1_teacher` is a live single-teacher seam (swap `teacher.model` within a family). Dataset-provided C1 summaries remain the default C1. Reader model can also be swapped (`gpt-4.1-mini` vs `gpt-5.6-luna`) as a **separate** robustness axis — do not mix that with a C0 vs C1 memory claim.

---

## Repo map (v0.1)

| Path | Role |
|------|------|
| `configs/baseline.yaml` | CLI default; currently C1. Standalone (not a parent of c0/c1) |
| `configs/c0_raw.yaml` | C0 raw dialog memory |
| `configs/c1_session_summary.yaml` | C1 session-summary memory (same condition as baseline.yaml) |
| `configs/c1_reader_gpt56_luna.yaml` | C1 memory + GPT-5.6 Luna answer model |
| `configs/c1_teacher.yaml` | Live single-teacher memory (`c1_teacher`) |
| `prompts/qa_v1.txt` | Fixed answer prompt |
| `prompts/teacher_session_v1.txt` | Teacher session-summary prompt |
| `docs/reports/engineering_notebook.md` | System map / extension points |
| `src/locomo_eval/` | Baseline package |
| `scripts/compare_cross_model.py` | Cross-model robustness (reader or teacher axis) |
| `src/metrics/locomo_qa.py` | Official LoCoMo category F1 |
| `data/raw/locomo10.json` | Dataset (gitignored; fetch) |
| `experiments/<run_id>/` | Human-auditable run pack |
| `docs/agent/SPEC_v1.md` | Phase 1 requirements |
| `docs/agent/HUMANS.md` | Human-facing brief |
| `docs/agent/traces/` | Doc version history |

### Package modules (`src/locomo_eval/`)

| File | Responsibility |
|------|----------------|
| `schemas.py` | Conversation, Question, Memory, Prediction |
| `dataset.py` | Load LoCoMo JSON → objects |
| `memory.py` | MemoryBuilder interface + C0 / C1 / c1_teacher |
| `prompts.py` | Load/render prompt text |
| `readers.py` | OpenAI + Mock readers, temp=0 |
| `models.py` | Model ids / families / Chat Completions kwargs |
| `teachers.py` | Single teacher (session summaries) |
| `cache.py` | Content-addressed API memo (not the run checkpoint) |
| `metrics.py` | EM, token F1, LoCoMo F1 |
| `report.py` | JSONL/CSV/plots |
| `run.py` | CLI: one condition from a YAML (`run_condition`) |
| `evaluate.py` | CLI: rescore stored predictions (no API) |

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

# Rescore only
python -m src.locomo_eval.evaluate --predictions experiments/<run_id>/predictions.jsonl

# Unit tests (parse, memory builders, metrics — not C1-only)
python -m pytest tests/test_pipeline_sanity.py tests/test_integration_sanity.py -q
# or (file path avoids a site-packages module named `tests` shadowing this folder)
python -m unittest tests/test_pipeline_sanity.py tests/test_integration_sanity.py

# Cross-model robustness (after two runs that differ by reader or teacher model)
python scripts/compare_cross_model.py --runs experiments/c1_mini experiments/c1_luna --axis reader --out experiments/compare_reader_mini_luna
python scripts/compare_cross_model.py --runs experiments/teacher_luna experiments/teacher_terra --axis teacher --out experiments/compare_teacher_family
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

---

## Design rules for agents

1. **Sandwich:** only change one middle variable per experimental claim later. v0.1 keeps reader prompt and metrics fixed for C0 vs C1. Reader-model and teacher-model swaps are a **different** axis (`compare_cross_model.py`).
2. **Orchestrator is software**, not one giant LLM call (future TeacherOrchestrator modules).
3. **Prefer small pure functions** over frameworks.
4. **Keep metrics dual-reported:** SPEC token F1/EM *and* LoCoMo category F1.
5. **Memoize API calls** (`experiments/cache/`, content-addressed). Per-run resume is `predictions.jsonl`. Do not delete user caches unless asked.
6. **Plain YAML**, plain JSON loaders, local CSV — no Hydra/W&B required. Each config file is standalone; `pipeline.memory` is a builder id, not another YAML.
7. **Update docs:** after behavior change, copy previous AGENTS/HUMANS into `docs/agent/traces/YYYY-MM-DD_topic.md`, then edit live files.

---

## Code, tests, and comments

From review. Follow these when adding or renaming code.

**Names.** Unambiguous, justified, and kept current. If a name no longer matches the behavior, rename it (e.g. a generic orchestrator must not be called `run_baseline` when “baseline” is a config). Do not overload research terms across different mechanisms (`cache` vs run checkpoint; baseline YAML vs C0/C1).

**Comments.** Explain *why* and the surrounding context, not a restatement of the next line. Ambiguous helpers need a one-liner on what they pin for later reproduction (`_git_hash`, `_file_sha256`). Distinguish lookalike layers (`ResponseCache` vs JSONL resume; `evaluate.py` vs `run.py`).

**Schemas / data-model classes.** The module docstring should map how types connect and which pipeline step uses them (load → memory → reader → prediction → score). Each class gets a short “what it is / who consumes it” note. Label gold answers as scorer-only (never in the reader prompt).

**Tests.**
- One unit test focuses on one function (`exact_match` tests stay separate from `token_f1` tests).
- Test names include the behavior **and** the expected outcome, e.g. `test_exact_match_returns_one_when_answers_match_after_normalization`.
- Group related cases in a `TestCase` per function or class; do not pile unrelated functions into one method.

**Experiments.** One YAML per `run.py` call. Compare C0 vs C1 with two runs, then `scripts/compare_runs.py` (no API). Compare reader or teacher **models** with `scripts/compare_cross_model.py` (also no API).

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
- [ ] Tests for parse, memory, normalize, **and** model-integration sanity (`test_integration_sanity`: reader swap, teacher family swap, LoCoMo vs SPEC scorer)  
- [ ] AGENTS.md + HUMANS.md updated + trace snapshot  

---

## Traceability

- Spec: `docs/agent/SPEC_v1.md`  
- LoCoMo pin: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`  
- Paper: Maharana et al., arXiv:2402.17753  
- Sandwich idea: Bowman et al. 2022 scalable oversight  
