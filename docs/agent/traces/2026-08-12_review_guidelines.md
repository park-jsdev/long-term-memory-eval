# Trace: 2026-08-12 — AGENTS.md before review-session guidelines

Snapshot of `docs/agent/AGENTS.md` before adding code/test/comment guidelines
from the MR review session (test naming, one-function tests, unambiguous
names, why-comments, schema pipeline notes). Also clarified cache vs
checkpoint and `run_condition` vs baseline YAML.

---

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

**Now:** `c0_raw` and `c1_session_summary` builders inject alternative `Memory.text` with frozen reader/metrics. C1 draft uses LoCoMo-provided summaries (not a live teacher API yet).

---

## Repo map (v0.1)

| Path | Role |
|------|------|
| `configs/baseline.yaml` | Default config (C1) |
| `configs/c0_raw.yaml` | C0 raw dialog memory |
| `configs/c1_session_summary.yaml` | C1 session-summary memory |
| `prompts/qa_v1.txt` | Fixed answer prompt |
| `docs/reports/engineering_notebook.md` | System map / extension points |
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
| `schemas.py` | Conversation, Question, Memory, Prediction |
| `dataset.py` | Load LoCoMo JSON → objects |
| `memory.py` | MemoryBuilder interface + SessionSummary |
| `prompts.py` | Load/render prompt text |
| `readers.py` | OpenAI + Mock readers, temp=0 |
| `cache.py` | Disk hash cache for API resumes |
| `metrics.py` | EM, token F1, LoCoMo F1 |
| `report.py` | JSONL/CSV/plots |
| `run.py` | CLI: full pipeline |
| `evaluate.py` | CLI: rescore predictions |

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

# Unit tests
python -m pytest tests/test_baseline.py -q
# or
python -m unittest tests.test_baseline
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
2. **Orchestrator is software**, not one giant LLM call (future TeacherOrchestrator modules).
3. **Prefer small pure functions** over frameworks.
4. **Keep metrics dual-reported:** SPEC token F1/EM *and* LoCoMo category F1.
5. **Cache API calls** (`experiments/cache/`); do not delete user caches unless asked.
6. **Plain YAML**, plain JSON loaders, local CSV — no Hydra/W&B required.
7. **Update docs:** after behavior change, copy previous AGENTS/HUMANS into `docs/agent/traces/YYYY-MM-DD_topic.md`, then edit live files.

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
- [ ] Tests for parse, memory, normalize  
- [ ] AGENTS.md + HUMANS.md updated + trace snapshot  

---

## Traceability

- Spec: `docs/agent/SPEC_v1.md`  
- LoCoMo pin: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`  
- Paper: Maharana et al., arXiv:2402.17753  
- Sandwich idea: Bowman et al. 2022 scalable oversight  
