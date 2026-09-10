# Reflection — 2026-08-08

**Scope:** LoCoMo Phase‑1 sandwich scaffold → live OpenAI path → draft C0 vs C1.  
**Audience:** code review + future audit of this tag of the repo.

---

## Changelog (what landed)

| Area | Change |
|------|--------|
| Baseline package | `src/locomo_eval/`: dataset, schemas, memory builders, prompts, OpenAI/mock readers, metrics, report, run/evaluate CLI |
| Metrics | Dual reporting: SPEC EM/token‑F1 + LoCoMo category F1 (`src/metrics/locomo_qa.py`) |
| Audit pack | Per run: `predictions.jsonl/csv`, `metrics.json`, category CSV, `run_meta.json`, plots |
| Secrets | Repo‑root `.env` via `python-dotenv` (`.env` gitignored; `.env.example` committed) |
| Conditions | **C0** `c0_raw` (raw dialog); **C1** `c1_session_summary` (LoCoMo summaries); configs `c0_raw.yaml` / `c1_session_summary.yaml` |
| Compare | `scripts/compare_runs.py` for side‑by‑side metrics |
| Rate limits | 429 backoff, pacing, disk API cache, resume same `--run-id` + checkpoint JSONL |
| Docs | `docs/reports/engineering_notebook.md`, `docs/agent/{AGENTS,HUMANS,SPEC_v1}.md`, `docs/agent/traces/` |

**Explicitly not built yet:** multi‑teacher write path, fusion/validator, live single‑teacher C1, retrieval budget experiments, KD training.

---

## Things we tried

1. **Scaffold first** without forking full LoCoMo repo — pin `locomo10.json` + vendor scoring logic.  
2. **Smoke path:** `mock` reader → then OpenAI (`gpt-4.1-mini`, temp 0) with `--max-questions` 3–20.  
3. **C0 vs C1** as first experimental axis: only swap memory builder; freeze prompt + reader + metrics.  
4. Hit **50 req/day RPD** on free‑tier style org → added resume/cache/backoff rather than inventing fake results.  
5. Clarified **shared `experiments/cache/`** is a store of *hashed prompts* (memory text inside prompt) — not cross‑condition answer reuse.

---

## Findings

| Finding | Implication |
|---------|-------------|
| End‑to‑end I/O works (load → memory → LLM → scores → CSV/plots) | Sandwich bottom is usable |
| Small‑n scores look weak (verbose answers vs span gold; summary lossiness) | Don’t over‑read n≈3; tighten prompt later and re‑run *all* conditions |
| Each uncached QA = 1 API call; full set ~2k Q | Large runs need higher rate limits / multi‑day resume |
| C1 is **dataset summaries**, not a live teacher | Fine as draft seam; rewrite builder later without touching reader/metrics |
| Injection surface is one string: `Memory.text` | Experiments should almost only change builders / future write→format |

---

## How to reproduce

```bash
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py
# .env: OPENAI_API_KEY=...

# Offline plumbing
python -m unittest tests.test_baseline -q
python -m src.locomo_eval.run --config configs/c0_raw.yaml --reader mock --max-questions 5 --run-id smoke_c0
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --reader mock --max-questions 5 --run-id smoke_c1

# Live comparison (freeze model/prompt; vary config/memory only)
python -m src.locomo_eval.run --config configs/c0_raw.yaml --max-questions 20 --run-id cmp_c0_n20
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --max-questions 20 --run-id cmp_c1_n20
# If interrupted (429): re-run SAME --run-id to resume; cache skips done work

python scripts/compare_runs.py --runs experiments/cmp_c0_n20 experiments/cmp_c1_n20 --out experiments/compare_c0_c1
```

**Audit a run:** open `experiments/<run_id>/run_meta.json` (model, prompt, data SHA, memory type) + `predictions.csv` + `metrics.json`.

---

## System at a glance (for review)

```text
FIXED TOP:     LoCoMo JSON → dataset.py
VARIABLE MID:  MemoryBuilder (c0_raw | c1_session_summary | …) → Memory.text
FIXED BOTTOM:  prompts/readers/qa_v1.txt → OpenAIReader → metrics → report
```

**Fair C comparison:** same prompt, model, decode, question subset; differ only `pipeline.memory` / condition config.  
**Extend later:** new builder or write/store→text; do not fork prompts per condition for the main table.

**Map doc:** `docs/reports/engineering_notebook.md`  
**Agent ops:** `docs/agent/AGENTS.md` · **Human runbook:** `docs/agent/HUMANS.md`

---

## Review checklist

- [ ] Changes stay inside memory/write, or intentionally retag a new frozen bottom  
- [ ] `run_meta` still records model, prompt version, data hash, memory type  
- [ ] New condition registered in `get_memory_builder` + has a config  
- [ ] Tests cover parse + new memory builder  
- [ ] Reflections / traces updated when behavior changes  
