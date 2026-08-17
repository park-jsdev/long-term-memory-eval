# Reflection — 2026-08-17 (code-review follow-through)

**Scope:** Naming, sandwich contracts, and LLM-cache confound after the 2026-08-12 review guidelines.  
**Audience:** code review + future audit.  
**Source traces:** `docs/agent/traces/2026-08-12_review_guidelines.md` (concerns recorded) and the 2026-08-17 snapshots below.

---

## Code review concerns → what we did

Guidelines from review (names match behavior; do not overload terms; tests = one function + expected outcome; comments say *why*; distinguish lookalike layers) were still only partly applied. This pass closed the remaining overloads in the live pipeline.

| Concern | Why it mattered | What we did | Trace |
|---------|-----------------|-------------|--------|
| `evaluate.py` sounds like an LLM judge | Future autoraters vs today’s string metrics would share a name | Renamed to `offline_evaluate.py`. CLI: `python -m src.locomo_eval.offline_evaluate`. Docs state: EM / token F1 / LoCoMo F1 only; no API | `2026-08-17_offline_evaluate.md` |
| `run_condition` is a generic orchestrator name | Review example: do not call a generic runner `run_baseline`. This function runs **one** memory YAML, not A vs B | Renamed to `run_locomo_pipeline_with_memory_config`. A vs B = two calls + `scripts/compare_runs.py` | `2026-08-17_run_pipeline_rename.md` |
| `test_pipeline_sanity.py` does not name a HLD component | HLD (i) pre-processing (ii) teacher memory (iii) post-processing (iv) evaluation is not locked; do not invent packages | Renamed to `tests/test_evaluation_pipeline.py` (component iv). Parse/C0/C1 stay here as **scorer inputs**. No production class rename until LLD | `2026-08-17_test_evaluation_pipeline.md` |
| `cache` / `ResponseCache` overloaded with JSONL resume | Two stores: hashed LLM replies vs per-run checkpoint. A live cache can **confound** C0 vs C1 if memories accidentally collide (shared prompt → shared hit) | Renamed to `LlmResponseCache`; keys include `pipeline_stage` (`answer_reader` / reserved `teacher` / `autorater`) | `2026-08-17_llm_response_cache.md` |
| Wiring an LLM cache during E2E validation hides bugs and can mix conditions | Goal is to trust the pipeline before optimizing | Moved to `src/locomo_eval/utils/llm_response_cache.py`. **Removed wiring** from `run.py`. Hook kept on `OpenAIReader` / `get_reader` for a later opt-in | `2026-08-17_unwire_llm_response_cache.md` |
| Confound belongs in a **test**, not in the runner | Different conditions must produce different memories → different prompts → different answer-reader keys (no shared hits *if* cache returns) | `TestC0AndC1WouldNotShareAnswerReaderCacheKeys` in `tests/test_evaluation_pipeline.py` until a memory-component test module exists | (live tests; after unwire) |
| Gold must never enter memory or the reader prompt | Sandwich: scorer-only gold | `tests/test_regressions.py` lock with `UNIQ_GOLD_REF_ZZZ` | (live tests) |
| `python -m unittest tests.test_*` can fail | Some conda envs have `site-packages/tests/` | Prefer `python -m unittest tests/test_evaluation_pipeline.py tests/test_regressions.py`; `tests/__init__.py` added | AGENTS command block |

**Not done (on purpose):** production rename of memory builders to match unfinished HLD; live teacher / autorater; re-wiring `LlmResponseCache`.

---

## Reusable artifacts

| Artifact | Reuse |
|----------|--------|
| `src/locomo_eval/utils/` | Home for helpers that are **not** a pipeline step |
| `utils/llm_response_cache.py` | Content-addressed LLM memo for **any** future call site; `pipeline_stage` in the key so stages cannot collide |
| Re-enable recipe (in the module + `run.py` comment) | Construct `LlmResponseCache(...)` and pass `llm_response_cache=` to `get_reader` |
| `tests/test_regressions.py` | Sandwich contracts: one YAML per run, audit pack, dual metrics, gold isolation, resume, unwired cache |
| `tests/test_evaluation_pipeline.py` | Component-iv tests + C0/C1 cache-key sanity (move with memory tests later) |
| `docs/agent/traces/YYYY-MM-DD_*.md` | Snap AGENTS/HUMANS **before** each behavior/name change |
| AGENTS “Code, tests, and comments” | Naming / test-shape rules for later MRs |

---

## Takeaways

1. **Name the mechanism, not the folder habit.** `cache`, `evaluate`, and `run_condition` each covered more than one job. Reviewers (and later agents) will wire the wrong layer.
2. **Lookalike stores need different names and tests.** JSONL resume (`predictions.jsonl`) ≠ LLM response cache. String `offline_evaluate` ≠ LLM autorater.
3. **Optimizations that share answers across requests can confound a memory experiment.** Keep the util; prove C0 ≠ C1 keys in a test; wire the cache only after E2E is trusted.
4. **Do not split packages to match a draft HLD.** Park memory-input tests in the evaluation file until the memory component has its own module.
5. **One YAML, one run.** Comparison stays in `compare_runs.py` (offline). `run.py` is not an experiment orchestrator.
6. **Trace first, then edit live AGENTS/HUMANS.** The 2026-08-17 files are the audit trail for this review pass.

---

## Code changes and impact (from traces)

```text
FIXED TOP:     LoCoMo JSON → dataset.py
VARIABLE MID:  MemoryBuilder (c0_raw | c1_session_summary) → Memory.text
FIXED BOTTOM:  qa_v1 → OpenAIReader (no LlmResponseCache) → string metrics → report
OFFLINE:       offline_evaluate.py (rescore JSONL)
UTIL (dark):   utils/llm_response_cache.py  — implemented, not in the experiment path
```

| Change | Impact on experiments |
|--------|------------------------|
| `evaluate.py` → `offline_evaluate.py` | Old `python -m src.locomo_eval.evaluate` no longer works. Rescore command is the new module path. Predictions are not rewritten. |
| `run_condition` → `run_locomo_pipeline_with_memory_config` | Callers and docs updated. Still one config → one `experiments/<run_id>/`. |
| `test_pipeline_sanity.py` → `test_evaluation_pipeline.py` | Same tests; file name matches HLD (iv). |
| `cache.py` / `ResponseCache` → `utils/llm_response_cache.py` / `LlmResponseCache` | Payload now includes `pipeline_stage` (would invalidate old `experiments/cache/*.json` hashes **if** re-wired). |
| Unwire from `run.py` | Every unanswered question hits the API/mock. Resume still skips finished JSONL rows. `run_meta` no longer records cache hits/dir. |
| YAML `llm_response_cache_dir` commented | Config no longer looks like the cache is on. |
| C0 vs C1 key tests | Fails if builders emit the same memory/prompt for the same question (the actual confound). Does **not** turn the cache on. |

**How to re-enable the cache later:** `src/locomo_eval/utils/llm_response_cache.py` docstring, steps 1–3. Then keep the C0/C1 key test green.

**How to run tests now:**

```bash
python -m unittest tests/test_evaluation_pipeline.py tests/test_regressions.py
```

---

## Trace index (this pass)

| File | Taken immediately before |
|------|--------------------------|
| `docs/agent/traces/2026-08-17_offline_evaluate.md` | `evaluate.py` rename |
| `docs/agent/traces/2026-08-17_run_pipeline_rename.md` | `run_condition` rename |
| `docs/agent/traces/2026-08-17_test_evaluation_pipeline.md` | test file rename |
| `docs/agent/traces/2026-08-17_llm_response_cache.md` | `ResponseCache` → `LlmResponseCache` |
| `docs/agent/traces/2026-08-17_unwire_llm_response_cache.md` | move to `utils/` + disconnect from `run.py` |

Earlier related: `2026-08-12_review_guidelines.md` (rules), `2026-08-12_test_pipeline_sanity.md` (`test_baseline` → `test_pipeline_sanity`).

---

## Review checklist (this tag)

- [ ] Names still match the job (`offline_evaluate`, `run_locomo_pipeline_with_memory_config`, `LlmResponseCache`)
- [ ] Live pipeline does **not** pass a cache to `get_reader`
- [ ] C0 vs C1 answer-reader keys still differ in `test_evaluation_pipeline.py`
- [ ] `test_regressions.py` still green (audit pack, gold lock, resume, unwired cache)
- [ ] New LLM call sites use a new `pipeline_stage` if/when cache is re-wired
- [ ] AGENTS/HUMANS traced before the next behavior change
