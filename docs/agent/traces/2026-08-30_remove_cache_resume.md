# Trace: 2026-08-30 — before removing cache/resume

Snapshot of the cache/resume policy in AGENTS.md / HUMANS.md before this change.

- `utils/llm_response_hash.py` (`LlmResponseHash`) was a disk memo of LLM
  replies, keyed by `llm_request_hash`. Implemented but unwired from `run.py`.
  Reader, teacher, autorater, and Mem0 extract/update/embed factories rejected
  a non-null store.
- Prediction rows carried `cached: false`. Autorater refused source rows with
  `cached=true`. Compare reports listed `n_llm_response_hash_hits`, `n_resumed`,
  and `cached_rate_in_predictions`.
- YAML still had `cache_dir` / commented `llm_response_hash_dir`.
- Mem0 and preprocess write-indexes skipped complete samples unless
  `--overwrite` (per-sample JSON resume, not QA resume).
- Answer runs already cleared generated artifacts and started from question one.
  Autorater already regenerated its pack.

Replacement: delete the store and all cache/resume seams. Every pipeline
invocation (QA, autorater, Mem0 index, preprocess index) clears its output
directory and regenerates. `llm_request_hash` stays as an offline distinctness
fingerprint, not a lookup.
