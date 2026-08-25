# Trace: 2026-08-25 — before cache-free evaluation enforcement

Behavioral snapshot of the prior AGENTS/HUMANS policy before this change:

- `run.py` did not wire `LlmResponseHash`, but reused finished rows from
  `predictions.jsonl` when invoked again with the same run id.
- `run_meta.json` reported `n_resumed` and `n_new_api_calls`.
- The autorater regenerated its own outputs, but accepted source prediction
  rows whose `cached` field was true.
- Documentation distinguished JSONL resume from a shared response cache.

The replacement policy removes that distinction for evaluation: answer,
teacher, and autorater factories reject response stores; answer runs start
from scratch; autorater rejects cached source rows.
