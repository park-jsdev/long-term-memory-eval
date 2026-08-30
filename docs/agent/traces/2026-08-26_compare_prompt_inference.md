# Trace: 2026-08-26 — before compare prompt inference

Comparison CLIs previously defaulted `--prompt` to `prompts/qa_v1.txt`.
That could rebuild the wrong request hashes for runs produced by the
Mem0-parity default.

The new behavior infers the frozen prompt path from each run's
`run_meta.json`, rejects mismatched prompt paths, and retains `--prompt` as an
explicit override.
