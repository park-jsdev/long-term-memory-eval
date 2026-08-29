# Trace: 2026-08-26 — before Mem0 baseline config rename

The Mem0-parity profile initially replaced the generic
`configs/baseline.yaml`. This rename makes the scope explicit:

- old path: `configs/baseline.yaml`
- new/default path: `configs/mem0_baseline.yaml`
- unchanged semantics: GPT-4o-mini reader, pinned Mem0 answer prompt and
  system-only layout, no explicit token limit, session-summary memory

Historical traces retain the old path as a record of prior repository state.
