# Trace: 2026-08-26 — before Mem0-parity baseline

Behavioral snapshot of the prior AGENTS/HUMANS policy:

- `configs/baseline.yaml` used `gpt-4.1-mini`, `prompts/qa_v1.txt`, a
  system-plus-user message layout, and `max_tokens: 64`.
- `configs/autorater.yaml` used GPT-4o with a 256-token limit.
- GPT-4o-mini was documented only as the released Mem0 judge option.
- The session-summary memory builder was already the default condition.

The replacement policy pins the default answer and judge controls to Mem0's
released GPT-4o-mini setup while retaining `session_summaries` as this
repository's memory condition. It does not claim Mem0 extraction/update parity.
