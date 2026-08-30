# Trace: 2026-08-30 — before dropping llm_request_hash

Snapshot of live policy before this change:

- `utils/llm_request_hash.py` hashed Chat Completions payloads. Live
  `OpenAIChatCaller.complete` recorded `llm_request_hash` in call meta.
  Compare scripts rebuilt hashes from stored `memory_text` for
  `fraction_same_llm_request_hash`.
- `regenerated_from_scratch: true` was a no-op audit flag on QA, autorater,
  and write-index metadata.
- Call sites passed `request_extra` (`pipeline_stage`, prompt body) only so
  the hash payload could be rebuilt offline.

Replacement: delete the hash module. Compare scripts keep memory-text /
answer distinctness. Isolation is the file reset at the start of each run.
