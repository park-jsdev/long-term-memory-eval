# Trace: 2026-08-26 — before Mem0 write-index

Snapshot of live agent docs before adding `src/locomo_eval/mem0/` (extract +
ADD/UPDATE/DELETE/NONE + in-memory graph dumps, `mem0`/`mem0g` builders).

Previous **Current phase** (AGENTS.md): end-to-end read path with
`raw_chunks` vs `session_summaries`; teacher_orchestrator passthrough; do not
implement multi-teacher fusion or claim schema.

Previous **North star Now** (AGENTS.md): those two builders plus
`teacher_session_summaries`; dataset session summaries as default.

Previous **Out of scope**: multi-teacher, claim fusion, validator loop,
retrieval budgets as experiments, training/distillation, web UI,
event-summarization / multimodal.

Previous HUMANS.md: HLD (i) session blocks unwired from `run.py`; memory
conditions were raw_chunks / session_summaries / teacher_session_summaries
only; no Mem0 index command.

Full pre-change files: `docs/agent/AGENTS.md` and `docs/agent/HUMANS.md` at
the commit immediately before this slice on `feat/preprocessing`.
