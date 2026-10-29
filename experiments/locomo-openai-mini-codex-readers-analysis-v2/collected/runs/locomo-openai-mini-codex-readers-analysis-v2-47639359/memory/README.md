# Memory dump for this run (`full_context`)

- Schema: **memory_io.v1**
- Human doc: `docs/schemas/memory_runtime.md`
- Machine schema: `docs/schemas/memory_io.schema.json`
- This run’s layout: `memory/schema.json`
- Full texts: `memory/by_sample/<sample_id>.txt`
- Per-question texts (when retrieve is question-dependent): `memory/by_question/<question_id>.txt`
- Index: `memory/index.jsonl`
- Writer LLM traces (when used): `memory/writer/`
- Graph snapshot (when used): `memory/graph/`
- Claim lineage: `memory/lineage.jsonl` (question → item → writer)
- Retrieve ranks (losers included): `memory/retrieve_ranks.jsonl`
- Attribution (call → role → claims): `ATTRIBUTION.md` / `attribution.jsonl`

This folder is the audit trail of **exactly** what `{memory}` contained (payload) plus claim links (`lineage.jsonl`, retrieve ranks, graph ingest).
