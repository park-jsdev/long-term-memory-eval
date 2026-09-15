# Claim audit completeness (`feat/analysis-pipeline`)

**Date:** 2026-09-15  
**Commit:** `fd0e900` *add claims audits*  
**Contract:** `audit_pack.v2` (`docs/schemas/experiment_pack.md`)

This is an evaluation of the eight-item **audit of claims** layer, not a new
feature. Call traces record that an LLM ran. Claim audit records **what
entered `{memory}`** and **who is responsible** for each injected item, so a
reviewer can attribute a QA outcome without assuming from input/output.

On-disk smoke: `python -m pytest tests/test_claim_audit.py tests/test_experiment_pack.py tests/test_regressions.py tests/test_teacher_orchestrator.py tests/test_run_isolation.py -q` (isolation + optional-field locks in claim-audit/pack/regressions, 2026-09-15).

## Verdict

**Complete as a sandwich dump contract for teacher-graph and retrieve
conditions. Not complete as construction-once attribution for every memory
method.**

The eight items exist as files, are written from `run.py`, load through
`audit_loader`, and have unit + one mock teacher-graph end-to-end check.
They are enough to stop guessing *which injected triple came from which
teacher* on `teacher_graph` / `pooled_teacher_graph` / `fused_teacher_graph`.
They do **not** yet give the same join for dump-all memories, and teacher
write-path logs can duplicate when the same conversation is built once per
question.

Roadmap `claim_fusion` (a later fusion *method*) is unrelated. This layer
audits claims already in `{memory}`.

## Checklist

| # | Requested item | Status | On disk | Complete when |
|---|----------------|--------|---------|---------------|
| 1 | Question → memory item → teacher | **Done for graphs; partial elsewhere** | `memory/lineage.jsonl` | Teacher-graph: `question_id` → `edge_id` → `proposed_by`. Retrieve: selected candidates only (usually no teacher). Dump-all: `source_ids` with empty `proposed_by` / `text_preview`. |
| 2 | Graph ingest after fusion | **Done for orchestrator graphs** | `memory/graph/ingest.jsonl` | MERGE / invalidate / skip_dup / node ops **after** `fuse_proposals`. Mem0g *index* ingest stays in `mem0_index/`, not this sandwich pack. |
| 3 | Retrieve ranks, not just winners | **Done where retrieve exists** | `memory/retrieve_ranks.jsonl` | RAG, mem0/mem0g, openai_memory (all selected), preprocess-index raw/summaries. Empty for dump-all `raw_chunks` / `session_summaries` / `full_context` / teacher graphs (inject all valid edges). |
| 4 | Teacher quality stats | **Done as software stats** | `memory/teachers/quality.json` | Parse ok / fallback, entity/relation yield, fusion keep rate. Not an LLM factuality judge. Inflated if the graph is rebuilt once per question. |
| 5 | Frozen config in the run dir | **Done** | `config.source.yaml`, `config.resolved.yaml` | Source YAML copy + YAML plus CLI overrides. `rng_seed` lives in the YAML, not a dedicated `run_meta` field. |
| 6 | Cost rollup | **Done for reader + teacher tokens** | `cost.json` | Pinned OpenAI list prices (`pricing_as_of` 2026-09-10). Claude / DeepSeek / embeddings: tokens or omitted, `usd=null`. Not an invoice. |
| 7 | Teacher input session text | **Done** | `memory/teachers/sessions/by_sample/<id>/session_<k>.txt` | Exact text the teacher saw, plus sha256 on the call row. |
| 8 | Human memory/results report | **Done as pointers + sample** | `SUMMARY.md` | Sandwich pins, cost, quality table, ingest/rank counts, “how to audit one claim,” first eight lineage rows. Not a full item browser. |
| 9 | Attribution and LLM roles | **Done** | `ATTRIBUTION.md`, `attribution.jsonl` | Each LLM call, sandwich role (`reader` / `teacher` / `teacher_graph`), claims made (triples, summaries, predicted answers), joined to fusion `kept` and lineage injection. |

## Calls vs claims

```text
CALL AUDIT (already existed)
  reader/traces.jsonl          the answer LLM ran
  memory/teachers/calls.jsonl  a teacher ran (reasoning, triples, usage)

CLAIM AUDIT (this branch)
  memory/lineage.jsonl         this question injected these items from these teachers
  memory/retrieve_ranks.jsonl  these candidates lost (or were never a contest)
  memory/graph/ingest.jsonl    fusion output actually MERGEd / invalidated
  memory/teachers/quality.json parse / yield / keep (not “was the triple true”)
  memory/teachers/sessions/    the session text that produced the triple
  attribution.jsonl            this LLM call, this role, these claims
  ATTRIBUTION.md               human call → role → claims report
  config.*.yaml + cost.json    what ran, what it cost
  SUMMARY.md                   human start: follow one question without re-running
```

## Coverage by memory condition

| Condition | Lineage → teacher | Ingest after fusion | Ranks incl. losers | Session text | Quality / cost |
|-----------|-------------------|---------------------|--------------------|--------------|----------------|
| `teacher_graph` / `pooled_*` / `fused_*` | Yes (`edge_id` + fusion `proposed_by`) | Yes | No retrieve (all valid edges injected) | Yes | Yes (see duplication note) |
| `teacher_session_summaries` | Yes (`session_N_teacher` → call `teacher_id`); preview empty | n/a | n/a | Yes | Yes (duplication note) |
| `rag` | Selected chunks; no teacher | n/a | Yes | n/a | Reader cost only |
| `mem0` / `mem0g` | Selected facts/edges; no teacher | Index pack, not sandwich | Yes | n/a | Reader (+ embed uncounted) |
| `openai_memory` | All entries selected | n/a | Rank list; no losers (retrieve-all) | n/a | Reader cost |
| `raw_chunks` / `session_summaries` via preprocess + `top_k` | Session units; no teacher | n/a | Yes | n/a | Reader cost |
| Same without preprocess index, or `full_context` | `source_ids` only; empty teacher/preview | n/a | File omitted | n/a | Reader cost |

## Gaps that still make you assume

These are the remaining places a reviewer would still guess.

1. **Write path re-runs per question.** `OrchestratedGraphMemoryBuilder.build` and
   `TeacherSessionMemoryBuilder.build` reconstruct memory on every QA call.
   `call_log` / `fusion_log` / `ingest_log` append again. On a 5-question
   conversation, `quality.json` and teacher `cost.json` can be ~5× construction.
   Session `.txt` files de-dupe; the JSONL counts do not. Mock e2e uses one
   question, so tests do not catch this.

2. **Dump-all lineage is thin.** `full_context` and default `raw_chunks` /
   `session_summaries` write lineage rows from `source_ids` with empty
   `text_preview` and empty `proposed_by`. You still have `{memory}` text and
   dia_ids; you do not get a per-item preview table.

3. **Cost is not the full bill.** Embeddings at RAG/mem0 retrieve time are not
   passed into `cost_rollup`. Unknown models stay token-only. USD pins are
   list prices, not invoices.

4. **Quality ≠ truth.** Keep rate is fusion software (`kept` vs extracted
   relation count). Parse ok ≠ the triple is grounded in the session text.
   Session text is dumped so a human can check that next.

5. **`eval_pipeline` does not list teacher-graph methods.** Those runs still
   get the claim pack via `python -m src.locomo_eval.run --config configs/…`.
   Paper methods (`rag`, `mem0`, …) inherit the same `run.py` dump.

## How to audit one claim (researcher)

1. Open `experiments/<run_id>/SUMMARY.md`.
2. `ATTRIBUTION.md` — LLM call → role → claims (machine: `attribution.jsonl`).
3. Pick a `question_id` from `predictions.jsonl`.
4. `memory/lineage.jsonl` — injected `item_id` and `proposed_by`.
5. `memory/teachers/sessions/` — session text that teacher saw.
6. `memory/teachers/fusion.jsonl` — votes / `kept` for that triple.
7. `memory/graph/ingest.jsonl` — `add_edge` / `invalidate` / `skip_dup`.
8. `memory/retrieve_ranks.jsonl` — losers (retrieve conditions only).

Load without re-running:

```python
from src.locomo_eval.experiments.audit_loader import load_sandwich_audit

audit = load_sandwich_audit("experiments/<run_id>")
audit.lineage_for(question_id="…")
audit.retrieve_ranks_for(question_id="…")
audit.fusion_kept_for(sample_id="…")
audit.teacher_calls_for("openai")
audit.attribution_for(role="teacher_graph")
```

## Tests that lock this

| Test | What it pins |
|------|----------------|
| `tests/test_claim_audit.py` | Losers in mem0/RAG ranks; ingest `skip_dup`; lineage join `edge_id` + `proposed_by`; parse/keep stats; OpenAI USD pin; dump writes SUMMARY / ATTRIBUTION / cost / lineage / session text; teacher/reader call → role → claims |
| `tests/test_teacher_orchestrator.py` (mock pipeline) | Live-shaped pack: `SUMMARY.md`, `lineage.jsonl`, `ingest.jsonl`, `sessions.jsonl`, `audit_pack.v2` |
| `tests/test_experiment_pack.py` | `audit_loader` indexes calls / fusion / QA pack keys |

Not locked: multi-question teacher-graph de-dupe; dump-all lineage previews;
embedding cost; `eval_pipeline --method teacher_graph`.

## What this is not

- Not Mem0 Platform / paper Table 1–2 J.
- Not claim-level *fusion* (`claim_fusion` on the condition roadmap).
- Not an LLM validator that the triple is true.
- Not a UI. `SUMMARY.md` is the human report.
