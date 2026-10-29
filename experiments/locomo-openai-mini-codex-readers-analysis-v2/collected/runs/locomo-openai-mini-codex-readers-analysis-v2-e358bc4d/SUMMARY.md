# Run audit: `locomo-openai-mini-codex-readers-analysis-v2-e358bc4d`

This folder is an **audit of claims** (what entered `{memory}` and who
proposed it), not only an audit of API calls.

## Sandwich

- **memory:** `session_summaries`
- **reader:** `codex/gpt-4o-mini` (prompt `qa_mem0_v1`)
- **agent:** `codex` persist=False tools=native
- **writer:** (none)
- **n questions:** 1986
- **LoCoMo F1:** 0.248
- **token F1 / EM:** 0.2392 / 0.0685

## LLM roles

- `ATTRIBUTION.md` — each LLM call, the role it played, and the claims it made
- `attribution.jsonl` — machine join (call → role → claims)

| role | layer | model | calls | claims |
|---|---|---|---:|---:|
| `agent` | read | `gpt-4o-mini` | 1986 | 1986 |

## Agent comparison controls

- status: **incomparable** · contract `53b492b84c1952f127aaa381db0f902dd9cfc471d5008043d4e543aa7ffa934d`
- backbone: `codex` / `gpt-4o-mini` snapshot=`not_applicable`
- prompt sha256: `c7f5e2d1fac61fc0cf9a014a1b675469df60b2657c36124b15a28ea73b9f69d5`
- context workspace sha256: `9a954932e7a1752d3aba1bbdf8e6aa33e55817dcdc1f874c8e7d9f2ac0d30ef6`
- retrieval: `{}`
- memory write: `{'persist_notes': False, 'reader_payload_shared': True, 'harness_persistence_footer': False}`
- judge: `{}`
- tool budget: `{'allowed_tools': [], 'max_tool_calls': None, 'max_retrieved_tokens': None, 'wall_clock_s': 600}`

## Frozen config

- `config.source.yaml` — YAML file used for this run
- `config.resolved.yaml` — YAML plus CLI overrides that actually ran
- git `None` · data sha256 `79fa87e90f04…`

## Cost rollup

- reader tokens: 22560782 (usd=3.392887; reasoning=0)
- writer tokens: 0 (usd=0.0; reasoning=0)
- **total tokens:** 22560782 · **usd:** 3.392887 · reasoning=0
- pricing pins as of 2026-09-16; unknown models are token-only

## Writer quality

(no writer calls this run)

## How to audit one claim

1. `ATTRIBUTION.md` / `attribution.jsonl` — LLM call → role → claims made.
2. Pick a `question_id` in `predictions.jsonl` (or `SUMMARY` lineage sample below).
3. `memory/lineage.jsonl` — which memory items were injected and which writer proposed them.
4. `memory/writer/sessions/` — the session text the writer saw.
5. `memory/graph/ingest.jsonl` — MERGE / invalidate (graph conditions).
6. `memory/retrieve_ranks.jsonl` — candidates that lost to the injected winners.

- writer calls this run: 0
- lineage rows (injected items): 55014

## Lineage sample (first injected items)

- `conv-26-q-0` ← `session_1_summary` (session_summary) proposed_by=— 
- `conv-26-q-0` ← `session_2_summary` (session_summary) proposed_by=— 
- `conv-26-q-0` ← `session_3_summary` (session_summary) proposed_by=— 
- `conv-26-q-0` ← `session_4_summary` (session_summary) proposed_by=— 
- `conv-26-q-0` ← `session_5_summary` (session_summary) proposed_by=— 
- `conv-26-q-0` ← `session_6_summary` (session_summary) proposed_by=— 
- `conv-26-q-0` ← `session_7_summary` (session_summary) proposed_by=— 
- `conv-26-q-0` ← `session_8_summary` (session_summary) proposed_by=— 
