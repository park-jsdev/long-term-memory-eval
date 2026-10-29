# Run audit: `locomo-openai-mini-codex-readers-analysis-v2-47639359`

This folder is an **audit of claims** (what entered `{memory}` and who
proposed it), not only an audit of API calls.

## Sandwich

- **memory:** `full_context`
- **reader:** `codex/gpt-4o-mini` (prompt `qa_mem0_v1`)
- **agent:** `codex` persist=True tools=native
- **writer:** (none)
- **n questions:** 1986
- **LoCoMo F1:** 0.264
- **token F1 / EM:** 0.2564 / 0.0262

## LLM roles

- `ATTRIBUTION.md` — each LLM call, the role it played, and the claims it made
- `attribution.jsonl` — machine join (call → role → claims)

| role | layer | model | calls | claims |
|---|---|---|---:|---:|
| `agent` | read | `gpt-4o-mini` | 1986 | 1986 |

## Agent comparison controls

- status: **incomparable** · contract `9766b4799913a88dbd30b226305ef2647e0bb57e91f6b5836ed4d007b2eea7bf`
- backbone: `codex` / `gpt-4o-mini` snapshot=`not_applicable`
- prompt sha256: `c7f5e2d1fac61fc0cf9a014a1b675469df60b2657c36124b15a28ea73b9f69d5`
- context workspace sha256: `f43f50d6322d1809397fdb8537e67d6ae58a5ece980b46b6b99a10cfd55eeb8d`
- retrieval: `{}`
- memory write: `{'persist_notes': True, 'reader_payload_shared': True, 'harness_persistence_footer': True}`
- judge: `{}`
- tool budget: `{'allowed_tools': [], 'max_tool_calls': None, 'max_retrieved_tokens': None, 'wall_clock_s': 600}`

## Frozen config

- `config.source.yaml` — YAML file used for this run
- `config.resolved.yaml` — YAML plus CLI overrides that actually ran
- git `None` · data sha256 `79fa87e90f04…`

## Cost rollup

- reader tokens: 71822139 (usd=10.794344; reasoning=0)
- writer tokens: 0 (usd=0.0; reasoning=0)
- **total tokens:** 71822139 · **usd:** 10.794344 · reasoning=0
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
- lineage rows (injected items): 1986

## Lineage sample (first injected items)

- `conv-26-q-0` ← `full_context:conv-26` (full_context_payload) proposed_by=— 1:56 pm on 8 May, 2023 | Caroline: Hey Mel! Good to see you! How have you been? 1:56 pm on 8 May, 2023 | Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? A…
- `conv-26-q-1` ← `full_context:conv-26` (full_context_payload) proposed_by=— 1:56 pm on 8 May, 2023 | Caroline: Hey Mel! Good to see you! How have you been? 1:56 pm on 8 May, 2023 | Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? A…
- `conv-26-q-2` ← `full_context:conv-26` (full_context_payload) proposed_by=— 1:56 pm on 8 May, 2023 | Caroline: Hey Mel! Good to see you! How have you been? 1:56 pm on 8 May, 2023 | Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? A…
- `conv-26-q-3` ← `full_context:conv-26` (full_context_payload) proposed_by=— 1:56 pm on 8 May, 2023 | Caroline: Hey Mel! Good to see you! How have you been? 1:56 pm on 8 May, 2023 | Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? A…
- `conv-26-q-4` ← `full_context:conv-26` (full_context_payload) proposed_by=— 1:56 pm on 8 May, 2023 | Caroline: Hey Mel! Good to see you! How have you been? 1:56 pm on 8 May, 2023 | Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? A…
- `conv-26-q-5` ← `full_context:conv-26` (full_context_payload) proposed_by=— 1:56 pm on 8 May, 2023 | Caroline: Hey Mel! Good to see you! How have you been? 1:56 pm on 8 May, 2023 | Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? A…
- `conv-26-q-6` ← `full_context:conv-26` (full_context_payload) proposed_by=— 1:56 pm on 8 May, 2023 | Caroline: Hey Mel! Good to see you! How have you been? 1:56 pm on 8 May, 2023 | Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? A…
- `conv-26-q-7` ← `full_context:conv-26` (full_context_payload) proposed_by=— 1:56 pm on 8 May, 2023 | Caroline: Hey Mel! Good to see you! How have you been? 1:56 pm on 8 May, 2023 | Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? A…
