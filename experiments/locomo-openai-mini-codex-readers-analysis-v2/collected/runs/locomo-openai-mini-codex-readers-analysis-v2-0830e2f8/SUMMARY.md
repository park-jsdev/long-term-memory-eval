# Run audit: `locomo-openai-mini-codex-readers-analysis-v2-0830e2f8`

This folder is an **audit of claims** (what entered `{memory}` and who
proposed it), not only an audit of API calls.

## Sandwich

- **memory:** `full_context`
- **reader:** `openai/gpt-4o-mini` (prompt `qa_mem0_v1`)
- **agent:** (none; one-shot reader)
- **writer:** (none)
- **n questions:** 1986
- **LoCoMo F1:** 0.422
- **token F1 / EM:** 0.4128 / 0.1636

## LLM roles

- `ATTRIBUTION.md` — each LLM call, the role it played, and the claims it made
- `attribution.jsonl` — machine join (call → role → claims)

| role | layer | model | calls | claims |
|---|---|---|---:|---:|
| `reader` | read | `gpt-4o-mini` | 1986 | 1986 |

## Frozen config

- `config.source.yaml` — YAML file used for this run
- `config.resolved.yaml` — YAML plus CLI overrides that actually ran
- git `None` · data sha256 `79fa87e90f04…`

## Cost rollup

- reader tokens: 55318860 (usd=8.30339; reasoning=0)
- writer tokens: 0 (usd=0.0; reasoning=0)
- **total tokens:** 55318860 · **usd:** 8.30339 · reasoning=0
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
