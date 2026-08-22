# Data

## LoCoMo (eval / optional train)

- Source: https://github.com/snap-research/locomo
- Paper: Maharana et al., [*Evaluating Very Long-Term Conversational Memory of LLM Agents*](https://arxiv.org/html/2402.17753v1) (ACL 2024)
- Pinned commit: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`
- File: `data/raw/locomo10.json` (fetched; not committed)
- License: **CC BY-NC 4.0** (research / non-commercial only)

Fetch:

```bash
python scripts/fetch_locomo.py
```

This writes `data/raw/locomo10.json` and prints a SHA256 for your notes.

Do not commit the raw JSON unless you intentionally accept redistribution under CC BY-NC attribution rules.

---

## What is in the JSON (document types)

Each of the **10 samples** (`sample_id` like `conv-26`) is one multi-session conversation plus annotations. It is a **benchmark package**, not a single document type.

| Field | Grain | What it is | Gold for **our QA sandwich**? |
|-------|--------|------------|-------------------------------|
| `conversation` / `session_N` | session → turns (`dia_id` `D{session}:{turn}`) | Source dialog (LLM agents + human edits) | **Input corpus.** `raw_chunks` uses this. |
| `qa` | question | Eval **task 1** (memory recall): question, answer, `evidence` turn ids, `category` | **Yes — scorer only.** Never in `Memory.text` / `SessionBlock`. |
| `event_summary` / `events_session_k` | events per speaker + date | Eval **task 2** (event-graph summarization). Graphs also **authored** the dialogs. | Not QA gold. Candidate gold later for a **memory-design / event** task. |
| `session_summary` | one blob per session | Agent short-term memory after each session; paper RAG corpus | **Supplement** (our `session_summaries` condition). |
| `observation` | assertions `[text, dia_id]` per speaker | Agent long-term memory; paper’s best RAG index | **Supplement** (unused in `raw_chunks` / `session_summaries`). |
| `blip_caption` / `img_url` | turn | Multimodal generation task; QA uses captions | Optional turn text, not QA gold. |

Paper eval is three tasks (QA, event summarization, multimodal generation). This repo scores **QA** only (SPEC + LoCoMo F1).

**Preprocess cut:** `SessionBlock` / `SessionDocument` = one LoCoMo `session_N` **inside** a `sample_id`. We do **not** cut by sample. Empty sessions are dropped (none in locomo10). On this file: **10 samples, 272 sessions, 5882 turns, 1986 QA**. Median session ≈ 20 turns / ~2.5k chars. Packed / noisy `evidence` strings in locomo10 (split in `session_documents.evidence_tokens`): `D8:6; D9:17`, space-separated ids, `D:11:26`, `D30:05`. Two ids have **no matching turn** (`D10:19` on conv-42-q-58, `D4:36` on conv-47-q-38) — annotation noise, not a preprocess bug.

---

## Category ids: official eval, not paper prose order

The paper §4.1 **lists** types as (1) single-hop, (2) multi-hop, (3) temporal, (4) open-domain, (5) adversarial.

The **JSON integer** `qa[].category` follows [official `task_eval/evaluation.py`](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py) (our pin), copied in `src/metrics/locomo_qa.py`:

| JSON `category` | Name we use | How official eval scores it |
|-----------------|-------------|-----------------------------|
| 1 | `multi_hop` | Multi-answer F1 (split on commas) |
| 2 | `temporal` | Token F1 |
| 3 | `open_domain` | Token F1; gold taken before first `;` |
| 4 | `single_hop` | Token F1 |
| 5 | `adversarial` | 1 iff prediction contains “not mentioned” or “no information available” |

`CATEGORY_ORDER = [4, 1, 2, 3, 5]` is **table column order** (single-hop first), matching LoCoMo reporting — not the JSON ids.

Counts in locomo10: 841 single-hop, 282 multi-hop, 321 temporal, 96 open-domain, 446 adversarial.

---

## Inspectable tables + plots (`data/processed/`)

Gitignored (derived). After fetch:

```bash
python scripts/export_session_documents.py --data data/raw/locomo10.json --out data/processed
```

Writes:

| File | One row is… |
|------|-------------|
| `session_documents.csv` | One **session document** (the preprocess unit): speakers, dates, turn/image/obs/event counts, summary + turn previews. **No gold answers.** |
| `turns.csv` | One dialog turn (`dia_id`, text, image flags) |
| `qa_joined.csv` | QA **gold** joined to sessions via `evidence` dia_ids (oracle). For manual checks only. |
| `observations.csv` | Session-level assertions |
| `events.csv` | Event-graph lines (possible later memory-design gold) |
| `images.csv` | Turns with `blip_caption` / `img_url` |
| `dataset_stats.json` | Counts + naive lexical retrieval recall |
| `plots/*.png` | Histograms (sessions/sample, turns/session, chars, QA/sample, evidence-session span, category counts) and naive retrieval bars |

**Naive retrieval** (sanity, not a paper RAG claim): rank sessions by **question** token overlap with (a) raw turn text (“agent on blocks”) or (b) `session_summary` (“naive memory”). Gold answers are not in the query. Recall = fraction of evidence session ids in top-k (k=5 single-hop, k=10 multi-hop).

Also: `python scripts/prepare_data.py --split all --no-jsonl` still writes the older flattened `qa_all.csv` (one row per question, full-conversation preview).
