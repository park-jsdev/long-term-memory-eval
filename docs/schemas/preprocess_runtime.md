# Runtime preprocess schema & logs

**Purpose:** document the session segmentation recorded during preprocess so later teacher / eval steps attribute the same session and turn ids.

**Code objects:** `src/locomo_eval/schemas.py` → `ProcessedConversation`, `SessionBlock`, `ProcessedTurn`  
**Pipeline:** `src/locomo_eval/preprocess/data_ingestor.py`, `preprocessing_pipeline.py`  
**Logging:** `src/locomo_eval/preprocess/conversation_log.py`  
**Dump (when wired):** `experiments/<run_id>/preprocess/` — **not** written by `run.py` in this slice.

Schema id: **`preprocess_io.v1`**

HLD: raw LoCoMo conversation → DataIngestor → PreprocessingPipeline (session segmentation, turn ids/metadata, speaker and time normalization) → Writer Orchestrator.

---

## 1. In-memory objects

### `ProcessedConversation`

| Field | Type | Meaning |
|-------|------|---------|
| `sample_id` | string | LoCoMo sample id |
| `speaker_a` / `speaker_b` | string | Conversation-level names from the release |
| `session_blocks` | list | Ordered non-empty sessions |
| `question_ids` | list[string] | Pass-through handles for later join. **Not** gold answers |
| `schema_version` | string | Always `preprocess_io.v1` |

Gold answers stay on `Conversation.questions` / `Prediction.reference_answer`. They must not appear on session blocks.

### `SessionBlock`

One LoCoMo `session_N` after preprocess. This is the unit `ModelOrchestrator` iterates.

| Field | Type | Meaning |
|-------|------|---------|
| `sample_id` | string | Parent sample |
| `session_id` | int | LoCoMo session number (`session_3` → `3`) |
| `session_index` | int | 0-based order among **non-empty** sessions |
| `source_key` | string | `session_{id}` as in the JSON |
| `date_time_raw` | string | Unmodified `session_N_date_time` |
| `date_time_normalized` | string or null | ISO-8601 when parseable, else `null` |
| `speaker_a` / `speaker_b` | string | Copied from the conversation |
| `turns` | list | `ProcessedTurn` rows |
| `schema_version` | string | `preprocess_io.v1` |

Segmentation follows the dataset’s `session_N` keys. This pipeline does **not** cut new sessions by time gaps or token budgets.

### `ProcessedTurn`

| Field | Type | Meaning |
|-------|------|---------|
| `turn_id` | string | Ours: `{sample_id}:s{session_id}:t{iii}` (3-digit index) |
| `source_dia_id` | string | LoCoMo `dia_id` (e.g. `D1:1`); may be empty |
| `turn_index` | int | 0-based inside the session |
| `speaker_raw` | string | Name as in the turn |
| `speaker_role` | `a` / `b` / `other` | Map onto `speaker_a` / `speaker_b` (strip + casefold) |
| `text` | string | Utterance |
| `blip_caption` | string or null | Optional image caption |

---

## 2. How dates are normalized

LoCoMo release strings look like `1:56 pm on 8 May, 2023`. Tests also use `1 Jan 2023`.

- Always keep `date_time_raw`.
- If `strptime` matches a known format → ISO-8601 (`2023-05-08T13:56:00` or `2023-01-01`).
- If not → `date_time_normalized` is `null` (do not invent a date).

---

## 3. Where preprocess state is logged

CLI `python -m src.locomo_eval.preprocess.run_index` writes `experiments/<run_id>/preprocess/` (no LLM). Gold answers are not in this dump.

| Path | Contents |
|------|----------|
| `schema.json` | Machine description of `preprocess_io.v1` |
| `run_meta.json` | data SHA, git hash, `llm_calls=0` |
| `index.jsonl` | One line per sample: counts, paths |
| `by_sample/<sample_id>/sessions.jsonl` | One `SessionBlock` JSON object per line |
| `by_sample/<sample_id>/documents.jsonl` | One `SessionDocument` JSON object per line (turns + dataset summary + obs/events; no QA gold) |

`raw_chunks` / `session_summaries` can load this dump (`--preprocess-index-run-id`). Default retrieve is concatenate-all; `preprocess.top_k` / `--retrieve-top-k` is a later naive-rank hook. `write_conversation_run_log` still writes sessions-only for older tests.

Complete sample (required before a later retrieve/format load): both `sessions.jsonl` and `documents.jsonl`.

---

## 4. index.jsonl row schema

```json
{
  "schema_version": "preprocess_io.v1",
  "sample_id": "conv-26",
  "n_session_blocks": 19,
  "n_documents": 19,
  "n_turns": 412,
  "question_ids": ["conv-26-q-0"],
  "sessions_path": "by_sample/conv-26/sessions.jsonl",
  "documents_path": "by_sample/conv-26/documents.jsonl"
}
```

---

## 5. Writer orchestrator (HLD ii, thin seam)

`src/locomo_eval/model_orchestrator.py` iterates **one `SessionBlock` at a time**. With no writer it returns a passthrough record (`status="passthrough"`). A writer set on `session_summaries` or `graph` generates memory for that block.
