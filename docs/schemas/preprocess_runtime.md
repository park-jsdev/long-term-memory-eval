# Runtime preprocess schema & logs

**Purpose:** document the session segmentation recorded during preprocess so later teacher / eval steps attribute the same session and turn ids.

**Code objects:** `src/locomo_eval/schemas.py` → `ProcessedConversation`, `SessionBlock`, `ProcessedTurn`  
**Pipeline:** `src/locomo_eval/preprocess/data_ingestor.py`, `preprocessing_pipeline.py`  
**Logging:** `src/locomo_eval/preprocess/conversation_log.py`  
**Dump (when wired):** `experiments/<run_id>/preprocess/` — **not** written by `run.py` in this slice.

Schema id: **`preprocess_io.v1`**

HLD: raw LoCoMo conversation → DataIngestor → PreprocessingPipeline (session segmentation, turn ids/metadata, speaker and time normalization) → Teacher Orchestrator.

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

One LoCoMo `session_N` after preprocess. This is the unit `TeacherOrchestrator` iterates.

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

Helper `write_conversation_run_log` writes a folder that can later sit at `experiments/<run_id>/preprocess/`:

| Path | Contents |
|------|----------|
| `schema.json` | Machine description of `preprocess_io.v1` |
| `index.jsonl` | One line per sample: counts, first/last session ids |
| `by_sample/<sample_id>/sessions.jsonl` | One `SessionBlock` JSON object per line |

`run.py` does **not** call this yet. Tests write to a temp dir.

---

## 4. index.jsonl row schema

```json
{
  "schema_version": "preprocess_io.v1",
  "sample_id": "conv-26",
  "n_session_blocks": 19,
  "n_turns": 412,
  "question_ids": ["conv-26-q-0"],
  "sessions_path": "by_sample/conv-26/sessions.jsonl"
}
```

---

## 5. Teacher orchestrator (HLD ii, thin seam)

`src/locomo_eval/teacher_orchestrator.py` iterates **one `SessionBlock` at a time** and returns a passthrough record (`status="passthrough"`). No LLM, no fusion, no `LlmResponseCache`. Promote to a `write/` package only when teachers actually generate memory.
