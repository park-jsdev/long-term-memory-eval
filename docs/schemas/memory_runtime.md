# Runtime memory schema & logs

**Purpose:** document *exactly* what string the answer LLM receives as `{memory}`, how it is built, and where each run records it for audit.

**Code objects:** `src/locomo_eval/schemas.py` → `Memory`  
**Builders:** `src/locomo_eval/memory.py`  
**Logging helpers:** `src/locomo_eval/memory_log.py`  
**Per-run dump:** `experiments/<run_id>/memory/`

Schema id: **`memory_io.v1`**

---

## 1. In-memory object (`Memory`)

Passed from builder → runner → prompt fill. Never includes the gold answer.

| Field | Type | Meaning |
|-------|------|---------|
| `memory_type` | string | Condition id, e.g. `c0_raw`, `c1_session_summary`, `c1_teacher` |
| `text` | string | **Full payload** inserted into `prompts/qa_v1.txt` as `{memory}` |
| `source_ids` | list[string] | Provenance ids (turn `dia_id`s or `session_k_summary`) |
| `schema_version` | string | Always `memory_io.v1` for this layout family |
| `teacher_model` | string or null | Write-path model id when using `c1_teacher` |
| `teacher_provider` | string or null | `openai` or `mock` |

JSON shape (also in `memory_io.schema.json`):

```json
{
  "schema_version": "memory_io.v1",
  "memory_type": "c1_session_summary",
  "text": "... full string ...",
  "source_ids": ["session_1_summary", "session_2_summary"]
}
```

---

## 2. How `text` is formatted at runtime

### Condition `c0_raw` (`RawConversationMemoryBuilder`)

Config: `configs/c0_raw.yaml` · optional `pipeline.memory_max_chars` (tail keep if over budget).

```text
Conversation between {speaker_a} and {speaker_b}.

DATE: {session_date_time}
SESSION {k}:
{speaker}: {turn text} [image: {blip_caption}? ] ({dia_id}?)
...

DATE: ...
SESSION {k+1}:
...
```

If truncated:

```text
[... earlier turns truncated to max_chars ...]
{last memory_max_chars characters of the full string}
```

### Condition `c1_session_summary` (`SessionSummaryMemoryBuilder`)

Config: `configs/c1_session_summary.yaml` · uses LoCoMo release field `session_summary`.

```text
[Session 1]
{session_1_summary text}

[Session 2]
{session_2_summary text}

...
```

Sessions omitted if empty. Order = chronological session number.

### Condition `c1_teacher` (`TeacherSessionMemoryBuilder`)

Config: `configs/c1_teacher.yaml` · `teacher.model` / `--teacher-model`.

Same `[Session k]` concatenation as C1, but each block is a **teacher** summary of that session's turns (prompt: `prompts/teacher_session_v1.txt`). `teacher_model` is stored on the Memory object and in `memory/schema.json` — not inside `{memory}` text (so the answer LLM does not see the teacher id).

### Injection into the fixed prompt

```text
# prompts/qa_v1.txt
Memory:
{memory}      ← Memory.text  (entire string above)
Question:
{question}    ← QA item only (no gold answer)
```

---

## 3. Where runtime memory is logged

Every Phase‑1 run writes under `experiments/<run_id>/memory/`:

| Path | Contents |
|------|----------|
| `memory/README.md` | Short pointer to this doc + condition used this run |
| `memory/schema.json` | Machine description of this run’s memory_type layout (`memory_io.v1`) |
| `memory/index.jsonl` | One line per **unique** `sample_id` used in the run: ids, char counts, sha256 of `text`, head/tail previews |
| `memory/by_sample/<sample_id>.txt` | **Full** `Memory.text` for that conversation (what the model saw as memory; same for all Qs under that sample for C0/C1) |
| `memory/prompt_fill_example.txt` | One concrete filled prompt (memory + first question), truncated if huge |

Also retained on every prediction row (full audit, larger files):

| Path | Fields |
|------|--------|
| `predictions.jsonl` | `memory_type`, `memory_text` (full), … |
| `predictions.csv` | `memory_chars`, `memory_preview` (clip) |

**Scorer / gold:** gold stays in `reference_answer` on predictions; it is **never** inside `Memory.text` or the user message built for the API.

---

## 4. index.jsonl row schema

```json
{
  "schema_version": "memory_io.v1",
  "sample_id": "conv-26",
  "memory_type": "c1_session_summary",
  "n_source_ids": 19,
  "n_chars": 20863,
  "text_sha256": "…",
  "text_head": "first 400 chars…",
  "text_tail": "last 200 chars…",
  "full_text_path": "memory/by_sample/conv-26.txt"
}
```

Use `text_sha256` to detect whether two conditions or two runs produced the same memory body.

---

## 5. Evolution

When you change how builders format `text` (new headers, claim JSON, etc.):

1. Bump `schema_version` (e.g. `memory_io.v2`) in `memory_log.py` / this doc.  
2. Document the new layout under section 2.  
3. Keep old run folders self-describing via their local `memory/schema.json`.
