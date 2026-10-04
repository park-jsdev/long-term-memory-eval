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
| `memory_type` | string | Condition id, e.g. `raw_chunks`, `session_summaries`, `graph` |
| `text` | string | **Full payload** inserted into the answer prompt as `{memory}` |
| `source_ids` | list[string] | Provenance ids (turn `dia_id`s or `session_k_summary`) |
| `schema_version` | string | Always `memory_io.v1` for this layout family |
| `writer_model` | string or null | Write-path model id when a writer produced this memory |
| `writer_provider` | string or null | `openai`, `anthropic`, `deepseek`, or `mock` |

JSON shape (also in `memory_io.schema.json`):

```json
{
  "schema_version": "memory_io.v1",
  "memory_type": "session_summaries",
  "text": "... full string ...",
  "source_ids": ["session_1_summary", "session_2_summary"]
}
```

---

## 2. How `text` is formatted at runtime

### Condition `raw_chunks` (`RawConversationMemoryBuilder`)

Config: `configs/writers/raw_chunks.yaml` · optional `pipeline.memory_max_chars` (tail keep if over budget).

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

### Condition `full_context` (`FullContextMemoryBuilder`)

Config: `configs/writers/full_context.yaml`. Mem0 paper transcript grammar (`{timestamp} | {speaker}: {text}`). No retrieval.

### Condition `rag` (`RagMemoryBuilder`)

Config: `configs/writers/rag.yaml`. Loads `experiments/<index_run_id>/rag_index/`. Question-dependent cosine top-k token chunks joined with `\n<->\n`. Per-question texts: `memory/by_question/<qid>.txt`.

### Condition `openai_memory` (`OpenAIMemoryBuilder`)

Config: `configs/writers/openai_memory.yaml`. Concatenates every extracted `{timestamp} | {speaker}: {fact}`. No top-k.

### Condition `session_summaries` (`SessionSummaryMemoryBuilder`)

Config: `configs/writers/session_summaries.yaml` · uses LoCoMo release field `session_summary`.

```text
[Session 1]
{session_1_summary text}

[Session 2]
{session_2_summary text}

...
```

Sessions omitted if empty. Order = chronological session number.

### Condition `session_summaries`

Config: `configs/writers/session_summaries.yaml`. With no `writer` block, `SessionSummaryMemoryBuilder` copies the dataset summaries. With `writer.model` (or `--writer-model`), `WriterSessionMemoryBuilder` writes one summary per session (`prompts/writers/session_summary_v1.txt`). `writer_model` is stored on the Memory object and in `memory/schema.json`, not inside `{memory}` text.

### Condition `graph`

Config: `configs/writers/graph.yaml` (includes `configs/writers/openai_mini.yaml`).

One writer model extracts Mem0-shaped triples; `ModelOrchestrator` writes **locked** `Mem0GraphMemory` (`ingest_triples`). `{memory}` is:

```text
Conversation between {speaker_a} and {speaker_b}.

Graph relations:
{source} -- {relationship} -- {target}
...
```

Swap the model with `writer.model` or `--writer-model`. Gold answers never enter writer prompts.

### Injection into the fixed prompt

```text
# prompts/readers/qa_v1.txt
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
| `memory/by_sample/<sample_id>.txt` | **Full** `Memory.text` for that conversation (what the model saw as memory; same for all Qs under that sample for current builders) |
| `memory/prompt_fill_example.txt` | One concrete filled prompt (memory + first question), truncated if huge |
| `memory/writer/` | Write-path LLM traces when a writer ran (`index.jsonl`, `calls.jsonl`, `by_writer/<id>/`, `sessions/`, `quality.json`) |
| `memory/graph/` | Mem0g snapshot (`by_sample/<id>.json`) plus `ingest.jsonl` (MERGE / invalidate / skip_dup) |
| `memory/lineage.jsonl` | Question → injected memory item → writer (`proposed_by`) |
| `memory/retrieve_ranks.jsonl` | Full ranked retrieve candidates, not just winners |

Run-root `ATTRIBUTION.md` / `attribution.jsonl` join each LLM call to its sandwich role and the claims it made (not stored under `memory/`).

Answer-LLM traces live under `experiments/<run_id>/reader/` (`traces.jsonl` + a copy of `predictions.jsonl`). Judge traces live under `autorater/traces.jsonl`. Run-root `predictions.jsonl` is a compatibility copy of `reader/predictions.jsonl`.

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
  "memory_type": "session_summaries",
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
