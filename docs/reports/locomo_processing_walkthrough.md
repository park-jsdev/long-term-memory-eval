# LoCoMo through this codebase

How one LoCoMo conversation becomes a scored prediction, and how a stuffed
Chat Completions call differs from a Codex agent loop. Use this to walk a
finished pack in `experiments/<run_id>/` from the JSON on disk to the numbers
in `metrics.json` and `autorater/`.

Code map: `src/locomo_eval/schemas.py` (the types), `dataset.py` (load),
`memory.py` (the middle), `readers.py` (one-shot answer), `agents/` (Codex
loop), `metrics.py` + `src/metrics/locomo_qa.py` (string scores),
`autorater.py` (Mem0 judge). Orchestrator: `run.py`
(`run_locomo_pipeline_with_memory_config`). Harness: `src/experiment_runner/`
launches **one cell** of that function; it does not reimplement scoring.

---

## 1. What LoCoMo is, and what this repo actually loads

LoCoMo (Maharana et al., arXiv:2402.17753) is a long-term conversational
memory benchmark. Each sample is a multi-session dialogue between two
speakers, plus questions whose answers are meant to be recoverable from that
dialogue. The paper’s QA tables use **token F1** (and a special rule for
adversarial questions). They do **not** use the Mem0 LLM judge.

This repo does not re-run the paper’s 50-conversation GPT-3.5 RAG grid.
It loads the released **locomo10** file:

`data/raw/locomo10.json` (fetched, not committed; pin
`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`). Ten conversations. Gold answers
stay in that file for scorers only. They are never copied into a reader
prompt, a teacher prompt, or an agent workspace.

One sample in the JSON:

| Field | What it is | Who may see it |
|---|---|---|
| `sample_id` | Conversation id (`conv-26`, …) | everyone |
| `conversation.speaker_a` / `speaker_b` | Names | memory text / workspace |
| `conversation.session_N` | List of turns: `dia_id`, `speaker`, `text`, optional `blip_caption` | memory / workspace |
| `conversation.session_N_date_time` | Session date string | memory / workspace |
| `session_summary.session_N_summary` | Dataset-written session summary | only the `session_summaries` builder |
| `observation.session_N_observation` | Dataset assertions | loaded, **not** used by current builders |
| `qa[]` | `question`, `answer`, `category`, `evidence` (list of `dia_id`) | question goes to the model; `answer` and `evidence` do not |

`dataset.parse_sample` turns that into `Conversation` + `Question`. Question
ids are synthetic: `{sample_id}-q-{index}` in **qa-array order** (0-based).
Category ids are the official JSON ids, not the paper’s prose numbering:

| id | name | LoCoMo F1 rule |
|---|---|---|
| 4 | single_hop | stemmed token F1 vs the gold string |
| 1 | multi_hop | multi-answer F1 (split prediction and gold on commas) |
| 2 | temporal | stemmed token F1 |
| 3 | open_domain | gold is split on `;`, **first clause only**, then token F1 |
| 5 | adversarial | 1 if the prediction contains “not mentioned” or “no information available”, else 0 |

Reporting order in tables is 4, 1, 2, 3, 5 (`CATEGORY_ORDER` in
`src/metrics/locomo_qa.py`).

---

## 2. Three ways a question is answered

Every QA cell is the same outer loop. The middle (what the model is allowed
to see) is the thing that changes.

```text
locomo10.json
    │  dataset.load_conversations          no LLM
    ▼
Conversation + Question[]                  gold stays on Question.answer
    │
    ├─ A. Stuffed reader (full_context, raw_chunks, session_summaries, …)
    │     Memory.text built in Python
    │     one Chat Completions call per question: prompt(memory, question)
    │
    ├─ B. Sandwich writer (teacher_*, mem0, mem0g, rag, openai_memory)
    │     Write LLMs run first (per session, or a shared index job)
    │     then the same frozen reader as A, one call per question
    │
    └─ C. Codex agent (workspace_files)
          Python writes session files (no LLM)
          one `codex exec` per question (the CLI runs the model+tool loop)
    │
    ▼
Prediction row                             question, gold, prediction, memory snapshot
    ├─ string scores now                   EM, token F1, LoCoMo F1     no LLM
    └─ autorater later, separate job       Mem0 J                      one judge call / question
```

**A is the “raw model call.”** The answer model sees a stuffed string and
returns one completion. There is no tool loop and no second model on the
read path.

**B keeps that same read path** and changes only how `Memory.text` was
produced. The reader YAML stays `gpt-4o-mini` + `prompts/readers/qa_mem0_v1.txt`
on sandwich cells. Do not read a writer-model change as a reader-model change.

**C does not stuff the transcript.** `Memory.text` is a manifest pointer.
The model (inside Codex) must read files. Tool calls are real CLI events,
not a Python retrieval function.

---

## 3. Order and parallelism

Inside one run (`run.py`, the `for i, (conv, q) in enumerate(pairs)` loop):

- Conversations are in **file order**.
- Questions inside a conversation are in **qa-array order**.
- The loop is **one question at a time**. The next question starts only
  after this answer (and, for Codex, this `codex exec`) returns.
- There is no thread pool over questions. `execution.max_concurrency` in
  experiment YAML is not read by the QA loop.

A smoke cap (`--max-questions`) defaults to **round-robin**: question 0 of
each conversation, then question 1 of each, and so on. Full runs do not use
that cap, so order is file order.

Across cells, parallelism is **one Cloud Run task per matrix cell**
(`experiment_runner execute-qa --run-index N`, or `gcloud run jobs execute
… --tasks=N`). Cell 0 does not share a process with cell 1. Shared Mem0/RAG
indexes are built once and reused; they are not rebuilt inside every reader
cell.

Persist-on Codex is order-sensitive: notes written on question *i* of a
conversation are still on disk for question *i+1* of that **same**
conversation. Questions from other conversations do not see that workspace.

---

## 4. Path A — stuffed reader (raw model call)

Example: `full_context`, `raw_chunks`, dataset `session_summaries`.

1. Load all ten conversations. No LLM.
2. If the builder is question-independent (the default; no `retrieve_top_k`),
   `builder.build(conv, first_question)` runs **once per conversation** and
   the same `Memory.text` is reused for every question in that conversation.
   `is_question_independent` lists those builders. Top-k retrieve makes the
   build per question.
3. For each question, `reader.answer(memory.text, question, prompt)`:
   - Render `prompts/readers/qa_mem0_v1.txt` with `{memory}` and `{question}`.
   - One Chat Completions request (`OpenAIChatCaller.complete`). Temperature 0.
   - Gold is not in the messages.
4. Append one `predictions.jsonl` row. Flush after every question so a crash
   leaves a partial file. The next invocation of the same `--run-id` **clears
   and regenerates**; there is no per-question resume.
5. After the loop, `summarize_predictions` writes `metrics.json` from the
   strings only.

What each builder puts in `{memory}` (still no read-path tools):

| Builder | LLM on the write? | Text the reader sees |
|---|---|---|
| `raw_chunks` | no | turns in session order, with `dia_id` |
| `session_summaries` | no | dataset `session_summary` blobs, concatenated |
| `full_context` | no | the whole dialogue (Mem0-style stuffed context) |
| `rag` | embedder at **index** time, not per question | top-k chunks from a prebuilt index |
| `openai_memory` | extractor at **index** time | all extracted facts, retrieved in full |
| `mem0` / `mem0g` | extract/update at **index** time | retrieved facts (mem0g also has the locked graph) |

Index jobs (`mem0.run_index`, `rag.run_index`, `openai_memory.run_index`) are
separate commands. QA only reads the dump.

LLM calls on path A, per question: **exactly one** (the reader), unless you
are still building an index. The judge is not in this process.

---

## 5. Path B — sandwich writer, then the same reader

Example: `session_summaries`, `graph`.

The reader is frozen. The extra LLMs are **teachers**, and they run when
`builder.build` runs (once per conversation, before that conversation’s
questions are answered).

`session_summaries`: for each session, one teacher Chat Completions
call (`prompts/writers/session_summary_v1.txt`) produces a summary. Those
summaries are concatenated into `Memory.text`. Then path A’s reader loop
runs. This is **not** the dataset `session_summary` field and **not** the
LoCoMo 2024 “Summary RAG top-5” row (that paper retrieved summaries with a
different reader and scored F1, not Mem0 J).

`graph`: the teacher emits entity/relation JSON
(`prompts/writers/graph_v1.txt`). Software (`ModelOrchestrator`)
writes the locked `Mem0GraphMemory` schema. The reader sees the
formatted graph, not the raw teacher JSON. This is **not** the Mem0 paper’s
`mem0g` extract/update loop. OSS `mem0` / `mem0g` are the index builders in
`src/locomo_eval/mem0/`, a different code path.

Writer traces land in `experiments/<run_id>/memory/teachers/` (calls,
session text, fusion). Graph ingest is `memory/graph/` when a graph was built.
`ATTRIBUTION.md` joins each teacher call to the claims it produced.

LLM calls on path B: **one writer-model call per session during build**,
then **one reader call per question**. Still no tools on the read path.
The YAML key is `teacher`; the runtime accepts one model.

---

## 6. Path C — Codex agent loop

Config: `pipeline.answer_mode: agent` and `pipeline.memory: workspace_files`
(`configs/agents/`, `configs/writers/workspace_files.yaml`).

### 6.1 What Python does before any model call

`workspace_files` writes, per conversation, with no LLM:

```text
experiments/<run_id>/agent/workspaces/<sample_id>/
  INDEX.md
  sessions/session_N.md      each turn keeps (dia_id) speaker: text
  memory/notes.md            only if persist is on
```

Gold `answer` is not in these files. `evidence` dia_ids are kept on the
`Question` object and applied **after** the agent returns, when the
trajectory is scored.

`Memory.text` for this builder is a short manifest (where the files are),
not the transcript. A stuffed-context comparison is a different cell
(`full_context`), not this workspace.

### 6.2 One process per question, and a loop inside it

For each question, Python starts **one** subprocess:

`codex exec --json` with `--ignore-user-config`, `--ignore-rules`,
`-c web_search="disabled"`, `-c sandbox_permissions=["disk-full-read-access"]`,
`--sandbox read-only` (or `workspace-write` if persist is on), `--ephemeral`
when persist is off, `--model <backbone>`, and `-o agent_answer.json`.

Python does **not** call the chat API itself and does **not** choose the
next tool. The Codex CLI runs its own loop: model → tool (read a file, write
notes) → model → … until it writes `agent_answer.json` or the timeout
(600s) hits. We parse the CLI’s JSONL (`agent/events.jsonl`) into normalized
events (`agent/trajectory.jsonl`).

Event kinds that matter:

| Kind | Meaning for scoring |
|---|---|
| `retrieve` | a read of a session file (or notes). Gold `dia_id`s are searched in the retrieved text. |
| `write` | a notes mutation. Not evidence. |
| catalog / `INDEX.md` | navigation. Not evidence. |
| `web_search`, `mcp` | outside the workspace. Counted in the audit. Never count as recall. |

A question with **no** successful workspace read is
`harness_execution_failure`, not a wrong answer from missing evidence.
`memory_recall` is the fraction of gold `dia_id`s that showed up in retrieve
text (union over the trajectory). `hop_to_evidence` is the first retrieve
step that hit one. `evidence_retrieved` is all-or-nothing (recall = 1).

The predicted string is `agent_answer.json` if present, else the last agent
message. That string is what LoCoMo F1 and Mem0 J both score. The agent
prompt asks for JSON `{"answer": ...}`; leakage of the wrapper is stripped
before scoring when the parser succeeds.

### 6.3 Three persist cells (same loop, different disk)

All three are still one `codex exec` per question, sequential, same backbone.

| Cell | Sessions on disk at QA time | Notes | What it measures |
|---|---|---|---|
| persist off + full | `sessions/` visible | prompt never mentions notes; `--ephemeral` | retrieval-only. The model can read the haystack. |
| persist on + full | `sessions/` visible | `memory/notes.md` accumulates across questions in that conversation | write ablation. The haystack is still there, so a high score does not mean the notes were necessary. |
| persist on + notes_only | after one ingest `codex exec` per conversation, `sessions/` is moved to `sessions_hidden/` | notes are the only store | harness-as-memory. QA cannot read the original turns. |

Ingest is an extra `codex exec` per conversation (`prompts/agents/ingest_notes_v1`),
**before** that conversation’s questions, and only for `notes_only`. Snapshots
of notes after ingest and after each question are
`agent/notes_snapshots/<sample>/<qid>.md` and `agent/notes_ledger.jsonl`.

### 6.4 How this differs from path A on the same backbone

| | Path A `full_context` | Path C Codex |
|---|---|---|
| Context | one stuffed string in the chat request | files; the model must read them |
| Calls you can see | one Chat Completions request in `reader/traces.jsonl` | one CLI process per question; many model/tool turns inside `agent/events.jsonl` |
| Tools | none | Codex file tools; web search disabled; user MCP ignored |
| Order effects | none (same memory every question) | persist-on notes carry to the next question of that conversation |
| Fair “model vs model+harness” | same backbone, path A vs path C | do not also change the reader prompt or the writer |

A sandwich writer comparison (path B vs Codex-as-writer) is a **third**
claim: Codex writes the summary or graph, then a frozen Chat Completions
reader answers. That reader never enters the agent loop. See
`docs/agent/RUNBOOK_OPENAI_MINI_VS_CODEX_WRITERS.md`.

---

## 7. Where every LLM call is

| When | Who | Sees gold? | Trace |
|---|---|---|---|
| Index build (mem0 / mem0g / rag / openai_memory) | extractor, embedder | no | `experiments/<index_run_id>/` |
| Writer build (path B) | teacher model, per session | no | `memory/teachers/` |
| Answer (path A/B) | reader, one chat completion per question | no | `reader/traces.jsonl` |
| Answer (path C) | model inside `codex exec`, many turns | no | `agent/events.jsonl`, `agent/traces.jsonl` |
| notes_only ingest (path C) | one extra `codex exec` per conversation | no | same agent logs, question id `…-ingest` |
| Judge | autorater, **separate job**, one call per non-category-5 question | **yes** (question + gold + prediction) | `autorater/traces.jsonl` |

Nothing in the answer path is allowed to put `Question.answer` in a prompt.
If you are auditing a trace and the gold string appears before the judge
job, that is a bug.

---

## 8. How scores are derived

Two different scores of the **same** `predicted_answer` string. They are not
interchangeable.

### 8.1 String scores (no API)

`metrics.score_row` on each prediction, written into `metrics.json` and
`predictions.csv` at the end of QA:

| Field | Rule |
|---|---|
| `exact_match` | 1 if SPEC-normalized strings match (lowercased, punctuation stripped, `a`/`an`/`the` removed) |
| `token_f1` | token overlap after that same SPEC normalize. Extra words hurt precision. |
| `locomo_f1` | category rules in §1 (stemmed; multi-hop splits on commas; category 3 uses the first `;` clause; category 5 is the “not mentioned” check) |

`offline_evaluate.py` recomputes these from `predictions.jsonl` with no API.
Campaign overall F1/EM **drop category 5** to match Mem0’s judged subset.
Category plots keep category 5. `runs.parquet` from the pack can still
contain the all-category F1; `load_pack` replaces overall means with the
category-5-excluded example mean.

LoCoMo 2024 Table 3 “Summary RAG” F1 is a **literature pin** from the paper
(different model, different retrieval, adversarial included, 50
conversations). It is not a row this pipeline computed.

### 8.2 Mem0 J (separate autorater job)

`execute-autorater` or `scripts.analysis.run_benchmark` reads the finished
predictions and calls the judge (`prompts/autoraters/autorater_mem0_v1.txt`,
default GPT-4o-mini in these campaigns). The judge returns
`CORRECT` or `WRONG`. `llm_score` is 1 or 0.

- Category 5 is **skipped** (not scored as 0). J’s denominator is the
  non-adversarial questions.
- J is generous about paraphrase. A long Codex answer can be J-correct and
  F1≈0 against a two-word gold string. That gap is the metric, not a second
  answer.
- Mock autorater output is plumbing only. It must not be reported as J.
- Paper Table 2 J (Chhikara et al.) is a literature pin. An OSS clone or a
  Codex cell is not that number.

Agent-only audit scores (not F1, not J), from the trajectory:

- `memory_recall`, `evidence_retrieved`, `hop_to_evidence`
- `failure_mode`: retrieval vs reasoning, using **LoCoMo F1 > 0** as “correct”
  in the on-disk field. Campaign recipe `failure_mode_judge` recomputes that
  split with J, because paraphrase makes F1-zero look like a reasoning failure.

---

## 9. Walk one pack end to end

Pick `experiments/<run_id>/` after QA (and `autorater/` after the judge job).

1. `config.resolved.yaml` — which memory, reader or agent, model, prompt path.
2. `prompts/` — snapshot of the prompt files that run actually loaded.
3. `TRACE.md` — include chain → prompt files → jsonl outputs.
4. Open `data/raw/locomo10.json`, find `sample_id`, and one `qa[i]`.
   Confirm `question_id` in the pack is `{sample_id}-q-{i}` and that
   `reference_answer` equals `qa[i].answer`.
5. **Path A/B:** `predictions.jsonl` row for that id. `memory_text` is what
   the reader saw. `reader/traces.jsonl` is the one completion (model,
   latency, token usage). Gold must not appear in `memory_text`.
6. **Path B only:** `memory/teachers/` for the teacher calls that built that
   memory; `memory/graph/` if it was a graph. `ATTRIBUTION.md` ties the call
   to the summary or triples.
7. **Path C:** `agent/workspaces/<sample_id>/sessions/` should match the JSON
   turns and must not contain the gold answer. `agent/events.jsonl` for that
   `question_id` is the CLI loop. `agent/trajectory.jsonl` is the normalized
   retrieve/write events. `agent_answer.json` content should match
   `predicted_answer` (after JSON unwrap). For `notes_only`, confirm
   `sessions/` was hidden and `notes_snapshots/.../_ingest.md` exists before
   the question snapshots.
8. Recompute LoCoMo F1 for that row with the category rule in §1 if you want
   to check `metrics.json` (overall is the mean; category 5 handling differs
   for campaign tables vs the raw pack mean — see §8.1).
9. If `autorater/autorater_verdicts.jsonl` exists: category 5 rows are
   skipped; others are CORRECT/WRONG. Mean of `llm_score` over non-skipped
   rows is J. The judge trace should contain the gold; the reader trace
   should not.

`_SUCCESS` means that job finished. A later run of the same id without
`--force` skips (`experiment_runner`) or, for a bare `locomo_eval.run`, clears and
regenerates. Do not append autorater files; each judge invocation rewrites
`autorater/`.

---

## 10. Short contrast you can say out loud

LoCoMo here is ten long dialogues plus gold questions. We load them once,
never show the gold to the answer model, and score the predicted string
twice: token overlap (LoCoMo F1) and, in a later job, a paraphrase judge
(Mem0 J).

A raw model call stuffs the memory string into one chat completion per
question, in order, no tools. A sandwich run spends extra model calls
building that string (summaries or a graph), then uses the same one-shot
reader. A Codex run never stuffs the dialogue: each question is one CLI
process whose internal loop reads files, and only then do we score the
string it wrote — plus whether the file reads actually touched the gold
turn ids.
