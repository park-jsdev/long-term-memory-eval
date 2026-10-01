# Unrolling the Experiment Pipeline

This repository scores long-term memory systems on a fixed test. LoCoMo supplies the conversations, the questions, and the token-F1 rules. A model or a Codex agent supplies an answer. Mem0 supplies a second judge on that same answer. We did not author those protocols. We fix the data and the scorer, and we change only how memory is built and read.

The shape is a sandwich. The top is LoCoMo. The bottom is the scorer. The middle is the system under test. A fair comparison changes the middle and leaves the bread alone.

One question moves through several systems. Each system owns a different computation. This note unrolls that path twice: once as a researcher auditing files, and once as whatever is allowed to "think" at that step. The second view is the contamination check. If the model can see the gold answer, the result is not a memory result.

## Who owns what

| System | Owns | Does not own |
|---|---|---|
| LoCoMo (Maharana et al., 2024) | The released dialogues, questions, gold answers, evidence ids, and the category F1 rules | The Mem0 judge, our memory methods, the agent loop |
| This repository | Loading, the memory middle, the answer call, string scoring, the audit pack, the experiment runner | The benchmark text, the paper's published tables |
| The answer model | One Chat Completions completion, or the model turns inside Codex | Retrieval policy, the score, the gold |
| Codex | The agent loop: model, tool, model, until it writes an answer | Which files we put on disk, and how we score |
| Mem0 (Chhikara et al., 2025) | The answer-prompt wording we pin, the CORRECT/WRONG judge, and the extract/update/graph prompts used by the optional `mem0` / `mem0g` clones | Our `graph` writer, and any number in a paper table |

A number in a notebook is one of those owners' numbers. It is not automatically the others'.

## One question, from YAML to a notebook

Take a single LoCoMo question. The same outer path runs for every question. What changes is the middle.

```text
configs/*.yaml
    → experiment runner          one cell, hashed id, no scoring
        → eval harness           load LoCoMo, build memory, ask, score strings
            → model or Codex     the only place an answer is invented
        → audit pack             predictions, traces, prompts, cost
    → autorater job              Mem0 judge, later, separate process
    → aggregate                  parquet over finished cells
    → notebook                   tables and plots, no new answers
```

### 1. Configs declare the cell

A YAML file names the memory method, the model, the prompt, and the seed. It does not contain logic. `includes:` deep-merges files; later keys win; lists replace.

**Researcher.** Open `config.resolved.yaml` in the finished pack. That is the file that actually ran, after includes and CLI overrides. `config.source.yaml` is the file you pointed at. `prompts/` in the pack is a copy of every prompt path that YAML named.

**Model.** Nothing. No model has been called. A config cannot leak a gold answer into a prompt by itself, but a prompt file can. The check is later, in the trace.

The experiment runner reads `configs/experiments/*.yaml` and expands a matrix into cells. One cell is one memory method, one reader or agent, one seed. The run id is `<experiment-slug>-<8 hex>`, hashed from that identity. Change the memory method and you get a new directory. Change the judge and you do not: the judge is a later job on the same predictions.

### 2. The experiment runner starts one cell

`src/experiment_runner` is a scheduler. It does not answer questions and it does not make the model faster. Inside a cell, questions are still answered one at a time. What can run in parallel is independent cells: one Cloud Run task per cell, after the YAML has been expanded.

The runner skips a cell that already has `_SUCCESS`, unless `--force`. A retry replaces the pack. It does not resume mid-question.

**Researcher.** `manifest/runs.jsonl` lists the cells. `_SUCCESS` means that stage finished. It does not mean the answers are good.

**Model.** Still nothing. The runner's job is to launch `locomo_eval` with the resolved config.

### 3. The harness loads LoCoMo and keeps gold on the side

`src/locomo_eval` is the evaluation harness. `dataset.py` reads `data/raw/locomo10.json` and does not edit it. The pin is commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` of the LoCoMo release: ten conversations.

Each sample has sessions of turns (`dia_id`, speaker, text, a date string), optional dataset-written session summaries, optional observations, and a `qa` list. We assign `question_id` as `{sample_id}-q-{index}` in that list's order.

| Field | Researcher | Model |
|---|---|---|
| Dialogue turns | Source of memory text and of workspace files | May see them, depending on the middle |
| Dataset session summaries | Used only when the memory method is `session_summaries` and no writer model is set | Sees the concatenated summaries, not the raw turns |
| Observations | Parsed onto the sample. Preprocess dumps can list them. Answer builders do not | Does not see them |
| `qa[].question` | The question we ask | Sees it |
| `qa[].answer` | Gold. Scorers and the later judge | Must not see it |
| `qa[].evidence` | Gold `dia_id`s. Used after the answer, to score a trajectory | Must not be used to choose what a live model reads |

Questions inside one conversation run in qa-array order. A smoke cap (`--max-questions`) is round-robin across conversations. A full run is file order. There is no thread pool over questions.

### 4. The middle builds what the answerer may see

This is the variable slice of the sandwich. Python builds it before the answer, except for Codex, where the model has to read files itself.

**Dataset text, no write-time model.** `raw_chunks` is the turns. `full_context` is the timestamped transcript. `session_summaries` with no `writer.model` is LoCoMo's own session-summary strings. These are deterministic. The same conversation yields the same memory text for every question.

**One writer model, then a frozen reader.** If `writer.model` is set, that one model writes `session_summaries` (one call per session, `prompts/writers/session_summary_v1.txt`) or a `graph` (entity/relation JSON, then software writes the locked graph schema). The reader that answers is still the frozen reader, usually `gpt-4o-mini` and `prompts/readers/qa_mem0_v1.txt`. A writer-model change is not a reader-model change.

**Optional Mem0-paper clones.** `mem0`, `mem0g`, `rag`, and `openai_memory` are built by a separate index job and reused. QA reads the dump. They are not in the current OpenAI-and-Codex claim. They exist so a later cell can compare against those methods without rewriting the scorer.

**Codex workspace.** Python writes `sessions/session_N.md` and `INDEX.md` with no model. Each turn keeps its `dia_id`. Gold answers are not in the files. `Memory.text` here is a pointer to the workspace, not the transcript.

**Researcher.** For a stuffed or written memory, `predictions.jsonl` field `memory_text` is what the reader received. Writer calls are under `memory/writer/`. A graph also has `memory/graph/`. For Codex, open `agent/workspaces/<sample_id>/` and confirm the gold string is absent.

**Model.** The writer sees one session at a time: speakers, date, session text. It does not see the question list and it does not see gold. The reader, on the next step, sees `{memory}` and `{question}` only. The Codex model sees the workspace files it chooses to read, plus the question in its prompt.

### 5. The answer is one call, or a loop we do not run ourselves

Two answer paths. Both produce one string, `predicted_answer`. LoCoMo F1 and the Mem0 judge both score that string.

**Chat Completions.** One request per question, temperature 0. The harness calls the API. There is no tool loop. The trace is one row in `reader/traces.jsonl`.

**Codex.** The harness starts one `codex exec` per question and then waits. Codex runs its own loop: the model asks to read a file, Codex returns the file, the model asks again, until it writes `agent_answer.json` or the timeout hits. We do not pick the next tool. We do turn off web search, ignore user config and user rules, and sandbox the process. Persist-off is read-only and ephemeral. Persist-on may write notes that the next question in the *same* conversation can see. `notes_only` hides `sessions/` after an ingest pass, so the only store left is the notes.

A question with no successful workspace read is a harness failure, not a wrong memory. Native Codex that cannot enforce the shared tool limits is an audit, not a comparison, until `comparison_status` is `comparable`.

**Researcher.** Path A/B: one completion in `reader/traces.jsonl`. Path C: `agent/events.jsonl` is the CLI loop; `agent/trajectory.jsonl` is the normalized retrieve and write events. `agent_answer.json` should match `predicted_answer` after the JSON wrapper is stripped.

**Model.** This is the only step that invents an answer. It knows the question and the memory it was given or the files it opened. It does not know the gold string, the category, or which `dia_id`s the benchmark marked as evidence. The mock agent used in tests *does* open session files that contain gold evidence ids, and it returns `Unknown.`. That mock is plumbing. It is not a result.

### 6. String scoring happens in the same process, with no model

When the answer returns, `score_prediction` in `src/metrics/locomo_qa.py` runs. This is the LoCoMo metric, kept close to `task_eval/evaluation.py` at the same pin. No API.

| Category | JSON id | Rule |
|---|---|---|
| Single-hop | 4 | Stemmed token F1 against the gold string |
| Multi-hop | 1 | Split prediction and gold on commas; multi-answer F1 |
| Temporal | 2 | Stemmed token F1 |
| Open-domain | 3 | Gold is cut at the first `;`, then stemmed token F1 |
| Adversarial | 5 | 1 if the prediction says "not mentioned" or "no information available", else 0 |

Normalization lowercases, strips punctuation, drops `a`/`an`/`the`/`and`, and stems. Reporting order is 4, 1, 2, 3, 5. Those ids are the ids in the JSON. They are not the prose numbering in the paper's section 4.1.

The pack also stores SPEC `exact_match` and `token_f1`. Those do not stem, and they do not use the category rules. They are a second string metric. They are not the LoCoMo number.

`offline_evaluate` recomputes the string metrics from `predictions.jsonl`. It does not call a model. Use it when you want to confirm `metrics.json` without trusting the process that wrote it.

**Researcher.** `metrics.json` and `predictions.csv` hold these scores. The pack's overall F1 includes category 5. Campaign tables that follow the Mem0 subset drop category 5 from the overall mean and keep it in the category plot. Read the column name before comparing two overalls.

**Model.** No model. The scorer is the first component that is supposed to see the gold answer.

### 7. The Mem0 judge is a later job, and it is allowed to see gold

`execute-autorater` reads the finished predictions and asks a judge model, prompt pinned from Mem0's `ACCURACY_PROMPT`, whether the prediction is CORRECT or WRONG. Category 5 is skipped, not scored as zero. J is the mean over the questions that were judged.

The judge sees the question, the gold answer, and the prediction. That is the protocol. It must not run inside the answer process. If you find the gold string in `reader/traces.jsonl`, `memory_text`, or a workspace file, the cell is invalid. If you find it only in `autorater/traces.jsonl`, that is correct.

J and LoCoMo F1 are not interchangeable. J accepts paraphrase. A long answer can be J-correct and F1 near zero against a two-word gold string. That gap is the metric.

A mock judge writes plumbing output. It must not be reported as J. Paper Table 1–2 J is a literature pin. An OSS clone, a Codex cell, or a local judge run is not that number.

### 8. The pack is the audit

Each cell writes `experiments/<run_id>/` and then stops. The runner does not interpret it.

| Path | What it proves |
|---|---|
| `predictions.jsonl` | Question, gold, prediction, memory text, string scores |
| `reader/traces.jsonl` | The Chat Completions request and response |
| `memory/writer/` | Writer calls and the session text the writer saw |
| `memory/lineage.jsonl` | Question → memory item → writer |
| `agent/events.jsonl` | The Codex loop, unprocessed |
| `agent/trajectory.jsonl` | Retrieve and write events, joined to evidence ids after the fact |
| `run_meta.json` | Models, prompt, data hash, git hash, layout version |
| `TRACE.md` | Include chain → prompt files → outputs |
| `cost.json` | Token totals |
| `autorater/` | Judge verdicts and judge traces, if that job has run |

`memory_recall` on an agent row is the fraction of gold `dia_id`s that showed up in text the agent actually retrieved. The join happens after the loop. It is not a hint the agent received.

### 9. Aggregation and notebooks do not answer new questions

`aggregate` collects finished cells into parquet: one row per cell, one row per question. `collect-full` also copies the packs, including `memory/` and `agent/`. `report` reads an analysis YAML and writes tables and plots. Missing packs are skipped and listed. They are not filled with zeros.

Notebooks are thin. They load that report. They are local by default. A plotted number should trace back to `examples.parquet`, then to `predictions.jsonl`, then to the trace from step 5 or 7.

**Researcher.** Start from the table cell, open the question row, recompute F1 with the category rule, and read the trace. The gold string in the judge trace is expected. The gold string anywhere the model could read is not.

**Model.** None of these stages call one.

## What we reproduce, and the differences that matter

**LoCoMo's data and F1 rules.** We load the released `locomo10` file at the pin above. We do not rerun the paper's larger GPT-3.5 retrieval grid, and we do not rewrite the dialogues. Category handling matches upstream `task_eval`: stemming, the comma split for multi-hop, the first clause before `;` for open-domain, and the adversarial phrase check. Two differences are ours, and both are labeled. We also store SPEC exact match and token F1, which the paper does not use. Campaign overalls drop category 5, which is Mem0's judged subset, not LoCoMo's overall.

**LoCoMo's session summaries.** When no writer model is set, `session_summaries` is the dataset field. That is not a model we ran, and it is not the paper's Summary RAG row. That paper row is a literature pin: a different retriever and a different reader.

**Mem0's answer prompt and judge.** `prompts/readers/qa_mem0_v1.txt` and `prompts/autoraters/autorater_mem0_v1.txt` are pinned text (judge pin `ece7ff6b`). The judge skips category 5, as Mem0 does. The model that judges is whatever the autorater config names, often `gpt-4o-mini`. Same prompt, different judge model, different J. Do not present it as Table 2.

**Mem0's memory methods.** `mem0` and `mem0g` follow the pinned extract, update, and graph prompts (`ece7ff6b`, graph `69a832dc`). `rag` is the paper's chunk-and-embed clone. `full_context` stuffs the transcript the way that baseline does. These are architecture clones. They are not Mem0 Platform numbers.

**Our graph is not `mem0g`.** `graph` is one writer model emitting relations, then software writing the locked graph schema. There is no Mem0 extract/update loop and no second model voting. Say `graph` when that is the cell. Say `mem0g` only for the index clone.

**Codex is the harness, not a memory paper.** We invoke the Codex CLI. The loop inside the CLI is Codex's. Our contribution is the workspace, the tool limits, the trajectory audit, and the frozen scorer on the string Codex returns.

## Why a result is not contaminated

All of the following hold for a live cell. A mock smoke is allowed to break the scientific reading; it is not allowed to be published as one.

1. `dataset.py` does not write back to `locomo10.json`.
2. The reader prompt template has `{memory}` and `{question}`. It has no gold slot.
3. Writer prompts have the session text. They have no question gold and no evidence list.
4. Workspace files are turns and, when persist is on, notes the agent itself wrote. Evidence ids are joined to the trajectory after `codex exec` returns.
5. Observations are not copied into memory text or into the workspace.
6. The string scorer and the judge are the components that receive `Question.answer`. The judge runs in a later process.
7. `verify_pack` checks that gold strings of sufficient length do not appear in `memory_text`. A failed check means the cell is invalid.
8. Category 5 is scored by LoCoMo's phrase rule in `metrics.json`, and omitted from J. Those are different denominators on purpose.
9. Shared `mem0` / `rag` indexes are built once from the dialogues, not from the questions' gold answers. QA only reads them.

If you are adding a memory method, the test to keep is the one this list already states: build the prompt the model will see, and show that the gold answer is not in it. Then score with the existing functions. Do not add a new F1.
