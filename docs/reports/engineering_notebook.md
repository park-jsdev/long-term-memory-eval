# Engineering notebook — system map & extension points

Living notes for humans reviewing the LoCoMo sandwich pipeline.  
**Update when** frozen/variable surfaces or condition IDs change.  
Related: `docs/agent/HUMANS.md`, `docs/agent/AGENTS.md`, `docs/agent/SPEC_v1.md`, `docs/reports/multi_teacher_methodologies.md` (teacher/fusion guide).

---

## 1. Experimental sandwich (what to freeze)

Borrowed from Bowman et al. (2022) “scalable oversight” sandwich: fix top + bottom; only vary the middle.

| Layer | Role | Status in this repo |
|-------|------|---------------------|
| **Top (fixed)** | LoCoMo conversations + questions | `dataset.py` + `data/raw/locomo10.json` |
| **Middle (variable)** | How “memory” text is produced (`raw_chunks`, `session_summaries`, `mem0`, `mem0g`, later teacher/fusion) | `memory.py` + `mem0/` write-index |
| **Bottom (fixed)** | Answer prompt, answer LLM, metrics, reporting | `prompts/`, `readers.py`, `metrics.py`, `report.py` |

**Attribution rule:** if you change prompt *or* model *and* memory between two runs, you cannot cleanly attribute the score delta to memory design alone.

---

## 1b. External APIs & models (inventory)

Keep this list current when providers/models change.

### Active

| Provider | API surface | SDK / endpoint | Auth | Models we use | Where configured | Code |
|----------|-------------|----------------|------|---------------|------------------|------|
| OpenAI Platform | Chat Completions + Embeddings | `openai` Python package | `.env` → `OPENAI_API_KEY` | **`gpt-4o-mini`** default reader, Mem0 writer, judge, and cheap OpenAI teacher; **`gpt-4.1-mini`** / **`gpt-5.6-luna`** (reader robustness); **`text-embedding-3-small`** (Mem0 cosine) | `reader.model`, `teacher.model`, `mem0.extract.model`, `mem0.embed.model`, `configs/autoraters/mem0_gpt-4o-mini.yaml` | `OpenAIChatCaller` / readers / teachers / autorater / extract |
| Anthropic | Messages API | `anthropic` Python package | `.env` → `ANTHROPIC_API_KEY` | **`claude-haiku-4-5`** cheap teacher (plumbing) | `teachers:` / `teacher.provider: anthropic` | `AnthropicTeacherCaller` (`teacher_callers.py`) |
| DeepSeek | OpenAI-compatible Chat Completions | `openai` package + `base_url=https://api.deepseek.com` | `.env` → `DEEPSEEK_API_KEY` | **`deepseek-v4-flash`** cheap teacher (plumbing) | `teachers:` / `teacher.provider: deepseek` | `OpenAIChatCaller` with DeepSeek base URL (`teacher_callers.py`) |

**Default baseline request shape (answer LLM):**

- One system message: `prompts/readers/qa_mem0_v1.txt` filled with memory + question
- GPT-4o-mini, `temperature=0.0`, no explicit completion-token limit
- Mirrors Mem0's released `evaluation/src/openai/predict.py` answer controls
- CLI ablations may override model/prompt/temperature/max-tokens/message-layout;
  regression tests keep `mem0_baseline.yaml` unchanged
- GPT-5.6 (Luna/Terra/Sol) **reader**: `max_completion_tokens` (min 16), `reasoning_effort=none`, no temperature (see `src/locomo_eval/models.py`)
- **Teacher thinking** (write path, default on): Claude extended thinking; DeepSeek V4 thinking; OpenAI GPT-5.x `reasoning_effort=high`. Switch with `teacher.thinking` / `--thinking off`. Ping forces off. Frozen reader is unchanged. `gpt-4o-mini` teachers log the flag but have no reasoning_effort API.

**Commands that call OpenAI:**

```bash
# Live answer generation (bills every Q on every invocation)
python -m src.locomo_eval.run --config configs/writers/raw_chunks.yaml --max-questions 20 --run-id cmp_raw_chunks_n20
python -m src.locomo_eval.run --config configs/writers/session_summaries.yaml --max-questions 20 --run-id cmp_session_summaries_n20
python -m src.locomo_eval.run --config configs/presets/mem0_baseline.yaml --run-id ...
python -m src.locomo_eval.run --config configs/presets/session_summaries_gpt-5.6-luna.yaml --max-questions 5 --run-id ...
python -m src.locomo_eval.run --config configs/writers/teacher_session_summaries.yaml --teacher-model gpt-5.6-luna --max-questions 3 --run-id ...
python -m src.locomo_eval.run --config configs/writers/pooled_teacher_graph.yaml --max-questions 3 --run-id live_pooled_teachers
python -m src.locomo_eval.run --config configs/writers/fused_teacher_graph.yaml --max-questions 3 --run-id live_fused_teachers
python -m src.locomo_eval.run --config configs/writers/fused_teacher_graph_resolve_top_voted.yaml --max-questions 3 --run-id live_fused_resolve_top_voted
python -m src.locomo_eval.mem0.run_index --config configs/writers/mem0.yaml --run-id mem0_locomo10
python -m src.locomo_eval.mem0.run_index --config configs/writers/mem0g.yaml --run-id mem0g_locomo10
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess --eval-questions 10
python -m scripts.analysis.run_benchmark --run experiments/<run_id>  # released GPT-4o-mini judge
```

**Commands that do *not* call OpenAI:**

```bash
python -m src.locomo_eval.run --reader mock ...
python -m src.locomo_eval.offline_evaluate --predictions experiments/<run>/predictions.jsonl
python -m scripts.analysis.run_benchmark --run experiments/<run_id> --autorater mock
python scripts/compare_full_runs.py --runs ... --out ...
python scripts/compare_cross_model.py --runs ... --axis reader --out ...
python -m scripts.analysis.compare_to_paper --runs experiments/full_context_qa experiments/rag_k2_256_qa experiments/mem0_qa --out experiments/compare_paper_vs_local
python -m scripts.analysis.context_window --out experiments/_campaign/context_window
python scripts/prepare_data.py --split all --no-jsonl
python -m src.locomo_eval.mem0.run_index --extractor mock --embedder mock ...
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess
python -m src.locomo_eval.preprocess.run_index --eval-questions 10 --eval-reader mock --run-id smoke_preprocess
python scripts/fetch_locomo.py   # HTTP to GitHub raw only
```

Mock autorater output is a token-overlap plumbing check, not LLM-as-a-Judge:
it is logged as provider/model `mock`, excluded from literature J plots/tables,
and may overrate contradictory answers that share dates or topic words.
Autorater never appends analysis. Every invocation clears its known
generated files plus `plots/` and `tables/`, then rates one prediction file
from scratch.

### Considerations

| Concern | Practice |
|---------|----------|
| Rate limits / RPD | Low tiers (~50/day): use small `max_questions`; each invocation starts from question one |
| Cost | Tokens ∝ memory string length (`raw_chunks` >> `session_summaries` typically); requests ∝ question count per invocation |
| Reproducibility | Log `reader_model`, temp, prompt version, data SHA in `run_meta.json`; freeze YAML in `config.resolved.yaml`; LLM traces in `reader/`, `memory/teachers/`, `autorater/`; claim audit in `SUMMARY.md` / `memory/lineage.jsonl` (completeness: `docs/reports/claim_audit_status.md`) |
| Isolation | Each run id is regenerated from scratch; JSONL is audit-only |
| Eval API | Read dumps via `src.locomo_eval.experiments.audit_loader` (`docs/schemas/experiment_pack.md`); dumps stay in `experiments.audit_writer` |
| Security | Never commit `.env`; example only in `.env.example` |

**Autorater prompt provenance:** local
`prompts/autoraters/autorater_mem0_v1.txt` is adapted from Mem0's pinned
[`ACCURACY_PROMPT`](https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py)
and the paper's
[Appendix A](https://arxiv.org/abs/2504.19413) “Prompt Template for LLM as a
Judge.”

**Run isolation:** every pipeline invocation (QA, autorater, Mem0 index,
preprocess index) clears its generated output and regenerates. `run.py`
clears artifacts for a run id and starts from question one;
`predictions.jsonl` is audit-only. Autorater clears and rewrites its pack.

### Planned (not implemented)

- Strong teacher models (GPT-5.x, Claude Sonnet/Opus, DeepSeek Pro) on the write path.
- Claim-level fusion + LLM validators (see `docs/reports/multi_teacher_methodologies.md` for current resolve baselines).
- Distilled `GraphMemory` subclass (freeze extract).
- Anthropic/DeepSeek as **frozen** answer readers for a robustness table (infra is in `readers.py` / `models.py`).
- Claude Code / OpenCode / Pi harness adapters (configs exist; factory is stubbed). Live Codex after local+GCP mock smokes.

**Agent-level eval (implemented, mock-first):** `workspace_files` dumps the conversation as session markdown. A harness (`mock` or `codex exec --json`) retrieves; we log the trajectory and split `retrieval_failure` vs `reasoning_failure`. Model-only comparison stays `full_context` + one-shot reader. Schema: `docs/schemas/agent_runtime.md`. Do not claim this is AMA-Bench / LongMemEval-V2.

---

## 2. Runtime data path (injection)

```text
Conversation + Question
        │
        ▼
  MemoryBuilder.build()     ← EXPERIMENTAL (middle)
        │
        ▼
  Memory { memory_type, text, source_ids }
        │
        │   .text is the only payload the answer model sees
        ▼
  prompts/*.txt  {memory} + {question}   ← FREEZE for fair memory-condition comparisons
        │
        ▼
  Reader (OpenAI / mock, temp=0)         ← FREEZE for fair memory-condition comparisons
        │
        ▼
  predicted_answer
        │
        ├──────────────► metrics (EM, token F1, LoCoMo F1)  ← deterministic
        │
 gold ──┴──────────────► Autorater (GPT-4o, Mem0 prompt)    ← online evaluator
        │
        ▼
  experiments/<run_id>/  CSV · JSONL · plots · run_meta
  experiments/<run_id>/autorater/  Mem0 F1/BLEU-1/J · tables · latency plots
```

Single-config runner (`run_locomo_pipeline_with_memory_config` in `run.py`)
injects that condition into the answer LLM:

```text
reader.answer(memory.text, q.question, prompt_template)
```

A vs B is two of those calls (different YAML / `--run-id`), then
`scripts/compare_full_runs.py`. This function never compares.

The reader is intentionally dumb about teachers, fusion, and stores.

---

## 3. Conditions (draft set)

| Condition id | What the answer LLM receives | Research question |
|--------------|------------------------------|-------------------|
| `raw_chunks` | Full raw dialog turns (chronological), optional char cap | Does structure help at all? |
| `session_summaries` | LoCoMo-released session summaries (dataset “memory”) | How strong is a structured session-memory bank? |
| `teacher_session_summaries` | Per-session summaries from one teacher LLM | Does a live teacher beat released summaries? Swap teacher model within a family as a robustness check. |
| `mem0` | Top-k timestamped facts from a Mem0 write-index dump (both speakers) | Does the paper extract+update path beat session summaries? Architecture clone — not paper J. |
| `mem0g` | `mem0` plus serialized graph relations | Does the graph add anything if extract is frozen? Later distilled graphs swap `GraphMemory` only. |
| `teacher_graph` | One teacher writes locked Mem0g triples | Does a live graph teacher beat mem0g / summaries? |
| `pooled_teacher_graph` | K teachers; equal_weight / random / round_robin | Does naive pooling help? |
| `fused_teacher_graph` | K teachers; majority-vote / resolve_* fusion | Does consensus fusion help? |
| `rag` | Cosine top-k tiktoken chunks | Mem0 paper RAG baseline |
| `full_context` | Entire timestamped dialog | Mem0 paper full-context baseline |
| `openai_memory` | All extracted timestamped facts (no top-k) | Paper OpenAI privileged-memory protocol clone |
| `top1_teacher` (future) | Top-1 of K teacher memories | Selection enough? |
| `whole_memory_aggregation` (future) | Aggregated whole memories | Synthesis enough? |
| `claim_fusion` (future) | Claim-level fused + validated store (+ retrieve) | Fine-grained fusion win? |

Default memory still uses **provided** LoCoMo session summaries (`session_summaries`). `teacher_session_summaries` is the live single-teacher *summary* replacement. `teacher_graph` / `pooled_teacher_graph` / `fused_teacher_graph` write locked Mem0g via the orchestrator.

Aliases for convenience (legacy numbered ids still resolve):

| Alias | Resolves to |
|-------|-------------|
| `raw` / `raw_dialog` / `c0_raw` | `raw_chunks` |
| `session_summary` / `c1_session_summary` | `session_summaries` |
| `teacher` / `c1_teacher` | `teacher_session_summaries` |

---

## 4. Where to change what

### 4.1 Answer prompt (bottom — freeze after lock-in)

| Item | Location |
|------|----------|
| Template text | Default parity: `prompts/readers/qa_mem0_v1.txt`; memory experiments: `prompts/readers/qa_v1.txt` |
| Loader | `src/locomo_eval/prompts.py` |
| Select in config | `pipeline.prompt_path` in YAML |
| CLI | `--prompt path/to/qa_v2.txt` |
| Logged as | `prompt_version` = filename stem |

Do **not** fork prompts per condition for the main table. If you ablate prompts, re-run **all** conditions under the new prompt and report the setup change.

### 4.2 Memory design (middle — primary experiment surface)

| Item | Location |
|------|----------|
| Interface | `MemoryBuilder.build(conversation, question) -> Memory` in `memory.py` |
| Registry | `get_memory_builder(name)` |
| Select in config | `pipeline.memory` |
| CLI | `--memory raw_chunks` or `--memory session_summaries` |
| Logged as | `memory_type` on each prediction + `run_meta` |

**Add a new condition**

1. Implement a class with a descriptive `name` (e.g. `top1_teacher`), not a numbered code.
2. Register it in `get_memory_builder`.
3. Prefer writing long intermediates to `experiments/<run_id>/memories/` later; still end by filling `Memory.text`.
4. Run with the **same** `--prompt` / reader model as other memory conditions.

**Teacher (`teacher_session_summaries` / `teacher_graph`):** YAML `teacher.model` (or `--teacher-model`) selects the write-path LLM. Multi-teacher configs use `teachers:` plus `orchestrator.pool` / `orchestrator.fusion`. Logged on `Memory.teacher_model`, `run_meta.json`, and each prediction row. Do not change `reader.model` in the same comparison if you want the delta attributed to the teacher.

**True multi-teacher write path (middle):** `TeacherOrchestrator` walks HLD (i) session blocks, calls K teachers (`gpt-4o-mini` / `claude-haiku-4-5` / `deepseek-v4-flash` for plumbing), pools or majority-fuses triples, and MERGE/invalidates into locked `Mem0GraphMemory`. Orchestrator is software (`fusion.py`), not one teacher-orchestrator LLM.

**Reader-model robustness:** YAML `reader.model` / `--model` / `configs/presets/session_summaries_gpt-5.6-luna.yaml`. Compare with `scripts/compare_cross_model.py --axis reader`. Do not mix with a memory-condition claim.

**Mem0 write-index (`mem0` / `mem0g`):** `python -m src.locomo_eval.mem0.run_index` walks HLD (i) session blocks as eval-style message pairs (`batch_size=2`, dual speaker indexes, role-flip, user-only extract). Vector update is ADD/UPDATE/DELETE/NONE vs top `s=10`. `mem0g` also fills an in-memory `GraphMemory` (no Neo4j). Dumps: `experiments/<run_id>/mem0_index/`. Builders **load** that dump and cosine-retrieve (`top_k=30` NL facts per speaker); they must not re-extract. Freeze extract+update **and** this retriever when the claim is “new graph only” (swap `GraphMemory`).

**Deterministic preprocess dump:** `python -m src.locomo_eval.preprocess.run_index` parses locomo10.json with no LLM (`llm_calls=0`). Dump: `experiments/<run_id>/preprocess/` (SessionBlocks + SessionDocuments). `raw_chunks` / `session_summaries` can format from that dump (`--preprocess-index-run-id`); default retrieve is concatenate-all. `--eval-questions 10` then runs the frozen reader on those two sanity memories (10 questions each). Later write/retrieve paths (top-k, LLM compress) should consume the same dump rather than re-parse JSON.

Default **reader** is `gpt-4o-mini` + `prompts/readers/qa_mem0_v1.txt` (Mem0 ANSWER_PROMPT with `{memory}`/`{question}`) for `raw_chunks`, `session_summaries`, `mem0`, and `mem0g`. Override YAML/`--model`/`--prompt` for a separate bottom-layer axis. This is an **architecture clone**, not a number clone of Tables 1–2.

### 4.3 Multi-teacher write path (middle)

`TeacherOrchestrator` + `fusion.py` (software, not one LLM):

```text
WRITE (variable): session blocks → K teachers → pool/fuse → Mem0GraphMemory
READ  (fixed): graph edges → Memory.text → frozen prompt → answer LLM → metrics
```

Conditions: `teacher_graph` (K=1, interchangeable model), `pooled_teacher_graph` (equal_weight / random / round_robin), `fused_teacher_graph` (majority_vote skeleton). Strong models and learned fusion come later.

### 4.4 Reader / metrics / reports (bottom — freeze)

| Item | Location | Notes |
|------|----------|--------|
| OpenAI / mock | `readers.py` | One Chat Completions call per question |
| Env / keys | `.env` + `env.py` | never commit secrets |
| Scoring | `metrics.py`, `src/metrics/locomo_qa.py` | dual: SPEC + LoCoMo F1 |
| Offline rescore | `python -m src.locomo_eval.offline_evaluate ...` | string metrics only; no API; not an LLM autorater |
| Online autorater | `python -m scripts.analysis.run_benchmark --run ...` | Mem0 prompt, GPT-4o default, category 5 skipped; fresh non-appending report every invocation |
| Autorater literature pins | `mem0_baselines.py` | Mem0 paper Tables 1–2; comparison only, not local Mem0 re-runs |
| Two-run compare | `scripts/analysis/compare_predictions.py` | paired LoCoMo F1 boxplot + histograms |
| Audit pack | `report.py` + `experiments/audit_writer.py` → `experiments/<run_id>/` | CSV/JSON/plots + claim audit (`SUMMARY.md`, `ATTRIBUTION.md`, lineage, ranks, ingest, cost) |

---

## 5. How to compare raw_chunks vs session_summaries

Freeze bottom, vary middle only:

```bash
# Same max_questions, model, prompt; different memory + run_id
python -m src.locomo_eval.run --config configs/writers/raw_chunks.yaml --max-questions 20 --run-id cmp_raw_chunks_n20
python -m src.locomo_eval.run --config configs/writers/session_summaries.yaml --max-questions 20 --run-id cmp_session_summaries_n20

# Side-by-side metrics table
python scripts/compare_full_runs.py \
  --runs experiments/cmp_raw_chunks_n20 experiments/cmp_session_summaries_n20 \
  --out experiments/compare_raw_chunks_session_summaries
```

Writes `overall.csv`, `by_category.csv`, `paired_questions.csv`, `SUMMARY.md`, `plots/` (including LoCoMo F1 **boxplot** + side-by-side **histograms**), and memory-text **sanity** (`fraction_same_memory_text` ≈ 0 means conditions filled the reader with different strings).

Two-pack LoCoMo F1 plots alone (no sandwich SUMMARY):

```bash
python -m scripts.analysis.compare_predictions \
  --a experiments/cmp_raw_chunks_n20 --b experiments/cmp_session_summaries_n20 \
  --out experiments/compare_raw_chunks_session_summaries
```

### Inspect the flattened QA table

```bash
python scripts/prepare_data.py --split all --no-jsonl   # data/processed/qa_all.csv
# omit --no-jsonl to also refresh the large JSONL with full context
```

| Artifact | Why |
|----------|-----|
| `metrics.json` | overall LoCoMo F1 / EM |
| `metrics_by_category.csv` | category-wise |
| `predictions.csv` | qualitative: same `question_id`, different `memory_type` |
| `run_meta.json` | confirm model + prompt_version match |

**Validity checklist**

- [ ] Same `reader_model`, `temperature`, `prompt_version`
- [ ] Same question subset (`max_questions` / sample filter)
- [ ] Differ only in `memory_type` / builder
- [ ] Both condition runs regenerated from scratch (reusing a run id does not keep old answers).

---

## 6. Control panel (frozen vs changeable)

| Piece | Freeze for the main memory-condition table? | Experiment instead? |
|-------|------------------------------|---------------------|
| Dataset + question IDs | Yes | No |
| Answer prompt | Yes (after lock) | Only as a separate meta-setup |
| Answer model + decode | Yes | Model swap = new setup label |
| Metrics | Always | No |
| **MemoryBuilder / write / store** | No | **Yes — main axis** |
| Retriever + token budget | Policy fixed across conditions | Budget ablations as a second axis |
| Report columns | Prefer stable | Additive fields OK |

---

## 7. Suggested growth order

1. `raw_chunks` vs `session_summaries` under frozen prompt/model (this draft).  
2. Lock short-answer prompt if needed; re-run both conditions.  
3. Mem0 / Mem0g write-index over locomo10, then a later QA command with frozen `qa_v1` (or a separate Mem0-answer-prompt setup).  
4. Swap only `GraphMemory` (distilled graph) with extract frozen.  
5. Later `top1_teacher` / `whole_memory_aggregation` / `claim_fusion` + fixed retriever budget.  

---

## 8. File index (engineering)

```text
configs/presets/mem0_baseline.yaml         # Mem0-parity controls + session_summaries memory
configs/writers/raw_chunks.yaml
configs/writers/session_summaries.yaml
configs/presets/session_summaries_gpt-5.6-luna.yaml  # session_summaries + GPT-5.6 Luna reader
configs/writers/teacher_session_summaries.yaml            # live single teacher
configs/writers/mem0.yaml / mem0g.yaml     # Mem0 write-index + load/retrieve seam
prompts/readers/qa_v1.txt                 # alternate reader prompt
prompts/readers/qa_mem0_v1.txt             # pinned released Mem0 answer prompt
prompts/teachers/teacher_session_v1.txt
prompts/writers/mem0_extract_v1.txt        # + mem0_update_v1 / mem0g_*.txt
src/locomo_eval/models.py          # model catalog / API kwargs
src/locomo_eval/teachers.py        # Mock + OpenAI teacher
src/locomo_eval/memory.py          # builders including mem0 / mem0g
src/locomo_eval/mem0/              # write-index package
src/locomo_eval/run.py             # wires builder → reader → report
scripts/compare_full_runs.py            # two memory conditions side-by-side
scripts/analysis/compare_predictions.py  # two-pack LoCoMo F1 boxplot + histograms
scripts/analysis/run_benchmark.py        # Mem0 F1/BLEU-1/J + tables/plots
src/locomo_eval/autorater.py             # GPT-4o / mock CORRECT-WRONG judge
src/locomo_eval/mem0_metrics.py          # Mem0 lexical metrics + latency
src/locomo_eval/mem0_baselines.py        # published Table 1–2 pins
scripts/compare_cross_model.py     # reader/teacher model robustness
docs/reports/engineering_notebook.md  # this file
```
