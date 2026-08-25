# LoCoMo multi-teacher memory research

conda + plain YAML + plain JSON + CSV/plots.  
**Now (v0.1):** sandwich **read path** with draft conditions **`raw_chunks`** (raw dialog) and **`session_summaries`**, fixed OpenAI answer model.  
**Later:** `teacher_session_summaries`, then `top1_teacher` / `whole_memory_aggregation` / `claim_fusion`. Maps: [`docs/reports/engineering_notebook.md`](docs/reports/engineering_notebook.md), [`docs/agent/HUMANS.md`](docs/agent/HUMANS.md).

---

## System diagrams

### 1. Architecture — files & components

How major packages relate. Solid arrows = call / data; dashed = configuration.

```mermaid
flowchart TB
  subgraph Config["Config & assets"]
    YAML["configs/*.yaml"]
    PROMPT["prompts/qa_*.txt"]
    ENV[".env OPENAI_API_KEY"]
    DATA["data/raw/locomo10.json"]
  end

  subgraph CLI["Entry points"]
    RUN["src/locomo_eval/run.py"]
    EVAL["src/locomo_eval/offline_evaluate.py"]
    PREP["scripts/prepare_data.py"]
    CMP["scripts/compare_full_runs.py"]
    FETCH["scripts/fetch_locomo.py"]
  end

  subgraph Core["src/locomo_eval"]
    DS["dataset.py"]
    SCH["schemas.py"]
    MEM["memory.py"]
    PR["prompts.py"]
    RD["readers.py"]
    HASH["utils/llm_response_hash.py\n(unwired)"]
    MET["metrics.py"]
    REP["report.py"]
  end

  LOCOMO["src/metrics/locomo_qa.py"]
  OUT["experiments/run_id/"]

  FETCH --> DATA
  YAML -.-> RUN
  PROMPT -.-> RUN
  ENV -.-> RD
  DATA --> RUN
  DATA --> PREP

  RUN --> DS
  RUN --> MEM
  RUN --> PR
  RUN --> RD
  RUN --> MET
  RUN --> REP
  DS --> SCH
  MEM --> SCH
  RD -.-> HASH
  RD --> PR
  MET --> LOCOMO
  REP --> OUT
  EVAL --> MET
  EVAL --> OUT
  CMP --> OUT
```

### 2. Memory path — configs, schema, I/O

Experimental **middle** = choose a `MemoryBuilder`. Everything else reads `Memory.text`.

```mermaid
flowchart LR
  subgraph Inputs
    CFG["YAML: pipeline.memory\nraw_chunks | session_summaries"]
    JSON["locomo10.json"]
  end

  subgraph Parse["dataset.py → schemas"]
    CONV["Conversation\nsessions, summaries, questions"]
    Q["Question\nid, text, answer, category"]
  end

  subgraph Builders["memory.py"]
    Raw["raw_chunks\nRawConversationMemoryBuilder"]
    Summ["session_summaries\nSessionSummaryMemoryBuilder"]
    Later["teacher / fusion later"]
  end

  subgraph OutMem["schemas.Memory"]
    MT["memory_type"]
    TXT["text  ← inject into prompt"]
    SRC["source_ids"]
  end

  JSON --> CONV
  JSON --> Q
  CFG --> Raw
  CFG --> Summ
  CFG --> Later
  CONV --> Raw
  CONV --> Summ
  Q -.->|"optional later\nquery-aware"| Raw
  Raw --> OutMem
  Summ --> OutMem
  Later --> OutMem
```

| Condition | Config | Input from dataset | Output (`Memory.text`) |
|-----------|--------|--------------------|-------------------------|
| `raw_chunks` | `configs/raw_chunks.yaml` | session turns + dates | chronological raw dialog |
| `session_summaries` | `configs/session_summaries.yaml` | `session_summary` | concatenated session summaries |
| later | *(future)* | teachers / store | fused or selected memory string |

### 3. Evaluation experiment pipeline — changing conditions

**Sandwich:** freeze top (data) + bottom (prompt, model, metrics); vary only memory.

```mermaid
flowchart TB
  subgraph FrozenTop["FIXED TOP"]
    D["LoCoMo dataset + questions"]
  end

  subgraph Variable["VARIABLE MIDDLE — change this"]
    direction LR
    M0["--memory raw_chunks\nor raw_chunks.yaml"]
    M1["--memory session_summaries\nor session_summaries.yaml"]
    MX["future conditions"]
  end

  subgraph FrozenBottom["FIXED BOTTOM — do not change mid-suite"]
    P["prompts/qa_v1.txt"]
    R["OpenAIReader temp=0"]
    S["metrics EM / token-F1 / LoCoMo F1"]
  end

  subgraph Artifacts["experiments/"]
    A["run A: predictions + metrics"]
    B["run B: predictions + metrics"]
    C["compare_full_runs.py → SUMMARY + plots"]
  end

  D --> M0
  D --> M1
  D --> MX
  M0 --> P
  M1 --> P
  MX --> P
  P --> R
  R --> S
  S --> A
  S --> B
  A --> C
  B --> C
```

**Fair comparison recipe**

1. Same `prompt_path`, `reader.model`, `temperature`, `max_questions` / sample filter.  
2. Differ only `pipeline.memory` (or config file).  
3. Distinct `--run-id`s → then `scripts/compare_full_runs.py`.  
4. Sanity: `fraction_same_llm_request_hash ≈ 0` and different mean `memory_chars` in the compare report. That hash is a SHA-256 of the intended LLM request (not a live store lookup).

### 4. Memory + LLM lifecycle — builder, schema, prompts

Request path for one QA item. `LlmResponseHash` exists under `utils/` but is **not** used by `run.py` yet.

```mermaid
sequenceDiagram
  participant Run as run.py
  participant MB as MemoryBuilder
  participant Mem as Memory schema
  participant Pr as prompts.py + qa_v1.txt
  participant Rd as OpenAIReader
  participant API as OpenAI API
  participant Sc as metrics + report

  Run->>MB: build(conversation, question)
  MB-->>Mem: memory_type, text, source_ids
  Run->>Pr: render_qa_prompt(template, memory.text, question)
  Pr-->>Rd: full user prompt
  Rd->>API: chat.completions
  API-->>Rd: predicted_answer
  Rd-->>Run: answer + meta(usage)
  Run->>Sc: Prediction row → score → CSV/JSONL/plots
```

**What is frozen vs experimental in this lifecycle**

| Stage | Freeze for memory-condition tables? | Knob |
|-------|---------------------------|------|
| Builder → `Memory.text` | No | condition |
| Prompt template file | Yes (after lock) | `prompts/qa_*.txt` |
| Reader model / decode | Yes | `reader.*` in YAML |
| Store | Prohibited in evaluation | `utils/llm_response_hash.py` remains unused; factories reject it |
| Metrics | Always | `metrics.py` / LoCoMo F1 |

---

## External APIs & models (living inventory)

Track every third-party API here so reviews and paper methods stay accurate.  
Update this section when you add Claude, Gemini, local HF, etc.

### Currently used

| Provider | Product | How we call it | Auth | Models in configs | Code entry |
|----------|---------|----------------|------|-------------------|------------|
| **OpenAI** | Platform API ([platform.openai.com](https://platform.openai.com)) | Official Python SDK `openai` ≥1.30 — **Chat Completions** (`client.chat.completions.create`) | `OPENAI_API_KEY` in repo-root `.env` (auto-loaded) | Reader: `gpt-4.1-mini`; autorater: `gpt-4o` (`gpt-4o-mini` for Mem0 released-script parity) | `readers.py` → `OpenAIReader`; `autorater.py` → `OpenAIAutorater` |

| Decode defaults (frozen bottom unless re-locked) | Value |
|-----------------------------------------------------------------|-------|
| temperature | `0.0` |
| max_tokens (completion) | `64` |
| System message | `"You answer questions using only the provided memory."` |
| User message | rendered `prompts/qa_v1.txt` with `{memory}` + `{question}` |
| min_request_interval_s | `0.5` |
| max_retries on 429 | `8` (Retry-After / exponential backoff) |

**Not used yet:** Anthropic Claude, Google Gemini, Azure OpenAI, local Hugging Face generation, batch/Responses API, Assistants API.

### Commands that hit (or skip) OpenAI

| Command | API? | Notes |
|---------|------|--------|
| `python -m src.locomo_eval.run --config configs/raw_chunks.yaml ...` | Yes if `reader.provider: openai` | One fresh Chat Completions call per QA; cache/resume disabled |
| `... --reader mock` | No | Offline plumbing |
| `python -m src.locomo_eval.offline_evaluate --predictions ...` | No | String-metric rescore only (not an LLM autorater) |
| `python -m scripts.analysis.run_benchmark --run ...` | Yes | GPT-4o Mem0 LLM-as-a-Judge over stored answers |
| `python -m scripts.analysis.run_benchmark --run ... --autorater mock` | No | Autorater/report plumbing smoke |
| `python scripts/compare_full_runs.py ...` | No | Metrics / plots / cache-key *rehash* offline |
| `python scripts/prepare_data.py ...` | No | Local JSON → CSV/JSONL |
| `python scripts/fetch_locomo.py` | GitHub raw HTTP | Dataset file only, not OpenAI |

### Config knobs

```yaml
# configs/raw_chunks.yaml, session_summaries.yaml, baseline.yaml
reader:
  provider: openai          # or mock
  model: gpt-4.1-mini       # override: --model gpt-4.1
  temperature: 0.0
  max_tokens: 64
  max_retries: 8
  min_request_interval_s: 0.5
  max_wait_s: 3600
```

CLI overrides:

```bash
python -m src.locomo_eval.run --config configs/session_summaries.yaml \
  --model gpt-4.1 --max-questions 20 --run-id cmp_session_summaries_gpt41_n20
```

### Cost & quota considerations

| Topic | Detail |
|-------|--------|
| Billing unit | 1 fresh Chat Completions request per question on every invocation |
| Full LoCoMo | ~1986 Qs per condition → ~1986 requests if starting cold |
| Free/low tier | Can hit **RPD ~50/day** → use small `max_questions` and distinct run ids |
| Prompt size | `raw_chunks` can be ~tens of k chars (truncated by `memory_max_chars`); drives **input tokens** not request count |
| LLM response hash | Utility remains in `utils/llm_response_hash.py`, but evaluation factories reject it |
| Raising limits | [Billing](https://platform.openai.com/account/billing) + [Rate limits](https://platform.openai.com/account/rate-limits) |

### Planned swaps (not wired)

| Role | Candidate | Why later |
|------|-----------|-----------|
| Fixed answer LLM | Claude (Anthropic API) | User preference for eval-phase; keep same prompt freeze rules |
| Teachers (write path) | GPT / Gemini / DeepSeek | Multi-teacher middle layer only |

When Claude is added: new `reader.provider: anthropic`, document model IDs and env var (`ANTHROPIC_API_KEY`) in this same section.

---

## Setup

```bash
conda create -n distillation python=3.11 -y
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py
```

For live runs, copy `.env.example` → `.env` and set `OPENAI_API_KEY` (gitignored; auto-loaded).

## OpenAI rate limits

See **External APIs & models** above for the authoritative inventory. Short form: low-tier Orgs may see **~50 RPD** for `gpt-4.1-mini`; full eval needs higher limits because runs do not resume.
---

## Phase 1 — raw_chunks vs session_summaries (frozen reader/prompt, vary memory)

See [`docs/reports/engineering_notebook.md`](docs/reports/engineering_notebook.md).

```bash
# Offline: both conditions mockable
python -m src.locomo_eval.run --config configs/raw_chunks.yaml --reader mock --max-questions 5 --run-id smoke_raw_chunks
python -m src.locomo_eval.run --config configs/session_summaries.yaml --reader mock --max-questions 5 --run-id smoke_session_summaries

# Live (same model/prompt; n=20 for a cheap comparison)
python -m src.locomo_eval.run --config configs/raw_chunks.yaml --max-questions 20 --run-id cmp_raw_chunks_n20
python -m src.locomo_eval.run --config configs/session_summaries.yaml --max-questions 20 --run-id cmp_session_summaries_n20
python scripts/compare_full_runs.py --runs experiments/cmp_raw_chunks_n20 experiments/cmp_session_summaries_n20 --out experiments/compare_raw_chunks_session_summaries
```

Outputs under `experiments/<run_id>/`: `predictions.csv`, `metrics.json`, `plots/`, `run_meta.json`, and **`memory/`** (exact `{memory}` texts + schema — see [`docs/schemas/memory_runtime.md`](docs/schemas/memory_runtime.md)).  
Compare also writes `SUMMARY.md`, `paired_questions.csv`, and cache/memory sanity plots under `experiments/compare_raw_chunks_session_summaries/`.

Flatten the dataset for inspection:

```bash
python scripts/prepare_data.py --split all --no-jsonl   # data/processed/qa_all.csv
```

Rescore with string metrics only (no API, not an LLM autorater):

```bash
python -m src.locomo_eval.offline_evaluate --predictions experiments/<run_id>/predictions.jsonl
```

Run the Mem0-style online autorater on those same stored predictions:

```bash
# Offline report smoke
python -m scripts.analysis.run_benchmark \
  --run experiments/<run_id> --autorater mock

# Live GPT-4o judge (Mem0 F1 / BLEU-1 / J + literature and latency plots)
python -m scripts.analysis.run_benchmark \
  --run experiments/<run_id>
```

Mock mode is only a token-overlap plumbing check. It can mark contradictory
dates correct, is logged as `mock_sanity_not_llm_judge`, and is excluded from
the literature J comparison. Actual J requires the live GPT-4o command.
Every invocation clears the prior generated autorater tables, plots, verdicts,
metrics, and metadata, then regenerates them from one prediction file. It
never resumes or appends autorater analysis.

This implements the Mem0 judge protocol and published baseline tables, not
Mem0 extraction/update. It skips adversarial category 5 as the paper does.
See `experiments/<run_id>/autorater/SUMMARY.md`, `tables/`, and `plots/`.

Prompt source: Mem0's pinned
[`ACCURACY_PROMPT`](https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py),
also printed in [paper Appendix A](https://arxiv.org/abs/2504.19413). The
local adaptation is `prompts/autorater_mem0_v1.txt`.

No evaluation cache or resume is active. Reader, teacher, and autorater
factories reject non-null `llm_response_hash`; answer runs clear prior
generated artifacts for the run id and regenerate from question one.
Autorater runs also regenerate and reject source rows marked `cached=true`.

Tests:

```bash
python -m unittest tests/test_evaluation_pipeline.py -q
python -m unittest tests/test_autorater_sanity.py -q
```

## Layout

```text
configs/                  # baseline, raw_chunks, session_summaries
prompts/                  # qa_v1, teacher_session_v1, autorater_mem0_v1
src/locomo_eval/          # run, memory, readers, autorater, metrics, reports
docs/schemas/             # memory_runtime.md + memory_io.schema.json
src/metrics/locomo_qa.py  # official LoCoMo F1
scripts/                  # fetch, prepare_data, compare; analysis/run_benchmark
docs/reports/             # engineering_notebook
docs/agent/               # SPEC, AGENTS, HUMANS, traces
docs/reflections/         # version audit writeups
experiments/<run_id>/     # audit pack
data/raw/locomo10.json    # fetched, gitignored
data/processed/qa_all.csv # flattened QA (no full context)
```

## Docs for humans and agents

- Humans: [`docs/agent/HUMANS.md`](docs/agent/HUMANS.md)
- Agents: [`docs/agent/AGENTS.md`](docs/agent/AGENTS.md)
- Engineering notebook: [`docs/reports/engineering_notebook.md`](docs/reports/engineering_notebook.md)
- Spec: [`docs/agent/SPEC_v1.md`](docs/agent/SPEC_v1.md)
- Runtime memory schema / logs: [`docs/schemas/memory_runtime.md`](docs/schemas/memory_runtime.md)
- History: [`docs/agent/traces/`](docs/agent/traces/)
- Reflections: [`docs/reflections/`](docs/reflections/)

## Citation

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
