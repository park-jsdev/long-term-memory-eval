# LoCoMo multi-teacher memory research

conda + plain YAML + plain JSON + CSV/plots.  
**Now (v0.1):** sandwich **read path** with draft conditions **C0** (raw dialog) and **C1** (session summaries), fixed OpenAI answer model.  
**Later:** multi-teacher fusion (C2–C4). Maps: [`docs/reports/engineering_notebook.md`](docs/reports/engineering_notebook.md), [`docs/agent/HUMANS.md`](docs/agent/HUMANS.md).

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
    EVAL["src/locomo_eval/evaluate.py"]
    PREP["scripts/prepare_data.py"]
    CMP["scripts/compare_runs.py"]
    FETCH["scripts/fetch_locomo.py"]
  end

  subgraph Core["src/locomo_eval"]
    DS["dataset.py"]
    SCH["schemas.py"]
    MEM["memory.py"]
    PR["prompts.py"]
    RD["readers.py"]
    CACHE["cache.py"]
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
  RD --> CACHE
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
    CFG["YAML: pipeline.memory\nc0_raw | c1_session_summary"]
    JSON["locomo10.json"]
  end

  subgraph Parse["dataset.py → schemas"]
    CONV["Conversation\nsessions, summaries, questions"]
    Q["Question\nid, text, answer, category"]
  end

  subgraph Builders["memory.py"]
    C0["c0_raw\nRawConversationMemoryBuilder"]
    C1["c1_session_summary\nSessionSummaryMemoryBuilder"]
    CX["c2…c4 later"]
  end

  subgraph OutMem["schemas.Memory"]
    MT["memory_type"]
    TXT["text  ← inject into prompt"]
    SRC["source_ids"]
  end

  JSON --> CONV
  JSON --> Q
  CFG --> C0
  CFG --> C1
  CFG --> CX
  CONV --> C0
  CONV --> C1
  Q -.->|"optional later\nquery-aware"| C0
  C0 --> OutMem
  C1 --> OutMem
  CX --> OutMem
```

| Condition | Config | Input from dataset | Output (`Memory.text`) |
|-----------|--------|--------------------|-------------------------|
| C0 | `configs/c0_raw.yaml` | session turns + dates | chronological raw dialog |
| C1 | `configs/c1_session_summary.yaml` | `session_summary` | concatenated session summaries |
| C2–C4 | *(future)* | teachers / store | fused or selected memory string |

### 3. Evaluation experiment pipeline — changing conditions

**Sandwich:** freeze top (data) + bottom (prompt, model, metrics); vary only memory.

```mermaid
flowchart TB
  subgraph FrozenTop["FIXED TOP"]
    D["LoCoMo dataset + questions"]
  end

  subgraph Variable["VARIABLE MIDDLE — change this"]
    direction LR
    M0["--memory c0_raw\nor c0_raw.yaml"]
    M1["--memory c1_session_summary\nor c1_*.yaml"]
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
    C["compare_runs.py → SUMMARY + plots"]
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
3. Distinct `--run-id`s → then `scripts/compare_runs.py`.  
4. Sanity: `fraction_same_cache_key ≈ 0` and different mean `memory_chars` in the compare report.

### 4. Memory + LLM lifecycle — builder, schema, prompts

Request path from build to cache key (one QA item).

```mermaid
sequenceDiagram
  participant Run as run.py
  participant MB as MemoryBuilder
  participant Mem as Memory schema
  participant Pr as prompts.py + qa_v1.txt
  participant Rd as OpenAIReader
  participant C as disk cache
  participant API as OpenAI API
  participant Sc as metrics + report

  Run->>MB: build(conversation, question)
  MB-->>Mem: memory_type, text, source_ids
  Run->>Pr: render_qa_prompt(template, memory.text, question)
  Pr-->>Rd: full user prompt
  Rd->>C: key = hash(model, temp, max_tokens, prompt)
  alt cache hit
    C-->>Rd: answer
  else cache miss
    Rd->>API: chat.completions
    API-->>Rd: predicted_answer
    Rd->>C: store answer
  end
  Rd-->>Run: answer + meta(cached, usage)
  Run->>Sc: Prediction row → score → CSV/JSONL/plots
```

**What is frozen vs experimental in this lifecycle**

| Stage | Freeze for C0–C4 tables? | Knob |
|-------|---------------------------|------|
| Builder → `Memory.text` | No | condition |
| Prompt template file | Yes (after lock) | `prompts/qa_*.txt` |
| Reader model / decode | Yes | `reader.*` in YAML |
| Cache | Side-effect | same key ⇔ same payload only |
| Metrics | Always | `metrics.py` / LoCoMo F1 |

---

## External APIs & models (living inventory)

Track every third-party API here so reviews and paper methods stay accurate.  
Update this section when you add Claude, Gemini, local HF, etc.

### Currently used

| Provider | Product | How we call it | Auth | Models in configs | Code entry |
|----------|---------|----------------|------|-------------------|------------|
| **OpenAI** | Platform API ([platform.openai.com](https://platform.openai.com)) | Official Python SDK `openai` ≥1.30 — **Chat Completions** (`client.chat.completions.create`) | `OPENAI_API_KEY` in repo-root `.env` (auto-loaded) | **Default:** `gpt-4.1-mini` · **Paper-style option:** `gpt-4.1` (set `reader.model`) | `src/locomo_eval/readers.py` → `OpenAIReader` |

| Decode defaults (frozen bottom for C0/C1 suite unless re-locked) | Value |
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
| `python -m src.locomo_eval.run --config configs/c0_raw.yaml ...` | Yes if `reader.provider: openai` | One Chat Completions call per *uncached* QA |
| `... --reader mock` | No | Offline plumbing |
| `python -m src.locomo_eval.evaluate --predictions ...` | No | Rescore only |
| `python scripts/compare_runs.py ...` | No | Metrics / plots / cache-key *rehash* offline |
| `python scripts/prepare_data.py ...` | No | Local JSON → CSV/JSONL |
| `python scripts/fetch_locomo.py` | GitHub raw HTTP | Dataset file only, not OpenAI |

### Config knobs

```yaml
# configs/c0_raw.yaml, c1_session_summary.yaml, baseline.yaml
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
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml \
  --model gpt-4.1 --max-questions 20 --run-id cmp_c1_gpt41_n20
```

### Cost & quota considerations

| Topic | Detail |
|-------|--------|
| Billing unit | 1 Chat Completions request per uncached question |
| Full LoCoMo | ~1986 Qs per condition → ~1986 requests if cache cold |
| Free/low tier | Can hit **RPD ~50/day** → use `max_questions`, resume same `--run-id`, disk cache |
| Prompt size | C0 raw can be ~tens of k chars (truncated by `memory_max_chars` on C0); drives **input tokens** not request count |
| Cache key | `hash(provider, model, temperature, max_tokens, full_prompt)` under `experiments/cache/` |
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

See **External APIs & models** above for the authoritative inventory. Short form: low-tier Orgs may see **~50 RPD** for `gpt-4.1-mini`; full eval needs higher limits or multi-day resume.
---

## Phase 1 — C0 vs C1 (frozen reader/prompt, vary memory)

See [`docs/reports/engineering_notebook.md`](docs/reports/engineering_notebook.md).

```bash
# Offline: both conditions mockable
python -m src.locomo_eval.run --config configs/c0_raw.yaml --reader mock --max-questions 5 --run-id smoke_c0
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --reader mock --max-questions 5 --run-id smoke_c1

# Live (same model/prompt; n=20 for a cheap comparison)
python -m src.locomo_eval.run --config configs/c0_raw.yaml --max-questions 20 --run-id cmp_c0_n20
python -m src.locomo_eval.run --config configs/c1_session_summary.yaml --max-questions 20 --run-id cmp_c1_n20
python scripts/compare_runs.py --runs experiments/cmp_c0_n20 experiments/cmp_c1_n20 --out experiments/compare_c0_c1
```

Outputs under `experiments/<run_id>/`: `predictions.csv`, `metrics.json`, `plots/`, `run_meta.json`, and **`memory/`** (exact `{memory}` texts + schema — see [`docs/schemas/memory_runtime.md`](docs/schemas/memory_runtime.md)).  
Compare also writes `SUMMARY.md`, `paired_questions.csv`, and cache/memory sanity plots under `experiments/compare_c0_c1/`.

Flatten the dataset for inspection:

```bash
python scripts/prepare_data.py --split all --no-jsonl   # data/processed/qa_all.csv
```

Rescore without API:

```bash
python -m src.locomo_eval.evaluate --predictions experiments/<run_id>/predictions.jsonl
```

Tests:

```bash
python -m unittest tests.test_pipeline_sanity -q
```

## Layout

```text
configs/                  # baseline, c0_raw, c1_session_summary
prompts/qa_v1.txt
src/locomo_eval/          # run, memory, memory_log, readers, metrics, report
docs/schemas/             # memory_runtime.md + memory_io.schema.json
src/metrics/locomo_qa.py  # official LoCoMo F1
scripts/                  # fetch, prepare_data, compare_runs
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
