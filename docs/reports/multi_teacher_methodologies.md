# Multi-teacher methodologies — researcher guide

One-page overview of how teachers, pooling, and fusion fit in the LoCoMo sandwich pipeline. For implementation detail see `docs/reports/engineering_notebook.md` §4.3 and `src/locomo_eval/fusion.py`.

---

## 1. Sandwich layout

| Layer | Fixed or variable | What it is |
|-------|-------------------|------------|
| **Top** | Fixed | LoCoMo conversations + questions (`data/raw/locomo10.json`) |
| **Middle** | **Variable** | How memory text is built (condition id in YAML) |
| **Bottom** | Fixed (for fair memory comparisons) | Answer prompt (`prompts/readers/qa_mem0_v1.txt`), reader (`gpt-4o-mini`), string metrics + Mem0 autorater |

Change only one middle variable per experimental claim. Reader-model or prompt swaps are a **separate** axis (`compare_cross_model.py`).

---

## 2. Teacher methodologies (condition ids)

### Single-teacher (summary path)

| Condition | Write path | Research question |
|-----------|------------|-------------------|
| `session_summaries` | LoCoMo-released session summaries (no LLM) | Baseline structured memory |
| `teacher_session_summaries` | One LLM summarizes each session | Does a live teacher beat released summaries? |

### Single-teacher (graph path)

| Condition | Write path | Research question |
|-----------|------------|-------------------|
| `teacher_graph` | One teacher extracts Mem0g triples per session | Does a live graph teacher help? |

### Multi-teacher (graph path)

All multi-teacher graph conditions share:

1. **Preprocess** — deterministic session blocks (`preprocess.run_index`)
2. **TeacherOrchestrator** — software controller (not one meta-LLM)
3. **K teachers** — each calls an LLM via `teacher_callers.py` → `teachers.py`
4. **Pool** — which teachers run this session
5. **Fusion / resolve** — how their triples combine
6. **Mem0GraphMemory** — locked ingest + graph conflict resolver
7. **Frozen reader** — `gpt-4o-mini` + `qa_mem0_v1.txt`

| Condition | Pool | Fusion | Question |
|-----------|------|--------|----------|
| `pooled_teacher_graph` | `equal_weight` / `random` / `round_robin` | `none` (union) | Does naive pooling help? |
| `fused_teacher_graph` | `equal_weight` | `majority_vote` (default) | Does exact-triple consensus help? |
| `fused_teacher_graph` + resolve configs | `equal_weight` | `resolve_*` (see §4) | How should slot disagreements be broken? |

**Future (not implemented):** `top1_teacher`, `claim_fusion`, learned/validator fusion, distilled `GraphMemory` subclass.

**Future MR (sandwich audit, not this branch):** question → injected triple/`source_id` → `proposed_by`; dump graph ingest MERGE/invalidate ops so fusion logs match the store the reader sees; retrieve cosine ranks; teacher empty-session / parse-fail rollup; copy YAML + `rng_seed` into `experiments/<run_id>/`. Do not mix that work into the multi-teacher write-path MR.

---

## 3. LLMs and where they run

### Write path (teachers) — variable middle

| Provider | Module | Models (default roster) | Role |
|----------|--------|-------------------------|------|
| OpenAI | `teacher_callers.py` → `OpenAIChatCaller` | `gpt-4o-mini` | Graph/summary extraction |
| Anthropic | `teacher_callers.py` → `AnthropicTeacherCaller` | `claude-haiku-4-5` | Graph/summary extraction |
| DeepSeek | `teacher_callers.py` → `OpenAIChatCaller` + DeepSeek base URL | `deepseek-v4-flash` | Graph/summary extraction |
| Mock | `MockTeacherCaller` | tagged mock ids | Offline smoke |

**Connection to orchestrator:**

```
TeacherOrchestrator._propose()
  → Teacher.extract_session_graph()   [teachers.py]
    → TeacherCaller.complete()          [teacher_callers.py]
  → fuse_proposals()                    [fusion.py]
  → Mem0GraphMemory.ingest_triples()
```

`teacher_callers.py` is **write-path only**. It is **not** the answer reader and **not** Mem0 extract/update (those live under `mem0/`).

**Thinking default:** on for teachers (`teacher.thinking: true`). Ping forces off. Reader stays unchanged.

**Prompts:** `prompts/teachers/teacher_graph_v1.txt` (graph), `prompts/teachers/teacher_session_v1.txt` (summaries).

### Read path (frozen for memory comparisons)

| Role | Model | Module |
|------|-------|--------|
| Answer reader | `gpt-4o-mini` | `readers.py` |
| Mem0 autorater (J) | `gpt-4o-mini` | `autorater.py` |

---

## 4. Pool and fusion policies

Configured under `orchestrator:` in YAML or via CLI `--pool` / `--fusion`.

### Pool (which teachers run per session)

| Policy | Behavior |
|--------|----------|
| `single` | First teacher in roster only |
| `equal_weight` | All K teachers every session |
| `round_robin` | One teacher, cycling by session index |
| `random` | One teacher, seeded RNG |

### Fusion — exact triple match

| Policy | Behavior |
|--------|----------|
| `none` | Union all triples (dedupe exact keys) |
| `majority_vote` | Keep triple if ≥ k teachers propose the **same** (source, rel, target); default k = ⌈K/2⌉ |

### Fusion — slot disagreement resolution

When teachers agree on `(source, relationship)` but propose **different targets** (e.g. `alice -- lives_in -- boston` vs `nyc`):

| Policy | 1-1-1 split | 1-2 split |
|--------|-------------|-----------|
| `resolve_top_voted` | Tie-break by roster order | Pick the 2-vote target |
| `resolve_first` | First teacher in roster wins | First among tied max-vote |
| `resolve_random` | Random among tied max-vote (seeded) | Pick 2-vote target |
| `resolve_round_robin` | Cycle among tied targets by session index | Pick 2-vote target |
| `resolve_confidence` | Highest Σ confidence weights | One high-confidence beats two low |

Per-relation `confidence` in teacher JSON is optional; default weight = 1.0.

**Runnable configs** (each uses `memory: fused_teacher_graph` with a different `orchestrator.fusion`):

- `configs/writers/fused_teacher_graph.yaml` — `majority_vote` (baseline)
- `configs/writers/fused_teacher_graph_resolve_top_voted.yaml`
- `configs/writers/fused_teacher_graph_resolve_first.yaml`
- `configs/writers/fused_teacher_graph_resolve_random.yaml`
- `configs/writers/fused_teacher_graph_resolve_round_robin.yaml`
- `configs/writers/fused_teacher_graph_resolve_confidence.yaml`

---

## 5. Analysis and where results live

### Run pack (`experiments/<run_id>/`)

| Artifact | Contents |
|----------|----------|
| `predictions.jsonl` / `.csv` | Per-question answers + memory text preview |
| `metrics.json` | EM, token F1, LoCoMo category F1 |
| `run_meta.json` | Models, pool, fusion, git hash, data hash |
| `memory/teachers/fusion.jsonl` | Per-session fusion audit (`relations`, `slots` for resolve-*) |
| `memory/teachers/calls.jsonl` | Per-teacher LLM calls (entities, relations, latency) |
| `memory/graph/` | Serialized graph index |
| `reader/traces.jsonl` | Answer LLM traces |
| `autorater/` | Mem0 J / F1 / BLEU-1 (after `run_benchmark`) |

### Compare two conditions (offline, no API)

```bash
python scripts/compare_full_runs.py \
  --runs experiments/run_a experiments/run_b \
  --out experiments/compare_a_b
```

Cross-model reader comparisons: `scripts/compare_cross_model.py`.

### Autorater (online judge)

```bash
python -m scripts.analysis.run_benchmark --run experiments/<run_id>
python -m scripts.analysis.run_benchmark --run experiments/<run_id> --autorater mock  # smoke
```

Multi-seed J aggregate: `python -m scripts.analysis.aggregate_seeds --packs ...`

---

## 6. How to reproduce

```bash
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py

# Mock smoke (no API)
python -m src.locomo_eval.run --config configs/writers/teacher_graph.yaml \
  --reader mock --teacher mock --max-questions 3 --run-id smoke_teacher_graph
python -m src.locomo_eval.run --config configs/writers/pooled_teacher_graph.yaml \
  --reader mock --teacher mock --max-questions 3 --run-id smoke_pooled
python -m src.locomo_eval.run --config configs/writers/fused_teacher_graph.yaml \
  --reader mock --teacher mock --max-questions 3 --run-id smoke_fused
python -m src.locomo_eval.run --config configs/writers/fused_teacher_graph_resolve_top_voted.yaml \
  --reader mock --teacher mock --max-questions 3 --run-id smoke_resolve_top_voted

# Provider plumbing (live keys in .env)
python -m src.locomo_eval.ping_teachers

# Live run (costly) — example
python -m src.locomo_eval.run --config configs/writers/fused_teacher_graph_resolve_first.yaml \
  --run-id fused_resolve_first_locomo10

# Unit tests
python -m pytest tests/test_teacher_orchestrator.py -q
```

Override fusion without a new YAML: `--fusion resolve_top_voted`.

---

## 7. Key source files

| File | Role |
|------|------|
| `src/locomo_eval/teacher_callers.py` | Write-path LLM clients (not reader) |
| `src/locomo_eval/teachers.py` | Teacher ABC + graph/summary extraction |
| `src/locomo_eval/teacher_orchestrator.py` | Session walk, pool, fuse, graph ingest |
| `src/locomo_eval/fusion.py` | Pool/fusion/resolve policies |
| `src/locomo_eval/memory.py` | Condition builders (`teacher_graph`, `pooled_*`, `fused_*`) |
| `src/locomo_eval/run.py` | Single-condition CLI (dumps a sandwich audit via `audit_writer`) |
| `src/locomo_eval/experiments/audit_writer.py` | Write `reader/`, `memory/teachers/`, `memory/graph/` during a run |
| `src/locomo_eval/experiments/audit_loader.py` | Read a finished `experiments/<run_id>/` sandwich audit |
| `src/locomo_eval/reasoning_extractor.py` | Shared reasoning-text helpers (reader + teachers) |

---

## 8. What we do not claim

- OSS Mem0/Mem0g clones are **not** paper Table 1–2 J numbers.
- Multi-teacher resolve policies are **baseline heuristics**; claim-level fusion and LLM validators are future work.
- Teacher attribution in this branch is **fusion/call logs** (`memory/teachers/fusion.jsonl`, `calls.jsonl`), not a per-question “which teacher answered this QA item” join.
- Do not mix reader/prompt changes into a fusion claim without labeling that axis separately.
