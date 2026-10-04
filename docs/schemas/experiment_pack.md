# Sandwich audit (`audit_pack.v3`)

On-disk contract for one LoCoMo sandwich run: `experiments/<run_id>/`.

The sandwich **design** freezes data + reader/eval and varies memory. These
modules dump and reload the **audit of those layers** so a condition maps to
results without re-running LLMs. Schema id is `audit_pack.v3`
(`run_meta.json` → `audit_layout.version`).

Call traces (`reader/traces.jsonl`, `memory/writer/calls.jsonl`) record that
an LLM ran. **Claim audit** files record what entered `{memory}` and who is
responsible for each item (question → memory item → writer). That is the
difference between an audit of calls and an audit of claims. Completeness vs
the eight-item checklist: `docs/reports/claim_audit_status.md`.

The pipeline never compares conditions in-process. It dumps **one directory
per YAML / `--method`**, then analysis scripts read that directory.

## How a run becomes an audit

```text
LoCoMo JSON
    │
    ▼
MemoryBuilder / ModelOrchestrator        variable middle (memory method)
        │
        ▼
frozen Reader (gpt-4o-mini)            {memory} + question → predicted answer
        │
        ├─ string metrics (EM, token F1, LoCoMo F1)
        └─ optional autorater (LLM judge, separate CLI)
        │
        ▼
audit_writer  →  experiments/<run_id>/     dump during the run
        │
        ▼
audit_loader  →  compare / autorater / paper-vs-local
```

| Python module | Role | Who imports it |
|---------------|------|----------------|
| `experiment_pack/audit_layout.py` | Folder/file names only. No I/O. | dump and analysis |
| `experiment_pack/claim_audit.py` | Lineage, ranks, writer quality, cost, attribution (call → role → claims), SUMMARY text | `audit_writer` |
| `experiment_pack/prompt_bundle.py` | Snapshot `prompts/` + `TRACE.md` (config → prompt → jsonl) | `run.py`, autorater CLI |
| `experiment_pack/audit_writer.py` | Create reader / writer / graph / claim files | `run.py`, `memory_log.py`, autorater CLI |
| `experiment_pack/audit_loader.py` | Read a finished sandwich audit. No LLM. | `scripts/compare_full_runs.py`, `scripts/analysis/` |

Eval / analysis code should import only:

```python
from src.locomo_eval.experiment_pack.audit_loader import load_sandwich_audit, load_qa_pack
from src.locomo_eval.experiment_pack.audit_layout import AuditPaths
from src.locomo_eval.experiment_pack.verify_pack import verify_pack
from src.locomo_eval.experiment_pack.verify_graph_years import diagnose_graph_year_stagnation
```

CLI: `python -m scripts.analysis.verify_experiments experiments/<run_id>`
(`--graph-years` for graph 2025 vs 2026). Not a Cloud Run job.

Do not import `src.locomo_eval.run`, the writer model, or `experiment_pack.audit_writer`
from an eval branch. That keeps git conflicts on the write path vs analysis
path small.

## What is on disk

```text
experiments/<run_id>/
  TRACE.md                     config → prompt → jsonl map
  SUMMARY.md                    human claim-audit report
  ATTRIBUTION.md                LLM call → role → claims (human)
  attribution.jsonl             same join (machine)
  config.source.yaml            copy of the CLI `--config` file
  config.resolved.yaml          loaded YAML + CLI overrides that actually ran
  cost.json                     token totals + pinned-USD rollup
  prompts/                     snapshot copies (same tree as repo `prompts/`)
    index.json
    readers/
    writers/
    autoraters/
  run_meta.json                 pins + audit_layout paths + agent comparison contract
  metrics.json                  overall string metrics
  predictions.jsonl             QA rows (compat copy at run root)
  predictions.csv
  plots/
  reader/                       frozen sandwich bottom (answer LLM)
    predictions.jsonl
    traces.jsonl
    metrics.json
    schema.json
  agent/                        coding-agent harness (optional; workspace_files)
    traces.jsonl
    trajectory.jsonl            retrieval events + evidence hits
    events.jsonl                raw adapter JSONL
    metrics.json                recall/precision + failure modes + comparison status
    COMPARISON.md               human view of the eight requested controls
    workspaces/<sample_id>/     conversation files the harness saw
  memory/                       sandwich middle ({memory} payload)
    schema.json
    lineage.jsonl               question → injected item → writer
    retrieve_ranks.jsonl        full ranked candidates (losers included)
    writer/                     only when a writer ran
      index.jsonl
      calls.jsonl
      quality.json              parse and yield rates
      sessions.jsonl            index of writer input texts
      sessions/by_sample/<id>/session_<k>.txt
      by_writer/<id>/calls.jsonl
    graph/                      only when a graph was dumped
      index.jsonl
      ingest.jsonl              MERGE / invalidate / skip_dup
      by_sample/<sample_id>.json
  autorater/                    sandwich eval; written by run_benchmark, not run.py
    autorater_verdicts.jsonl
    traces.jsonl
```

| Slice | Load entry | Typical files |
|-------|------------|----------------|
| QA / reader | `load_qa_pack(run_dir)` | `predictions.jsonl`, `metrics.json`, `run_meta.json` |
| Full sandwich | `load_sandwich_audit(run_dir)` | plus traces, writer calls, lineage, ranks, ingest, cost, attribution |
| Paths only | `AuditPaths.from_run_dir(run_dir)` | no I/O |

`load_qa_pack` return keys match `scripts.compare_full_runs.load_pack`
(`dir`, `run_id`, `metrics`, `meta`, `predictions`, `by_qid`).

Missing optional layers load as empty lists (a dataset `session_summaries` run has
no writer calls). `SUMMARY.md` and `cost.json` are written for every QA run.

## How to audit one claim

1. `SUMMARY.md` — sandwich pins, cost, writer quality, pointers.
2. `TRACE.md` / `prompts/` — config include chain and snapshot prompts used.
3. `ATTRIBUTION.md` / `attribution.jsonl` — each LLM call, its role, and claims it made.
4. `memory/lineage.jsonl` — for a `question_id`, which items were injected and
   which writer proposed them.
5. `memory/writer/sessions/` — the session text the writer saw.
6. `memory/graph/ingest.jsonl` — MERGE / invalidate / skip_dup.
7. `memory/retrieve_ranks.jsonl` — candidates that lost to the injected winners.

## Attribution helpers

```python
audit = load_sandwich_audit("experiments/smoke_graph")
openai_calls = audit.writer_calls_for("openai")
items = audit.lineage_for(question_id="conv-26-q-0", sample_id="conv-26")
ranks = audit.retrieve_ranks_for(question_id="conv-26-q-0", sample_id="conv-26")
roles = audit.attribution_for(role="graph")
reader = audit.attribution_for(role="reader", question_id="conv-26-q-0", sample_id="conv-26")
```

Omit a filter to list every matching row (`lineage_for()` is the whole run).
Pass `sample_id` when the same `question_id` could exist in two conversations.
Graph provenance is keyed `(sample_id, edge_id)` so two graphs that both mint
`e0000` cannot leak `proposed_by`. An ingested edge is kept. Missing optional
dump layers load as `[]` / `{}`.

Index keys: `sample_id`, `session_id`, `writer_id`, `question_id`, `item_id`.

`cost.json` USD uses pinned list prices for known OpenAI ids only
(`experiment_pack/claim_audit.py`, `pricing_as_of`). Unknown models contribute
tokens with `usd=null`. This is a rollup, not an invoice.

## Branch split (to avoid merge collisions)

| Branch work | Touch |
|-------------|--------|
| Pipeline / writer / dumps | `run.py`, `writer_model.py`, `experiment_pack/audit_writer.py`, `experiment_pack/claim_audit.py` |
| Evaluation / claims on dumps | `scripts/analysis/`, new eval modules; `audit_loader.py` only if the **on-disk contract** changes |
| Path names / version | `experiment_pack/audit_layout.py` + this doc (rare; bump `audit_pack.v3`) |
