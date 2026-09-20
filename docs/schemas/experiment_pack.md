# Sandwich audit (`audit_pack.v2`)

On-disk contract for one LoCoMo sandwich run: `experiments/<run_id>/`.

The sandwich **design** freezes data + reader/eval and varies memory. These
modules dump and reload the **audit of those layers** so a condition maps to
results without re-running LLMs. Schema id is `audit_pack.v2`
(`run_meta.json` → `audit_layout.version`).

Call traces (`reader/traces.jsonl`, `memory/teachers/calls.jsonl`) record that
an LLM ran. **Claim audit** files record what entered `{memory}` and who is
responsible for each item (question → memory item → teacher). That is the
difference between an audit of calls and an audit of claims. Completeness vs
the eight-item checklist: `docs/reports/claim_audit_status.md`.

The pipeline never compares conditions in-process. It dumps **one directory
per YAML / `--method`**, then analysis scripts read that directory.

## How a run becomes an audit

```text
LoCoMo JSON
    │
    ▼
MemoryBuilder / TeacherOrchestrator     variable middle (memory method)
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
| `experiments/audit_layout.py` | Folder/file names only. No I/O. | dump and analysis |
| `experiments/claim_audit.py` | Lineage, ranks, teacher quality, cost, attribution (call → role → claims), SUMMARY text | `audit_writer` |
| `experiments/prompt_bundle.py` | Snapshot `prompts/` + `TRACE.md` (config → prompt → jsonl) | `run.py`, autorater CLI |
| `experiments/audit_writer.py` | Create reader / teachers / graph / claim files | `run.py`, `memory_log.py`, autorater CLI |
| `experiments/audit_loader.py` | Read a finished sandwich audit. No LLM. | `scripts/compare_full_runs.py`, `scripts/analysis/` |

Eval / analysis code should import only:

```python
from src.locomo_eval.experiments.audit_loader import load_sandwich_audit, load_qa_pack
from src.locomo_eval.experiments.audit_layout import AuditPaths
from src.locomo_eval.experiments.verify_pack import verify_pack
from src.locomo_eval.experiments.verify_graph_years import diagnose_graph_year_stagnation
```

CLI: `python -m scripts.analysis.verify_experiments experiments/<run_id>`
(`--graph-years` for teacher_graph 2025 vs 2026). Not a Cloud Run job.

Do not import `src.locomo_eval.run`, `teachers`, or `experiments.audit_writer`
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
    teachers/
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
    lineage.jsonl               question → injected item → teacher
    retrieve_ranks.jsonl        full ranked candidates (losers included)
    teachers/                   only when teachers ran
      index.jsonl
      calls.jsonl
      fusion.jsonl              proposed_by / kept
      quality.json              parse / yield / keep rates
      sessions.jsonl            index of teacher input texts
      sessions/by_sample/<id>/session_<k>.txt
      by_teacher/<id>/calls.jsonl
    graph/                      only when a fused graph was dumped
      index.jsonl
      ingest.jsonl              MERGE / invalidate after fusion
      by_sample/<sample_id>.json
  autorater/                    sandwich eval; written by run_benchmark, not run.py
    autorater_verdicts.jsonl
    traces.jsonl
```

| Slice | Load entry | Typical files |
|-------|------------|----------------|
| QA / reader | `load_qa_pack(run_dir)` | `predictions.jsonl`, `metrics.json`, `run_meta.json` |
| Full sandwich | `load_sandwich_audit(run_dir)` | plus traces, teachers, lineage, ranks, ingest, cost, attribution |
| Paths only | `AuditPaths.from_run_dir(run_dir)` | no I/O |

`load_qa_pack` return keys match `scripts.compare_full_runs.load_pack`
(`dir`, `run_id`, `metrics`, `meta`, `predictions`, `by_qid`).

Missing optional layers load as empty lists (a `session_summaries` run has
no teacher calls). `SUMMARY.md` and `cost.json` are written for every QA run.

## How to audit one claim

1. `SUMMARY.md` — sandwich pins, cost, teacher quality, pointers.
2. `TRACE.md` / `prompts/` — config include chain and snapshot prompts used.
3. `ATTRIBUTION.md` / `attribution.jsonl` — each LLM call, its role, and claims it made.
4. `memory/lineage.jsonl` — for a `question_id`, which items were injected and
   which teacher proposed them.
5. `memory/teachers/sessions/` — the session text that teacher saw.
6. `memory/teachers/fusion.jsonl` — `proposed_by` / `kept` for that triple.
7. `memory/graph/ingest.jsonl` — MERGE / invalidate / skip_dup after fusion.
8. `memory/retrieve_ranks.jsonl` — candidates that lost to the injected winners.

## Attribution helpers

```python
audit = load_sandwich_audit("experiments/smoke_fused_teachers")
openai_calls = audit.teacher_calls_for("openai")
kept = audit.fusion_kept_for(sample_id="conv-26")
items = audit.lineage_for(question_id="conv-26-q-0", sample_id="conv-26")
ranks = audit.retrieve_ranks_for(question_id="conv-26-q-0", sample_id="conv-26")
roles = audit.attribution_for(role="teacher_graph")
reader = audit.attribution_for(role="reader", question_id="conv-26-q-0", sample_id="conv-26")
```

Omit a filter to list every matching row (`lineage_for()` is the whole run).
Pass `sample_id` when the same `question_id` could exist in two conversations.
Graph provenance is keyed `(sample_id, edge_id)` so two graphs that both mint
`e0000` cannot leak `proposed_by`. Missing optional dump layers load as `[]` /
`{}`. An omitted fusion `kept` flag is not treated as kept.

Index keys: `sample_id`, `session_id`, `teacher_id`, `question_id`, `item_id`.

`cost.json` USD uses pinned list prices for known OpenAI ids only
(`experiments/claim_audit.py`, `pricing_as_of`). Unknown models contribute
tokens with `usd=null`. This is a rollup, not an invoice.

## Branch split (to avoid merge collisions)

| Branch work | Touch |
|-------------|--------|
| Pipeline / teachers / dumps | `run.py`, `teachers.py`, `experiments/audit_writer.py`, `experiments/claim_audit.py` |
| Evaluation / claims on dumps | `scripts/analysis/`, new eval modules; `audit_loader.py` only if the **on-disk contract** changes |
| Path names / version | `experiments/audit_layout.py` + this doc (rare; bump `audit_pack.v2`) |
