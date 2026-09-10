# Sandwich audit (`audit_pack.v1`)

On-disk contract for one LoCoMo sandwich run: `experiments/<run_id>/`.

The sandwich **design** freezes data + reader/eval and varies memory. These
modules dump and reload the **audit of those layers** so a condition maps to
results without re-running LLMs. Schema id is `audit_pack.v1`
(`run_meta.json` → `audit_layout.version`).

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
| `experiments/audit_writer.py` | Create `reader/`, `memory/teachers/`, `memory/graph/`, judge traces | `run.py`, `memory_log.py`, autorater CLI |
| `experiments/audit_loader.py` | Read a finished sandwich audit. No LLM. | `scripts/compare_full_runs.py`, `scripts/analysis/` |

Eval / analysis code should import only:

```python
from src.locomo_eval.experiments.audit_loader import load_sandwich_audit, load_qa_pack
from src.locomo_eval.experiments.audit_layout import AuditPaths
```

Do not import `src.locomo_eval.run`, `teachers`, or `experiments.audit_writer`
from an eval branch. That keeps git conflicts on the write path vs analysis
path small.

## What is on disk

```text
experiments/<run_id>/
  TRACE.md                     config → prompt → jsonl map (start here)
  config.resolved.yaml         merged YAML actually used
  config.source.yaml           copy of the CLI `--config` file
  prompts/                     snapshot copies (same tree as repo `prompts/`)
    index.json
    readers/
    writers/
    teachers/
    autoraters/
  run_meta.json                 pins + audit_layout paths
  metrics.json                 overall string metrics
  predictions.jsonl           QA rows (compat copy at run root)
  predictions.csv
  plots/
  reader/                      frozen sandwich bottom (answer LLM)
    predictions.jsonl
    traces.jsonl
    metrics.json
    schema.json
  memory/                      sandwich middle ({memory} payload)
    schema.json
    teachers/                  only when teachers ran
      index.jsonl
      calls.jsonl
      fusion.jsonl             proposed_by / kept
      by_teacher/<id>/calls.jsonl
    graph/                     only when a fused graph was dumped
      index.jsonl
      by_sample/<sample_id>.json
  autorater/                   sandwich eval; written by run_benchmark, not run.py
    autorater_verdicts.jsonl
    traces.jsonl
```

| Slice | Load entry | Typical files |
|-------|------------|----------------|
| QA / reader | `load_qa_pack(run_dir)` | `predictions.jsonl`, `metrics.json`, `run_meta.json` |
| Full sandwich | `load_sandwich_audit(run_dir)` | plus `reader/traces.jsonl`, `memory/teachers/*`, `autorater/*` |
| Paths only | `AuditPaths.from_run_dir(run_dir)` | no I/O |

`load_qa_pack` return keys match `scripts.compare_full_runs.load_pack`
(`dir`, `run_id`, `metrics`, `meta`, `predictions`, `by_qid`).

Missing optional layers load as empty lists (a `session_summaries` run has
no teacher calls).

## Attribution helpers

```python
audit = load_sandwich_audit("experiments/smoke_fused_teachers")
openai_calls = audit.teacher_calls_for("openai")
kept = audit.fusion_kept_for(sample_id="conv-26")
```

Index keys: `sample_id`, `session_id`, `teacher_id`, `question_id`.

## Branch split (to avoid merge collisions)

| Branch work | Touch |
|-------------|--------|
| Pipeline / teachers / dumps | `run.py`, `teachers.py`, `experiments/audit_writer.py` |
| Evaluation / claims on dumps | `scripts/analysis/`, new eval modules; `audit_loader.py` only if the **on-disk contract** changes |
| Path names / version | `experiments/audit_layout.py` + this doc (rare; bump `audit_pack.v1`) |
