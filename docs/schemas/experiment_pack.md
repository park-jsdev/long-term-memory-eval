"""Read-only contract for experiments/<run_id>/ (eval-branch API).

**Writers** (`run.py`, teachers, autorater) produce this layout.
**Evaluation** should import only:

```python
from src.locomo_eval.experiments.load import load_experiment_pack, load_qa_pack
from src.locomo_eval.experiments.layout import PackPaths
```

Do not import `src.locomo_eval.run`, `teachers`, or `experiments.write` from an
eval branch. That keeps git conflicts on the write path vs analysis path small.

Schema id: **`audit_pack.v1`** (`run_meta.json` → `audit_layout.version`)

---

## Modules

```text
experiments/<run_id>/
  run_meta.json                 # pins + audit_layout paths
  metrics.json / predictions.jsonl   # QA slice (compat at root)
  reader/                       # answer LLM
  memory/                       # {memory} payload + optional teachers/ + graph/
  autorater/                    # judge (optional; separate CLI)
```

| Module | Load entry | Typical files |
|--------|------------|----------------|
| QA / reader | `load_qa_pack(run_dir)` | `predictions.jsonl`, `metrics.json`, `run_meta.json` |
| Full pack | `load_experiment_pack(run_dir)` | plus `reader/traces.jsonl`, `memory/teachers/*`, `autorater/*` |
| Paths only | `PackPaths.from_run_dir(run_dir)` | no I/O |

`load_qa_pack` return keys match `scripts.compare_full_runs.load_pack` (`dir`, `run_id`, `metrics`, `meta`, `predictions`, `by_qid`).

Missing optional modules load as empty lists (a `session_summaries` run has no teacher calls).

---

## Attribution helpers

```python
pack = load_experiment_pack("experiments/smoke_fused_teachers")
openai_calls = pack.teacher_calls_for("openai")
kept = pack.fusion_kept_for(sample_id="conv-26")
```

Index keys: `sample_id`, `session_id`, `teacher_id`, `question_id`.

---

## Branch split (to avoid merge collisions)

| Branch work | Touch |
|-------------|--------|
| Pipeline / teachers / dumps | `run.py`, `teachers.py`, `experiments/write.py` |
| Evaluation / claims on dumps | `scripts/analysis/`, new eval modules, `experiments/load.py` only if the **on-disk contract** changes |
| Path names / version | `experiments/layout.py` + this doc (rare; bump `audit_pack.v1`) |
