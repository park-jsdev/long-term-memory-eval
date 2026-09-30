# Experiment matrix YAML (`experiment_matrix.v1`)

Harness input for `python -m src.experiment_runner`. Scientific execution still uses one locomo_eval method YAML per cell (`full_context.yaml`, `mem0.yaml`, …).

## Required blocks

```yaml
experiment:
  name: locomo-poc          # slug prefix of hashed run_id
  type: sandwich|sweep|ablation|calibration
  freeze:
    prompt_path: prompts/readers/qa_mem0_v1.txt
    reader: { catalog: gpt-4o-mini }   # required for sandwich

benchmark:
  name: locomo
  dataset_path: data/raw/locomo10.json

matrix:
  reader: [ { catalog: gpt-4o }, ... ]   # omitted when freeze.reader is set
  memory_method: [full_context, rag, mem0]
  seed: [1]

judge:
  provider: openai
  api_model_id: gpt-4o-mini

storage:
  backend: local|gcs
  path: experiments          # locomo_eval --output-dir (scratch on Cloud Run)
  bucket: # gcs only; env EXPERIMENT_RUNNER_BUCKET (MEMORYBENCH_BUCKET still accepted)
  dataset_object: data/locomo10.json  # gcs blob; upload once
```

## Optional

- `model_catalog`: path to `configs/models/generation_catalog.yaml`
- `subset.max_samples` / `max_questions` / `sample_id` — PoC only; omit for a full LoCoMo cell
- `shared_indexes.mem0.index_run_id` / `rag.index_run_id` — Cloud Run downloads `shared/<index_run_id>/` into the local experiments root before QA
- `execution.reader_provider: mock` — plumbing; still logs the matrix `api_model_id`
- `execution.judge_provider: mock`

## Expansion

Cartesian product over axes in order `reader`, `memory_method`, `writer`, `seed`.  
When `matrix.writer` is set, cells that do not use a live writer (`full_context`, `rag`, `mem0`, …) keep only `writer: none`; `session_summaries` and `graph` keep a writer.  
`run_id = <experiment-slug>-<sha256(identity)[:8]>`. Identity excludes timestamps and judge model.

## On-disk after execute-qa

```text
experiments/<run_id>/          # locomo_eval audit pack + harness files
  predictions.jsonl
  examples.parquet
  summary.parquet
  run.json
  _SUCCESS
  autorater/_SUCCESS           # after execute-autorater
experiments/<experiment_name>/
  manifest/runs.jsonl
  aggregate/                   # after `experiment_runner aggregate` / memorybench-aggregate job
    SUMMARY.md
    status.json
    cells.jsonl
    runs.parquet
    examples.parquet
    by_run/<run_id>/           # copied SUMMARY, TRACE, ATTRIBUTION, metrics, cost
    _SUCCESS
  collected/                   # after `experiment_runner collect-full` (Cloud Run job memorybench-collect-full)
    SUMMARY.md
    runs/<run_id>/             # complete pack including memory/, reader/, plots
    _SUCCESS
experiments/mem0_locomo10/     # shared write-index (not per reader)
experiments/rag_locomo10/
```
