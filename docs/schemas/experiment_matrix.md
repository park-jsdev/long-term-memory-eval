# Experiment matrix YAML (`experiment_matrix.v1`)

Harness input for `python -m src.memorybench`. Scientific execution still uses one locomo_eval method YAML per cell (`full_context.yaml`, `mem0.yaml`, …).

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
  path: experiments          # locomo_eval --output-dir; shared indexes live here
```

## Optional

- `model_catalog`: path to `configs/models/generation_catalog.yaml`
- `subset.max_samples` / `max_questions` / `sample_id` — PoC only; omit for a full LoCoMo cell
- `shared_indexes.mem0.index_run_id` / `rag.index_run_id`
- `execution.reader_provider: mock` — plumbing; still logs the matrix `api_model_id`
- `execution.judge_provider: mock`

## Expansion

Cartesian product over axes in order `reader`, `memory_method`, `writer`, `seed`.  
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
  aggregate/examples.parquet
  aggregate/runs.parquet
experiments/mem0_locomo10/     # shared write-index (not per reader)
experiments/rag_locomo10/
```
