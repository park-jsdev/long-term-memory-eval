# Configs

Interchangeable pieces. Compose with `includes:` (deep-merge; later keys win). Not Hydra.

```text
configs/
  data/          dataset pins (locomo10, processed export)
  layouts/       answer prompt + Chat Completions message shape
  readers/       answer LLM request controls
  writers/       memory methods (the sandwich middle)
  teachers/      write-path teacher rosters / request knobs
  autoraters/    LLM-as-a-Judge (separate from QA)
  stacks/        frozen combinations of data+layout+reader+run
  run/           output_dir
  presets/       one-axis overlays (Mem0-parity reader, Luna reader, …)
  models/        generation catalog (snapshots), list prices, context windows
  experiments/   memorybench matrices
  analysis/      campaign vs experiment recipes (tables/plots); engines in scripts/analysis/
```

A **writer** file is runnable: it includes `stacks/qa_default.yaml` (frozen gpt-4o-mini + `qa_mem0_v1`). Swap one piece by including another file after it:

```yaml
includes:
  - configs/writers/session_summaries.yaml
  - configs/readers/gpt-5.6-luna.yaml
```

CLI default: `configs/presets/mem0_baseline.yaml` (session_summaries + Mem0-parity reader/layout).

Prompt files live under `prompts/{readers,writers,teachers,autoraters}/` with the same roles. After a run, `experiments/<run_id>/TRACE.md` lists the include chain and which jsonl each prompt filled. See `prompts/README.md`.
