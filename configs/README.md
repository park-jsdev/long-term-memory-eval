# Configs

Interchangeable pieces. Compose with `includes:` (deep-merge; later keys win). Not Hydra.

```text
configs/
  data/          dataset pins (locomo10, processed export)
  layouts/       answer prompt + Chat Completions message shape
  readers/       answer LLM request controls
  agents/        coding-agent harness (adapters / persist / tools / comparison profiles)
  writers/       memory methods and the one write-path model
  autoraters/    LLM-as-a-Judge (separate from QA)
  stacks/        base stack: dataset pin, layout, reader, output directory
  run/           output_dir
  presets/       one-axis overlays (Mem0-parity reader, Luna reader, Codex GPT-5, …)
  models/        generation catalog (snapshots), list prices, context windows
  experiments/   experiment-runner matrices
  analysis/      design over finished packs (tables/plots); report code in scripts/analysis/
```

A **writer** file is runnable: it includes `stacks/qa_default.yaml` (frozen gpt-4o-mini + `qa_mem0_v1`). Swap one piece by including another file after it:

```yaml
includes:
  - configs/writers/session_summaries.yaml
  - configs/readers/gpt-5.6-luna.yaml
```

CLI default: `configs/presets/mem0_baseline.yaml` (session_summaries + Mem0-parity reader/layout).

Agent comparison profiles live at `configs/agents/comparison/`. `strict_mock.yaml`
is the reference hard-budget smoke profile; `audit_codex.yaml` records Codex
controls but intentionally cannot certify a strict cross-agent comparison.

Prompt files live under `prompts/{readers,writers,agents,autoraters}/` with the same roles. After a run, `experiments/<run_id>/TRACE.md` lists the include chain and which jsonl each prompt filled. See `prompts/README.md`.
