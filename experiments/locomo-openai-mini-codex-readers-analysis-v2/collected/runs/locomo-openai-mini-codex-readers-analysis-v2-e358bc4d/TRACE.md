# Trace `locomo-openai-mini-codex-readers-analysis-v2-e358bc4d`

Walk **config YAML → prompt txt → jsonl**. Snapshot copies live under `prompts/`.

## 1. Configs

1. `configs/data/locomo10.yaml`
2. `configs/layouts/qa_mem0_v1.yaml`
3. `configs/readers/gpt-4o-mini.yaml`
4. `configs/run/local_experiments.yaml`
5. `configs/stacks/qa_default.yaml`
6. `configs/writers/session_summaries.yaml` (CLI `--config`)

Resolved merge (later keys win): `config.resolved.yaml`
Entry file copy: `config.source.yaml`

## 2. Prompts

| Role | Config key | Source | Snapshot | Version | JSONL |
|------|------------|--------|----------|---------|-------|
| reader | `pipeline.prompt_path` | `prompts/readers/qa_mem0_v1.txt` | `prompts/readers/qa_mem0_v1.txt` | `qa_mem0_v1` | `predictions.jsonl`, `reader/traces.jsonl`, `reader/predictions.jsonl` |
| autorater | `autorater.prompt_path` | `prompts/autoraters/autorater_mem0_v1.txt` | `prompts/autoraters/autorater_mem0_v1.txt` | `autorater_mem0_v1` | `autorater/traces.jsonl`, `autorater/autorater_verdicts.jsonl` |

## 3. Outputs in this directory

| File | What it is |
|------|------------|
| `predictions.jsonl` | One QA row: `{memory}` text, pred, gold (scorer-only) |
| `reader/traces.jsonl` | Answer LLM call (filled reader prompt → completion) |
| `reader/predictions.jsonl` | Same QA rows as the run-root copy |
| `memory/` | `{memory}` payload actually inserted into the reader prompt |
| `memory/writer/calls.jsonl` | Write-path calls (if a writer ran) |
| `autorater/traces.jsonl` | Judge LLM (separate job; gold is visible here) |
| `run_meta.json` | Pins: models, `prompt_path`, hashes |
| `SUMMARY.md` | Claim audit: sandwich pins, cost, how to follow one question |
| `ATTRIBUTION.md` | LLM call → role → claims made |

