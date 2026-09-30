# Prompts

Same roles as `configs/`. Filename stems stay the prompt **version** (`qa_mem0_v1`, `autorater_mem0_v1`, …).

```text
prompts/
  readers/       answer LLM  ← configs/layouts/*.yaml  pipeline.prompt_path
  agents/        coding-agent harness (workspace files) ← configs/agents/ + layouts/qa_workspace_v1.yaml
                 qa_workspace_v1 persist-off; qa_workspace_persist_v1 append notes;
                 qa_workspace_notes_only_v1 sessions hidden; ingest_notes_v1 write pass
  writers/       session summary, graph, mem0 / mem0g / openai-memory extract
  autoraters/    LLM-as-a-Judge                        ← configs/autoraters/
```

## Manual trace (one run)

1. Open the YAML you passed to `--config` (or a writer + overlay). Follow `includes:` depth-first; later keys win.
2. Each `*_prompt_path` value is a file in this tree. Layouts pin the reader prompt; writers pin session summary, graph, and extract/update; autoraters are a **separate** job.
3. After `python -m src.locomo_eval.run`, open `experiments/<run_id>/TRACE.md`. It lists the include chain, copies the prompt files under `experiments/<run_id>/prompts/`, and points at the jsonl each template filled:
  - reader → `predictions.jsonl`, `reader/traces.jsonl`
  - agents → `agent/traces.jsonl`, `agent/trajectory.jsonl` (harness runs)
  - writer → `memory/writer/calls.jsonl`
   - mem0/openai extract → the **index** dump (`experiments/<index_run_id>/mem0_index/`), not the QA pack
   - autorater → `autorater/traces.jsonl` (written by `run_benchmark`)

Resolved merge of the YAML: `experiments/<run_id>/config.resolved.yaml`.
