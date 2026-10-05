# Layout

LoCoMo eval, two answer paths, one writer model.

- **Model-only.** Chat Completions. The conversation or a memory string is stuffed into the prompt (`full_context`, `session_summaries`, `raw_chunks`, and the optional Mem0-paper writers).
- **Codex agent.** One `codex exec` per question. Turns live in workspace files. Trajectories land under `agent/`.
- **Evals.** LoCoMo F1 is string overlap. Mem0 J is a later autorater job. Gold answers stay out of reader and writer prompts.
- **One writer model.** `ModelOrchestrator` calls one interchangeable model for `graph` and for `session_summaries` when `writer.model` is set. Swap the model with `writer.model` or `--writer-model`. `mem0` and `mem0g` remain writer configs. They are not in the current model-only versus Codex matrices.
- **GCP.** `infra/gcp/` and `docs/gcp.md`. Cloud ids come from the environment, not from tracked configs.

## Configs

| Path | What it holds |
|------|----------------|
| `configs/models/` | Catalog ids, list prices, published context windows |
| `configs/readers/` | Answer-model request controls |
| `configs/writers/` | One memory method per file |
| `configs/agents/` | Codex adapter, persist, tools |
| `configs/writers/openai_mini.yaml` | The one writer-model slot |
| `configs/experiments/` | Design matrices (one run spec = one Cloud Run task) |
| `configs/analysis/` | Campaign tables and plots |
| `prompts/` | Prompt text, mirrored by role |

## One run

`experiments/<run_id>/` is the only place a run's models, memory, and trajectories meet. Run packs are gitignored. The exception is the published reader comparison: `experiments/locomo-openai-mini-codex-readers-analysis-v2/`, read by `notebooks/17_openai_mini_codex_readers_analysis.ipynb`.

| Directory | Contents |
|-----------|----------|
| `reader/` | Answer-model traces and the prediction copy |
| `memory/` | `{memory}` text, lineage, retrieve ranks |
| `memory/writer/` | Writer-model calls and session text |
| `memory/graph/` | Graph ingest when a graph was built |
| `agent/` | Codex trajectories, workspaces, `metrics.json`, `COMPARISON.md` |
| `autorater/` | Judge verdicts, traces, tables, plots |
| (root) | `predictions.jsonl`, `metrics.json`, `cost.json`, `run_meta.json`, `TRACE.md` |

Model ids live in `configs/models/` and are copied into `run_meta.json`. Trajectories stay under `agent/`. Do not drop either next to the dataset.
