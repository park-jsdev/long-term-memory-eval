# Experiment collect `locomo-openai-mini-codex-readers-analysis-v2` (full)

On-demand full collect: complete run packs including `{memory}` dumps and traces.

- expected: 6
- QA `_SUCCESS`: 6
- autorater `_SUCCESS`: 6
- failed / incomplete: 0
- not started: 0
- collected_at: `2026-10-05T15:09:14.191790+00:00`

## Notebooks

- `runs.parquet` — one row per completed QA cell
- `examples.parquet` — one row per question (judge columns if autorater ran)
- `cells.jsonl` / `status.json` — every matrix row, including missing cells

## Cells

| index | run_id | memory | reader | stage | catalog |
|------:|--------|--------|--------|-------|---------|
| 0 | `locomo-openai-mini-codex-readers-analysis-v2-0830e2f8` | `full_context` | `gpt-4o-mini` | autorater_completed | `runs/locomo-openai-mini-codex-readers-analysis-v2-0830e2f8` |
| 1 | `locomo-openai-mini-codex-readers-analysis-v2-4deb78f9` | `full_context` | `gpt-4o-mini` | autorater_completed | `runs/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9` |
| 2 | `locomo-openai-mini-codex-readers-analysis-v2-47639359` | `full_context` | `gpt-4o-mini` | autorater_completed | `runs/locomo-openai-mini-codex-readers-analysis-v2-47639359` |
| 3 | `locomo-openai-mini-codex-readers-analysis-v2-56211064` | `session_summaries` | `gpt-4o-mini` | autorater_completed | `runs/locomo-openai-mini-codex-readers-analysis-v2-56211064` |
| 4 | `locomo-openai-mini-codex-readers-analysis-v2-e358bc4d` | `session_summaries` | `gpt-4o-mini` | autorater_completed | `runs/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d` |
| 5 | `locomo-openai-mini-codex-readers-analysis-v2-44872da1` | `session_summaries` | `gpt-4o-mini` | autorater_completed | `runs/locomo-openai-mini-codex-readers-analysis-v2-44872da1` |

Per-cell `runs/<run_id>/` is the complete audit pack (`memory/`, `reader/`, predictions, plots, autorater). Same hashed `run_id` as QA and autorater.

GCS prefix: `experiments/locomo-openai-mini-codex-readers-analysis-v2/collected/`

