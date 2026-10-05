# Experiment collect `locomo-openai-mini-codex-readers-analysis-v2` (catalog)

Default third wave: parquet + thin audit catalog. Trigger `collect-full` for complete `memory/` dumps.

- expected: 6
- QA `_SUCCESS`: 6
- autorater `_SUCCESS`: 6
- failed / incomplete: 0
- not started: 0
- collected_at: `2026-10-05T14:38:45.036870+00:00`

## Notebooks

- `runs.parquet` — one row per completed QA cell
- `examples.parquet` — one row per question (judge columns if autorater ran)
- `cells.jsonl` / `status.json` — every matrix row, including missing cells

## Cells

| index | run_id | memory | reader | stage | catalog |
|------:|--------|--------|--------|-------|---------|
| 0 | `locomo-openai-mini-codex-readers-analysis-v2-0830e2f8` | `full_context` | `gpt-4o-mini` | autorater_completed | `by_run/locomo-openai-mini-codex-readers-analysis-v2-0830e2f8` |
| 1 | `locomo-openai-mini-codex-readers-analysis-v2-4deb78f9` | `full_context` | `gpt-4o-mini` | autorater_completed | `by_run/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9` |
| 2 | `locomo-openai-mini-codex-readers-analysis-v2-47639359` | `full_context` | `gpt-4o-mini` | autorater_completed | `by_run/locomo-openai-mini-codex-readers-analysis-v2-47639359` |
| 3 | `locomo-openai-mini-codex-readers-analysis-v2-56211064` | `session_summaries` | `gpt-4o-mini` | autorater_completed | `by_run/locomo-openai-mini-codex-readers-analysis-v2-56211064` |
| 4 | `locomo-openai-mini-codex-readers-analysis-v2-e358bc4d` | `session_summaries` | `gpt-4o-mini` | autorater_completed | `by_run/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d` |
| 5 | `locomo-openai-mini-codex-readers-analysis-v2-44872da1` | `session_summaries` | `gpt-4o-mini` | autorater_completed | `by_run/locomo-openai-mini-codex-readers-analysis-v2-44872da1` |

Per-cell `by_run/<run_id>/` holds SUMMARY / TRACE / ATTRIBUTION / metrics / cost. Full `{memory}` dumps: run `python -m src.experiment_runner collect-full <config>` (Cloud Run job `memorybench-collect-full`).

GCS prefix: `experiments/locomo-openai-mini-codex-readers-analysis-v2/aggregate/`

