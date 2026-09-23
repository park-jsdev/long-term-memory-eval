# OpenAI agent reader/writer campaign

Thinking is off for all cells. Run the four matrices separately:

| YAML | QA cells | Claim |
|---|---:|---|
| `openai_agent_readers_gcs.yaml` | 21 | fixed/index memory × OpenAI readers |
| `openai_agent_writers_gcs.yaml` | 6 | OpenAI single teachers × frozen mini reader |
| `openai_codex_readers_gcs.yaml` | 3 | workspace-only Codex answer harness |
| `openai_codex_writers_gcs.yaml` | 9 | Codex summary/fact/graph writers × frozen mini reader |
| `openai_codex_end_to_end_gcs.yaml` | 3 | persistent Codex reader+writer workspace (sessions visible) |
| `openai_codex_persist_memory_gcs.yaml` | 3 | persist-off / persist-on / notes_only memory-method test |

Focused persist-as-memory operator copy-paste: `docs/agent/RUNBOOK_OPENAI_CODEX_PERSIST.md`.

`openai_codex_*` require the `codex-api-key` Secret Manager secret. Native
Codex tool runs are audit-only until their comparison status is `comparable`.

For every YAML: write its manifest, set `$env:EXPERIMENT_YAML`, deploy, run QA
with the matrix cell count, wait for all QA `_SUCCESS`, then run autorater with
the same count. Aggregate and collect-full always use one task. The job
definition is snapshotted on execution, so deployment of another YAML does not
change a running execution.

Pull aggregate directories before:

```powershell
python -m src.memorybench report configs/analysis/campaign_openai_codex_poc.yaml
python -m src.memorybench report configs/analysis/campaign_openai_agents.yaml
python -m src.memorybench report configs/analysis/campaign_openai_codex_persist_memory.yaml
```

Reports include tool-audit bars (`n_web_search`, `n_mcp` should be ~0),
per-question failure-mode counts, `harness_failed_rate`, Mem0 Table 2 /
local-clone J pins, and generation (model year) lines. Codex CLI is
a 2026 harness around 2024/2025/2026 backbones — do not read Codex-mini as
paper 2024. Native Codex cells stay audit-only unless `comparison_status` is
`comparable`. A 1–2% workspace-read miss is not a cell-level harness failure.
Chat Completions sandwich writers (`openai_agent_writers`) sit
beside Codex writers on overlapping methods only.
