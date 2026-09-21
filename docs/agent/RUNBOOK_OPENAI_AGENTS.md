# OpenAI agent reader/writer campaign

Thinking is off for all cells. Run the four matrices separately:

| YAML | QA cells | Claim |
|---|---:|---|
| `openai_agent_readers_gcs.yaml` | 21 | fixed/index memory × OpenAI readers |
| `openai_agent_writers_gcs.yaml` | 6 | OpenAI single teachers × frozen mini reader |
| `openai_codex_readers_gcs.yaml` | 3 | workspace-only Codex answer harness |
| `openai_codex_writers_gcs.yaml` | 9 | Codex summary/fact/graph writers × frozen mini reader |
| `openai_codex_end_to_end_gcs.yaml` | 3 | persistent Codex reader+writer workspace |

`openai_codex_*` require the `codex-api-key` Secret Manager secret. Native
Codex tool runs are audit-only until their comparison status is `comparable`.

For every YAML: write its manifest, set `$env:EXPERIMENT_YAML`, deploy, run QA
with the matrix cell count, wait for all QA `_SUCCESS`, then run autorater with
the same count. Aggregate and collect-full always use one task. The job
definition is snapshotted on execution, so deployment of another YAML does not
change a running execution.

Pull aggregate directories before:

```powershell
python -m src.memorybench report configs/analysis/campaign_openai_agents.yaml
```
