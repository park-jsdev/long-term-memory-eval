# GPT-4o-mini Codex reader/writer PoC

Two independent three-cell campaigns:

- `openai_codex_poc_readers_gcs.yaml`: full-context control, no-memory Codex
  reader, persistent end-to-end Codex workspace.
- `openai_codex_poc_writers_gcs.yaml`: Codex session-summary, fact, and graph
  artifacts read by frozen GPT-4o-mini.

Before deploying, create and grant the `codex-api-key` Secret Manager secret;
the image installs Codex CLI and `deploy_gcp.ps1` injects it as `CODEX_API_KEY`.

Both QA waves may run concurrently after their own deployment snapshot. For
each campaign, run three QA tasks; wait for its three `_SUCCESS` objects; then
run three autorater tasks; then one aggregate task. Run `collect-full` only
after aggregate and after pulling the thin aggregate pack locally.

Acceptance gates:

1. The no-memory Codex trace contains successful workspace reads.
2. The persistent cell writes non-empty `agent/workspaces/*/memory/notes.md`.
3. Summary/fact/graph writer dumps are non-empty and graph parsing is `ok`
   (not `fallback_mock`).
4. QA and autorater each have exactly three `_SUCCESS` markers.
