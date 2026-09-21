# Agent-harness runtime (`agent_trajectory.v1`)

On-disk contract for **agent-level** LoCoMo eval: the coding-agent harness
retrieves from conversation files instead of a one-shot `{memory}` fill.

This is a different sandwich bottom than `readers.Reader`. Model-only
comparison remains `full_context` + Chat Completions. Do not mix a reader
prompt swap into a harness claim.

Schema id: **`agent_trajectory.v1`** (`agent/metrics.json` → `schema_version`
on each `agent/trajectory.jsonl` row).

## Why

Recent agent-memory benchmarks (AMA-Bench, MemoryArena, WorldMemArena,
LongMemEval-V2 AgentRunbook-C) evaluate *harness + model*, not stuffed
context. The split that matters:

| Failure | Meaning |
|---------|---------|
| `retrieval_failure` | required gold `dia_id`s never appeared in retrieved text |
| `reasoning_failure` | evidence was retrieved and the answer is still wrong |
| `parametric_success` | answer is correct but evidence was never retrieved |
| `none` | correct, with evidence in hand (or no gold evidence ids) |

Catalog (`INDEX.md` / `ls`) is navigation, not gold evidence. Session-file
reads are `retrieve`. Hosted `web_search` and `mcp` are outside-workspace
kinds: they never count toward recall. Codex argv sets
`web_search="disabled"` and `--ignore-user-config`; `agent/metrics.json`
still records `n_web_search_sum`, `n_mcp_sum`, and
`used_non_workspace_tools_rate` so a leak is visible.

## Workspace (`workspace_files`)

```text
experiments/<run_id>/agent/workspaces/<sample_id>/
  INDEX.md
  sessions/session_N.md
  memory/notes.md          # only when agent.persist_memory is on
```

Gold answers never enter these files. Each turn keeps `(dia_id)` so
trajectory scoring can join LoCoMo `Question.evidence`.

`Memory.text` for this builder is a **manifest pointer**, not the transcript.
Stuffed full-context lives in the `full_context` condition.

## Audit pack extras

```text
experiments/<run_id>/agent/
  traces.jsonl         one harness invocation per question
  trajectory.jsonl     normalized retrieval events + recall/precision
  events.jsonl         raw adapter events (Codex JSONL items)
  metrics.json         run-level means + failure_mode counts
  workspaces/          conversation files the harness saw
```

Existing `predictions.jsonl` / autorater / cost / context-window analyses
still apply. Harness billed tokens are stored as `usage.prompt_tokens` /
`completion_tokens` on the prediction row (same fields campaign plots call
`agent_input_tokens` — that name already meant “answer-path tokens”).

## Config

`configs/agents/` composes like other roles:

| Piece | Role |
|-------|------|
| `harness_minimal.yaml` | persist off, native tools, ephemeral |
| `adapters/{mock,codex,claude_code,opencode,pi}.yaml` | which harness |
| `persist/{on,off}.yaml` | durable harness memory |
| `tools/{native,controlled}.yaml` | native file tools vs later allowlist |

Runnable: `configs/writers/workspace_files.yaml` (mock) and
`configs/presets/agent_codex_gpt5.yaml` (live Codex + GPT-5, model knobs
minimal).

## Adapters

| id | Status |
|----|--------|
| `mock` | implemented (no API) |
| `codex` | implemented (`codex exec --json`) |
| `claude_code` / `opencode` / `pi` | stub configs; factory raises `NotImplementedError` |

Codex isolation: `--cd` to the conversation workspace (absolute) and an
absolute last-message file (`agent_answer.json`, no leading dot). Codex output
schema mode is not used because it can suppress intermediate workspace tool
calls; the task prompt and final-message parser retain the one-key JSON answer
contract. A repo-relative `experiments\...` path is resolved from the workspace
cwd and 404s on Windows. `--ephemeral` when persist is off,
`--ignore-user-config` always (skip user MCP), `-c web_search="disabled"`,
`--skip-git-repo-check`, read-only sandbox unless persist writes.
Auth: `CODEX_API_KEY`, else `OPENAI_API_KEY` from `.env` (ChatGPT CLI login is
a fallback; stale tokens 401).

## Comparison contract

`agent_comparison.v1` is stored in `run_meta.json` and summarized at
`agent/COMPARISON.md`. It pins the requested backbone, task-prompt hash,
workspace hash/context ceiling, retrieval settings, memory-write settings,
judge, and tool budget. Each value is either enforced, observed, or explicitly
`not_applicable`; `null` is never evidence that two conditions matched.

Only an adapter that declares it can enforce every requested hard tool limit
may write `status: comparable`. Native Codex file tooling is presently
audit-only (`incomparable`) because its CLI has no portable per-tool-call or
retrieved-token cap. A run with no successful workspace read/search is
`harness_failed`, not a retrieval failure.
