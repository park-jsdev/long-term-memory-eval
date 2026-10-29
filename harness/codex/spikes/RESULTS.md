# Hello-world results

Observed on `codex-cli 0.155.1` (`harness/codex/VERSION`). Model:
`gpt-4o-mini`. Raw logs for this record are gitignored under
`harness/codex/spikes/runs/20261018T214359Z/`.

These rows are smoke checks: hook log plus an echoed token on a one-line
prompt. They do not show that a realistic task used the injected text.
Cell a and cell b differ by more than `--ignore-user-config`.

Implementation: `harness/codex/plugin/hooks/marker_hook.py`,
`harness/codex/plugin/hooks/hooks.json`,
`harness/codex/plugin/plugin.json` (`extensions.com.openai.hooks`),
`configs/agents/hooks/hello_world.yaml`,
`harness/codex/spikes/run_codex_hello_world.py`.

| Spike | Cell | Verdict | What the model said |
| --- | --- | --- | --- |
| 0 pin | version and features | pass | `hooks` stable on. `plugin_hooks` removed. `plugins` and `memories` stable on. |
| 1 | `a_current_adapter` | absent | `MISSING`. The hook log is empty. |
| 1 | `b_project_hooks` | pass | `SEEN HARNESS_MARKER_ALPHA` |
| 1 | `c_project_hooks_no_ephemeral` | pass | `SEEN HARNESS_MARKER_ALPHA` |
| 1 | `d_plugin` | pass | `SEEN HARNESS_MARKER_ALPHA` |
| 2 | abstain on cell b | pass | `MISSING`. Log action `abstain`. |
| 2 | inject `HARNESS_MARKER_BETA` on cell b | pass | `SEEN HARNESS_MARKER_BETA` |
| 3 | Stop capture on cell b | pass | Log action `capture` and a capture line. |

Cell a is the flag set `CodexAgentRunner` uses today (`--ignore-user-config`,
`--ignore-rules`, `--ephemeral`). The hook file was on disk and the model
still did not see the marker. Cells b and c drop `--ignore-user-config`,
trust the workspace, and pass `--dangerously-bypass-hook-trust`. Cell c
also drops `--ephemeral`. Both echoed the marker. In this record `--ephemeral` did not stop the
hook. Codex's own memories pipeline was not measured.

Cell d installs `harness-marker@harness-spikes` into a disposable
`CODEX_HOME` and does not write a project `hooks.json`. The plugin hook
injects the same token. `codex plugin marketplace add` and
`codex plugin add` both exited 0.

On each smoke exec that loaded the hook, `Stop` appended one capture line
and the exec ended. That is not an exactly-once check for multi-step
execs. Abstain returned `{}`, and the model answered `MISSING` for a
token that was not in the prompt.

Workspaces were created outside this git repository. Codex binds a project
`.codex/` directory to the git root, so a hook folder inside the repo would
not be the spike workspace. Trust keys on Windows are the resolved path,
lowercased, with backslashes.

`codex exec` prints `Reading additional input from stdin...` when stdin is
not a terminal. The prompt argument is still the task. The line is not a
failed hook.
