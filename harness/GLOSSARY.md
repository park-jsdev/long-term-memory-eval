# Harness glossary

Words for the Codex hello world. The experiment-repo glossary is
[docs/glossary.md](../docs/glossary.md). This page is only the agent
program and the knobs around it.

Pinned run: `codex-cli 0.155.1`, written up in
[spikes/RESULTS.md](codex/spikes/RESULTS.md). A newer CLI is unverified
until spike 0 is repeated. Statements are labeled observed (that run),
documented (OpenAI docs or this CLI's help), or assumption (not a result).
Flow and the unrun test list are in [README.md](README.md).

## The stack

```text
this repo (evaluation harness)
  CodexAgentRunner or the hello-world spike
    codex exec          <- the agent harness, one process
      hooks / plugin    <- our code, started by Codex
      model call        <- GPT, inside Codex
```

The model writes the reply. Codex decides when to call the model, which
tools exist, and which of our programs run. This repository starts Codex
and stores the reply. It does not contain the model.

Components, in the order a plugin run uses them. The flow diagrams are in
[README.md](README.md).

| Component | What it is | What we change |
| --- | --- | --- |
| Marketplace | Catalog. Cell d only. | Which folder `codex plugin add` installs. |
| Plugin | Installable package. Root `plugin.json` points `extensions.com.openai.hooks` at `hooks/hooks.json`. | The package contents. The policy is not inside it. |
| `hooks.json` | Which script runs at which event. | Project file (cells b, c) or the copy inside the plugin (cell d). |
| Hook | `marker_hook.py`. | How a policy becomes `additionalContext` or `{}`, plus the log line. |
| Policy | `memory/policy.json` in the workspace. | Inject a fixed block, abstain, or choose per prompt. |
| Exec flags | The `codex exec` command this repo builds. | Cell a skips the hook load. Cells b, c, and d allow it. |
| Codex CLI | Installed binary, pin in `harness/codex/VERSION`. | Left unchanged. |
| GPT | The model named by `--model`. | Which model writes the final message. |

## Pieces

### Model

The language model named with `--model`. The hello world used
`gpt-4o-mini` in the 0.155.1 record. Codex adds its own system prompt and
tool list before the user task. Input size is whatever
`turn.completed.usage.input_tokens` says on that exec. It is not a fixed
wrapper size. Changing the model changes the writer of the reply. It
does not install hooks.

### Harness

The program that runs the agent loop: prompt, model, tools, repeat, until
a final message. Codex is one harness. OpenCode and pi are others. In
this repo the word also shows up as "evaluation harness," which is our
Python pipeline. Those are different programs. See [Evaluation harness](#evaluation-harness).

### Codex

OpenAI's coding-agent CLI (`codex` on `PATH`). Version pin:
`harness/codex/VERSION`. We do not fork it. We start it with `codex exec`
and attach our behavior through hooks or a plugin.

### codex exec

One non-interactive Codex process. It takes a prompt, talks to the model,
may call tools, and exits. The hello world runs one exec per cell. The
LoCoMo adapter runs one exec per question (`CodexAgentRunner`).

### Turn

One user request inside an exec, from `UserPromptSubmit` through as many
model and tool cycles as Codex uses, until the exec finishes. The
hello-world prompt happened to finish in one model call. A longer task
can call the model more than once. `Stop` is the end of that request,
not the end of each model call.

### Final message

The last assistant text. `codex exec -o last_message.txt` writes it. The
spike judges that file. `SEEN HARNESS_MARKER_ALPHA` means the model
echoed our injected token. `MISSING` means it did not.

## Where Codex reads files

### CODEX_HOME

Codex's private directory. The default is `~/.codex`. It holds
`config.toml`, `auth.json`, installed plugins, and Codex's own memories.
Every spike exec sets `CODEX_HOME` to a fresh directory so the user's real
config and hooks are left alone. Auth is an API key when one is set.
Otherwise only `auth.json` is copied. `config.toml` is never copied.

### config.toml

Settings inside `CODEX_HOME`: feature flags, trusted projects, enabled
plugins. `--ignore-user-config` skips this file. Auth still uses
`CODEX_HOME`.

### Project

The directory Codex treats as the work tree. If that directory sits inside
a git repo, the project is the git root, not the subdirectory you passed
to `--cd`. Project files live in `<project>/.codex/`. Spike workspaces are
created under the system temp directory so this repository's git root is
not the project.

### Workspace

The directory passed to `--cd`. The agent’s shell starts there. For a
project hook, the workspace and the project are the same directory only
when that directory is not inside a different git root.

### Trust

A project key in `config.toml`:

```toml
[projects.'c:\path\to\workspace']
trust_level = "trusted"
```

On Windows the key is the resolved path, lowercased, with backslashes.
Until that key says `trusted`, Codex does not load `<project>/.codex/hooks.json`.
The spike writes this table into the disposable `CODEX_HOME` and also
passes the same key as `-c`.

## Extensions

### Hook

A program Codex starts at a named point in the turn. Codex writes one JSON
object to the program's stdin. The program writes one JSON object to
stdout. Logs must not go to stdout. For `UserPromptSubmit`, plain stdout
text is inserted into the model context.

Our hook is `harness/codex/plugin/hooks/marker_hook.py`.

### hooks.json

The list of which program runs at which event. Two locations that both
worked on 0.155.1:

| Location | Cell | When it loads |
| --- | --- | --- |
| `<project>/.codex/hooks.json` | b, c | Project is trusted, and `--ignore-user-config` is off. |
| An installed plugin's `hooks/hooks.json` | d | The plugin is enabled in `CODEX_HOME`. |

### Event

The name of the point in the turn. We use two.

| Event | When | What our script returns |
| --- | --- | --- |
| `UserPromptSubmit` | The user prompt is about to be sent to the model. | Either `additionalContext`, or `{}`. |
| `Stop` | The turn is about to finish. | `{}`, after appending one capture line. |

`decision: "block"` on `Stop` does not cancel the turn. It asks Codex to
keep going, using the reason as a new prompt. The capture hook returns
`{}` so the turn ends.

### additionalContext

Text the hook adds for this turn, inside:

```json
{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "HARNESS_MARKER_ALPHA"}}
```

Codex places that string in the developer context the model reads. The
hello-world prompt tells the model to echo a token only if that context
contains it. The prompt itself does not contain the full token, so an
echo is evidence the hook text arrived.

### Abstain

The hook ran and added nothing. Stdout is `{}`. The log action is
`abstain`. On 0.155.1 the model then answered `MISSING`.

### Policy file

`memory/policy.json` in the workspace. This is our switch, not Codex's.

| File | Hook behavior |
| --- | --- |
| Absent | Inject `HARNESS_MARKER_ALPHA`. Used to prove the hook started. |
| `{"action": "abstain"}` | Return `{}`. |
| `{"action": "inject", "text": "HARNESS_MARKER_BETA"}` | Put that text in `additionalContext`. |

Later personalization arms are meant to differ by this file. The hook
program stays the same.

### Plugin

A directory Codex can install. This experiment's manifest is the root
file `harness/codex/plugin/plugin.json` (`name`, `version`, and
`extensions.com.openai.hooks` pointing at `./hooks/hooks.json`). There is
no `.codex-plugin/plugin.json` in this package. The policy file stays in
the workspace. The hook reads it. Installing the plugin does not edit
Codex's source. On 0.155.1 the feature flag named `plugin_hooks` is
removed. In that record, plugin hooks still loaded with `hooks` on and
the plugin enabled.

### Marketplace

A catalog that points at plugin directories. Ours is
`harness/codex/.agents/plugins/marketplace.json`, name `harness-spikes`.
Cell d runs:

```text
codex plugin marketplace add <staged copy>
codex plugin add harness-marker@harness-spikes
```

The id is `plugin-name@marketplace-name`. Both commands write into the
disposable `CODEX_HOME`, then `config.toml` sets
`plugins."harness-marker@harness-spikes".enabled = true`.

### PLUGIN_ROOT

A token Codex substitutes with the installed plugin directory. The
committed `hooks.json` uses it (`python ${PLUGIN_ROOT}/hooks/marker_hook.py`).
The live spike rewrites a staged copy to an absolute Python path so a
failure means "plugin not loaded," not "python was missing from PATH."

### Hook trust

Codex will not run a newly defined hook until a person reviews it in
`/hooks`. `--dangerously-bypass-hook-trust` skips that review for one
exec. It does not load a hook that was never discovered. Cell a does not
pass this flag. Cells b, c, and d did, in a disposable `CODEX_HOME`, for
`marker_hook.py` only. Do not point that flag at an unreviewed
third-party hook. The hook process can write files when the model's
shell is `read-only`.

## Knobs on codex exec

These are the fields in `configs/agents/hooks/hello_world.yaml`. The spike
turns them into CLI flags. `CodexAgentRunner` does not read that file yet.
LoCoMo runs still use cell a's flags.

| YAML field | CLI | What changes |
| --- | --- | --- |
| `ignore_user_config: true` | `--ignore-user-config` | Skip `CODEX_HOME/config.toml`. On 0.155.1 this also skips the project `.codex/` layer. The hook file can be on disk and the process never starts. Cell a. |
| `ignore_rules: true` | `--ignore-rules` | Skip execpolicy `.rules` files, so a local rule cannot block the run. Every cell leaves this on. |
| `ephemeral: true` | `--ephemeral` | Documented: skip persisting the session rollout. Cells a, b, and d. In the 0.155.1 record the hook still ran with the flag on (cell b) and with it off (cell c). Codex's memories pipeline was not measured. |
| `bypass_hook_trust: true` | `--dangerously-bypass-hook-trust` | Run discovered hooks without the interactive review. |
| `trust_project: true` | `-c projects.'<path>'.trust_level="trusted"` plus the same table in `config.toml` | Make the workspace's `.codex/hooks.json` eligible. |
| `enable_hooks: true` | `--enable hooks` | Turn the hooks feature on for this process. On 0.155.1 it is already stable and on. |
| `hook_source: project` | write `<workspace>/.codex/hooks.json` | Cells a, b, c. |
| `hook_source: plugin` | install the plugin, do not write project `hooks.json` | Cell d. |
| `model` | `--model` | Which GPT Codex calls. |

Other flags the spike always passes, matching the LoCoMo adapter:

| CLI | What it does |
| --- | --- |
| `--json` | Print events as JSON lines. |
| `--skip-git-repo-check` | Allow a workspace that is not a git repo. The temp workspaces are not. |
| `--sandbox read-only` | The model's shell cannot write the workspace. The hook is a separate process and can still write its log. |
| `-c web_search="disabled"` | No hosted web search. |
| `-c sandbox_permissions=["disk-full-read-access"]` | The read-only sandbox may still read the workspace. |

## Words this repo added

### Spike

A small live check with a pass rule, not a LoCoMo score. Spike 0 records
the CLI version and does not call the model. Spikes 1–3 call GPT.

| Spike | Question |
| --- | --- |
| 0 | Which Codex is installed, and are hooks enabled? |
| 1 | Can a hook put one marker into the model context? |
| 2 | Can the policy file inject or abstain? |
| 3 | Does `Stop` append a capture line without continuing the turn? |

### Cell

One row of `hello_world.yaml`: a flag combination and a hook source.
Cells b, c, and d passed on 0.155.1. Cell a did not.

### Marker

A fixed token, `HARNESS_MARKER_ALPHA` or `HARNESS_MARKER_BETA`. It stands
in for a future memory block. The model is asked to echo it only when the
hook added it.

### Verdict

How the spike scores one exec.

| Verdict | Meaning |
| --- | --- |
| `pass` | The hook log and the final message agree with the cell. |
| `absent` | The hook log is empty and the token is absent. Cell a. |
| `hook_ran_context_missing` | The log says `inject` and the model did not echo the token. |
| `contamination` | The token appeared and the hook did not log `inject`. |

### Disposable CODEX_HOME

A new directory for one exec. Codex may write trust and plugin state
there. The user's `~/.codex` is not that directory.

### Adapter

`src/locomo_eval/agents/adapters/codex.py`. It builds the `codex exec`
command for a LoCoMo question. Today that command is cell a. The hello
world does not go through the adapter, because the adapter's flags are
part of what the spike measures.

## Experimental conditions

One exec per condition. The Codex binary and `marker_hook.py` stay fixed.
The condition is the launch flags, the attach path, and `memory/policy.json`.
The same table is in [README.md](README.md).

| Condition | Attach | Policy | 0.155.1 record |
| --- | --- | --- | --- |
| `hook_off` | Cell a. Several flags, including `--ignore-user-config`. | Unused. | `MISSING`. Log empty. Not a single-flag A/B. |
| `marker_project` | Cell b. Project `hooks.json`. | Absent. Hook injects `HARNESS_MARKER_ALPHA`. | Smoke: log `inject`, final message `SEEN`. |
| `marker_plugin` | Cell d. Marketplace, then the plugin. | Same injection. | Smoke: same echo. |
| `abstain` | Cell b in the record. | `{"action": "abstain"}`. Stdout `{}`. | Smoke: log `abstain`, token absent. |
| `inject_fixed` | Cell b or d. | `{"action": "inject", "text": "..."}`. | Smoke echo of `HARNESS_MARKER_BETA`. A realistic block is not run. |
| `inject_selective` | Cell b or d. | A relevance decision per prompt, then inject or abstain. | Not run. Hook actions are not this policy. |
| `capture` | `Stop` on a loaded hook. | Whichever policy the prompt used. | One `captured` line on each smoke exec. Multi-step exactly-once is not run. |

`inject_fixed` is the unfiltered arm: every prompt receives the same block.
`inject_selective` is the application arm: something must decide relevance,
and the score includes injections that should have been withheld and
abstentions that should have injected. `hook_off` is the launcher LoCoMo
runs use today.

Unrun checks, including the single-flag discovery pair and the paired
answer test, are listed in [README.md](README.md).

## Similar names

### Evaluation harness

Our Python pipeline: load data, build memory, call an answerer, score.
It lives under `src/`. It is not Codex.

### Agent harness

The adapter, workspace, and trajectory around one Codex process. One
answer path inside the evaluation harness. Defined in
[docs/glossary.md](../docs/glossary.md).

### Codex memories

Codex's own memory feature. On 0.155.1, `codex features list` showed
`memories` as stable and on. Documented behavior is session extraction
and consolidation under `CODEX_HOME`, skipped for ephemeral sessions.
This experiment did not read or write that store, and did not test the
skip. The hook is a separate program. In the recorded run it still fired
when `--ephemeral` was set.

### Feature `plugin_hooks`

A removed feature flag on 0.155.1 (`removed`, off). The working switch is
`hooks`, which is stable and on. Enabling the old name is not required
for cell d.

### Sandbox

The policy on commands the model runs. `read-only` blocks those writes.
It is not the policy on our hook process. A hook that has been allowed to
run can still append `hook_log.jsonl`.
