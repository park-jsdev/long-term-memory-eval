# Glossary

Names for researchers and engineers new to this repository. Diagrams are in
[architecture.md](architecture.md). The path of one question, from the
released JSON to a score, is in [loop.md](loop.md).

## Contents

- [Configuration composition](#configuration-composition)
- [Evaluation flow](#evaluation-flow)
- [The comparison](#the-comparison)
  - [Controlled comparison](#controlled-comparison)
  - [Frozen reader](#frozen-reader)
  - [Controlled-comparison contract](#controlled-comparison-contract)
  - [Factor](#factor)
  - [Condition](#condition)
  - [Arm](#arm)
  - [Design matrix](#design-matrix)
  - [Design point](#design-point)
  - [Replicate](#replicate)
  - [Experiment](#experiment)
  - [Run spec](#run-spec)
  - [Experiment pack](#experiment-pack)
  - [Design](#design)
- [The measurement](#the-measurement)
  - [Fixed inputs](#fixed-inputs)
  - [Dataset](#dataset)
  - [Memory system](#memory-system)
  - [Memory representation](#memory-representation)
  - [Memory stage](#memory-stage)
  - [Fixed evaluation](#fixed-evaluation)
  - [Reader](#reader)
  - [Writer](#writer)
  - [Agent](#agent)
  - [Answerer](#answerer)
  - [Gold](#gold)
- [The repository](#the-repository)
  - [Configuration](#configuration)
  - [Core](#core)
  - [Evaluation harness](#evaluation-harness)
  - [Agent harness](#agent-harness)
  - [Experiment runner](#experiment-runner)
- [Older notes](#older-notes)

## Configuration composition

Before a run, YAML files are merged into one resolved configuration. Files
named under `includes:` are merged first. The file you pass on the command
line is merged last. When two files set the same field, the later file wins.
A list in a later file replaces the earlier list.

The base stack is the bundle merged first. It pins the dataset and the
reader together, so it is not one stage of the evaluation flow below.

| Order | File | What it contributes |
|---|---|---|
| 1, merged first | `configs/data/`, `configs/layouts/`, `configs/readers/`, `configs/run/` | Dataset pin, answer prompt, reader request, output directory |
| 2 | `configs/stacks/` | The base stack: those pieces as one bundle |
| 3 | `configs/writers/(method).yaml` | The memory system, and one writer model when the method needs one |
| 4, wins on a shared field | A preset, or the experiment file | One replacement, such as the reader model or the storage backend |

## Evaluation flow

This is the path of one question. It is not the merge order above.

| Stage | What is held fixed or varied | What moves |
|---|---|---|
| Fixed inputs | The LoCoMo release used by the run | Conversation text and the question. Gold answers and evidence ids stay with the scorers. |
| Experimental variable | The memory system under test | One memory representation passed to the answerer, or a workspace the agent reads. |
| Fixed evaluation | The reader and the scorers, when the comparison requires it | A predicted answer, then LoCoMo F1. The Mem0 judge is a separate later job on that same string. |

## The comparison

### Controlled comparison

A comparison that holds the dataset and the evaluation fixed and changes
one factor. When that factor is the memory system, the reader stays frozen.
When the factor is the reader model, the comparison is a reader sweep, and
the reader is no longer frozen.

### Frozen reader

The answer model and the answer prompt, held constant while the memory
system changes. Two packs that share this reader and differ only in the
memory system can be compared. Some experiment files label this constraint
`sandwich`. In this documentation the name is frozen reader.

### Controlled-comparison contract

The rules that make those packs comparable. One configuration writes one
pack. A rerun replaces that pack rather than appending to it. Gold answers
never enter a prompt. Across arms of a frozen-reader comparison, the reader
and the prompt match.

### Factor

One axis of a design matrix: memory system, reader, writer, seed, persist,
or session mode. A frozen-reader experiment has one reader assignment. It
does not have a reader factor with several levels.

### Condition

The scientific contrast inside a comparison: one memory system, or one
persist setting, under a stated evaluation.

### Arm

One condition inside a comparison.

### Design matrix

The Cartesian product of factors in one experiment file.

### Design point

One assigned setting in that matrix, before it is given a run id. One
memory system, one reader, one seed.

### Replicate

The same condition and models, run again with a different seed. Report the
seed. Seeds stay separate run ids.

### Experiment

One design matrix and the run specs expanded from it.

### Run spec

One executable assignment: the condition, the model ids, the prompt, and
the seed. The experiment runner launches one run spec per task. Questions
inside a run spec stay in file order. The run id is a hash of those QA
fields. Changing the judge model does not change the run id.

### Experiment pack

The directory written by one run spec: predictions, prompts, memory
artifacts, traces, cost, and the git hash. Reports read packs. They do not
call the answerer again. An aggregate is a table collected from many packs.

### Design

The analysis plan over finished experiments: which packs, which groupings,
which metrics. Analysis files still use the YAML key `campaign:`.

## The measurement

### Fixed inputs

The pinned LoCoMo release, plus any subset cap declared on the run. A subset
cap is a smoke limit. It is not a benchmark result.

### Dataset

The LoCoMo JSON, or a table of stored predictions. A dataset is not an
experiment pack. The pack is the record of one run over that dataset.

### Memory system

The method under test, such as session summaries, a graph, retrieval, or
workspace files. In the dataflow it is the experimental variable. In the
configuration it is the field set by the memory-method file after the base
stack has been merged.

### Memory representation

What the answerer can use in place of the raw dialogue.

| System | What the answerer receives |
|---|---|
| Raw dialogue or full context | Dialogue text in the reader request |
| Session summaries | Summaries from the dataset, or summaries written by one model |
| Graph | A fixed graph schema rendered into the reader request |
| Retrieval (`rag`, `mem0`, `mem0g`, `openai_memory`) | Text retrieved from a prebuilt index |
| Workspace files | Files on disk. The request points at the workspace. |

### Memory stage

The step that builds that representation, between the dataset and the
answerer. Some systems build it once per conversation. Retrieval with a
cutoff builds it once per question. Codex reads workspace files during its
own loop rather than receiving one stuffed string.

### Fixed evaluation

The reader and the scorers, when they are not the factor. String scores are
exact match, token F1, and LoCoMo F1. The Mem0 judge is a separate protocol
on the same predicted string. LoCoMo category tables keep category 5
(adversarial). The Mem0 judge skips it.

### Reader

The Chat Completions answer path. One completion per question. No tool loop.

### Writer

The single model that writes summaries or a graph before the reader
answers. A writer is not the reader. When Codex writes memory, the answer
path is still the frozen Chat Completions reader.

### Agent

The Codex process that answers from workspace files. One process per
question. The model and tool loop inside that process is described in
[loop.md](loop.md).

### Answerer

Whoever produces the predicted answer: the reader, or Codex. The answerer
sees the memory representation and the question. It does not see the gold
answer or the evidence ids.

### Gold

The released answer and the evidence turn ids. Scorers may see them.
Readers, writers, and workspaces may not. If either string appears in a
prompt or a workspace file, the audit fails.

## The repository

### Configuration

A YAML file that names models, methods, and report settings. It carries no
scoring logic. Files live under `configs/`.

### Core

The shared measurement code: memory builders, readers, scorers, and offline
reports.

### Evaluation harness

The code that loads LoCoMo, builds a memory representation, calls the
answerer, writes predictions, and computes string scores. The Mem0 judge
is a later job, not part of the answer call.

### Agent harness

The adapter, workspace, and trajectory record around Codex. This is one
answer path inside the evaluation harness. It is not a name for the
repository.

### Experiment runner

The scheduler. It expands a design matrix into run specs and launches one
run spec per task. It does not answer or score questions.

This repository is a research pipeline: the harness measures, and the
runner schedules.

## Older notes

Bowman et al. 2022 use "sandwich" for a scalable-oversight protocol. This
repository does not use that word for the frozen reader. A few filenames
and the YAML key `campaign:` still carry an older name. The meaning is the
term in this glossary.

| If an older note says | It means |
|---|---|
| sandwich, sandwich contract | frozen reader; controlled-comparison contract |
| memory middle | memory system, memory representation, or memory stage |
| fixed top, variable middle, fixed bottom | fixed inputs, experimental variable, fixed evaluation |
| frozen bottom | base stack (merged first) |
| recipe | configuration |
| engine | core |
| campaign | design, design matrix, or design point |
| cell | condition, design point, or run spec |
