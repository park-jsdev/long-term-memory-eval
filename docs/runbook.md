# Installing, building, and running

Local steps for the evaluation harness. Campaign reproduction is in [REPRODUCE.md](REPRODUCE.md). Cloud Run is in [gcp.md](gcp.md).

Never commit `.env` or an API key.

## Install

```bash
conda create -n distillation python=3.11 -y
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py
```

`scripts/fetch_locomo.py` writes `data/raw/locomo10.json`. That file is gitignored. The pin in `configs/data/locomo10.yaml` is commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.

For a live model, copy `.env.example` to `.env` and set the keys you need: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`. On Windows: `copy .env.example .env`. The loader reads `.env` at run start. A mock run does not need a key.

## Build

The image runs the experiment runner. It also installs the Codex CLI, which a live agent cell needs.

```bash
make image
```

That is `docker build -t memorybench .`. The image tag stays `memorybench` so an existing registry deploy keeps the same name. The process entrypoint is `python -m src.experiment_runner`.

A laptop does not need the image. Use the conda environment above.

## Run

One cell, no API key. The reader is a mock. Five questions, round-robin across conversations:

```bash
python -m src.locomo_eval.run \
  --config configs/presets/mem0_baseline.yaml \
  --reader mock \
  --max-questions 5 \
  --run-id smoke_mock
```

The pack is `experiments/smoke_mock/`. Predictions, metrics, traces, and the memory text are in that directory.

The same command with the mock flag omitted calls the model named in the config. That spends API credits.

```bash
python -m src.locomo_eval.run \
  --config configs/presets/mem0_baseline.yaml \
  --max-questions 3 \
  --run-id smoke_openai
```

A Codex smoke uses the agent preset. Mock mode does not need the `codex` binary. A live run does, plus `CODEX_API_KEY` or a Codex CLI login.

```bash
python -m src.locomo_eval.run \
  --config configs/presets/agent_codex_gpt5.yaml \
  --reader mock \
  --max-questions 3 \
  --run-id smoke_agent_codex
```

### A matrix of cells

The experiment runner expands one YAML into hashed run ids and runs one cell per task. Questions inside a cell are still answered one at a time.

```bash
python -m src.experiment_runner write-manifest configs/experiments/poc.yaml
python -m src.experiment_runner execute-qa configs/experiments/poc.yaml --run-index 0
python -m src.experiment_runner execute-autorater configs/experiments/poc.yaml --run-index 0
python -m src.experiment_runner aggregate configs/experiments/poc.yaml
python -m src.experiment_runner status configs/experiments/poc.yaml
```

`execute-qa` skips a cell that already has `_SUCCESS`. Pass `--force` to regenerate it. `make poc` is the manifest step plus cell 0.

`make test` runs the experiment-runner unit tests. The full suite is mock-only:

```bash
python -m pytest tests/ -q
```

## Cloud

Set `EXPERIMENT_RUNNER_BUCKET` (or `MEMORYBENCH_BUCKET`) in the environment. Do not put a project id or bucket name in a tracked file. Job names and the image tag are documented in [gcp.md](gcp.md).
