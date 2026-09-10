# Agent Long-Term Memory Experiment Pipeline

## 1. Objective

Implement a reproducible, cloud-agnostic experiment pipeline for evaluating LLM-based conversational long-term-memory systems.

The initial proof-of-concept deployment target is Google Cloud, using:

* Git repository for source control
* Docker for reproducible execution
* Google Artifact Registry for container images
* Google Cloud Run Jobs for parallel experiment execution
* Google Cloud Storage for experiment artifacts and aggregated results
* Google Secret Manager for LLM API credentials
* Parquet as the canonical structured result format
* Jupyter notebooks for downstream analysis

Model training and GPU workloads are explicitly out of scope.

The system should remain portable so that future execution backends such as PACE/Slurm, Runpod, local Docker, or another batch-compute system can reuse the same experiment container without modifying experiment logic.

---

# 2. Core architectural principle

Separate the system into four layers:

```text
Experiment definition
        |
        v
Experiment runner
        |
        v
Storage abstraction
        |
        v
Execution backend
```

Specifically:

```text
Git repository
      |
      v
Docker image
      |
      +--------------------+
      |                    |
      v                    v
Local Docker        Google Cloud Run Jobs
                           |
                   N parallel tasks
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
          Run 001       Run 002       Run N
             |             |             |
             +-------> Object Storage <--+
                           |
                           v
                    results/*.parquet
                           |
                           v
                    aggregate_results
                           |
                           v
                    results.parquet
                           |
                           v
                  analysis/*.ipynb
```

Experiment code must not contain GCP-specific execution logic.

Cloud-specific behavior belongs under an executor/deployment layer.

---

# 3. Functional requirements

The system must support the following workflow:

```text
1. Define an experiment matrix.
2. Expand the matrix into deterministic individual runs.
3. Assign each run a unique run_id.
4. Execute runs independently.
5. Make LLM API calls.
6. Evaluate responses using benchmark-specific evaluation logic.
7. Record raw outputs, metrics, timing, token usage, and metadata.
8. Upload run artifacts to object storage.
9. Retry failed runs safely.
10. Aggregate completed runs into Parquet datasets.
11. Load the resulting Parquet dataset from a Jupyter notebook.
```

A single experiment run must be independently reproducible using one command.

Example:

```bash
python -m memorybench.run \
    --config configs/experiments/poc.yaml \
    --run-index 17
```

The same command must work:

```text
locally
inside Docker
inside Cloud Run
on a future Slurm/PACE worker
```

---

# 4. Repository structure

Implement approximately this structure:

```text
agent-memory-bench/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .dockerignore
├── .gitignore
├── Makefile
│
├── configs/
│   ├── experiments/
│   │   ├── poc.yaml
│   │   └── locomo.yaml
│   │
│   └── models/
│       ├── openai.yaml
│       ├── anthropic.yaml
│       └── gemini.yaml
│
├── src/
│   └── memorybench/
│       ├── __init__.py
│       │
│       ├── cli.py
│       ├── run.py
│       ├── aggregate.py
│       │
│       ├── config.py
│       ├── manifests.py
│       ├── schemas.py
│       │
│       ├── experiments/
│       │   ├── base.py
│       │   └── matrix.py
│       │
│       ├── benchmarks/
│       │   ├── base.py
│       │   ├── locomo.py
│       │   ├── longmemeval.py
│       │   └── locomo_plus.py
│       │
│       ├── memory/
│       │   ├── base.py
│       │   ├── none.py
│       │   └── mem0.py
│       │
│       ├── llm/
│       │   ├── base.py
│       │   ├── openai.py
│       │   ├── anthropic.py
│       │   └── gemini.py
│       │
│       ├── evaluation/
│       │   ├── base.py
│       │   └── llm_judge.py
│       │
│       ├── storage/
│       │   ├── base.py
│       │   ├── local.py
│       │   └── gcs.py
│       │
│       └── utils/
│           ├── ids.py
│           ├── logging.py
│           └── timing.py
│
├── scripts/
│   ├── generate_manifest.py
│   ├── run_local.sh
│   ├── deploy_gcp.sh
│   ├── execute_gcp.sh
│   └── aggregate.sh
│
├── infra/
│   └── gcp/
│       ├── README.md
│       └── setup.sh
│
├── notebooks/
│   ├── 00_validate_results.ipynb
│   └── 01_results_analysis.ipynb
│
└── tests/
    ├── test_config.py
    ├── test_matrix.py
    ├── test_run_ids.py
    ├── test_storage.py
    └── test_smoke.py
```

Do not create unnecessary framework abstractions beyond these responsibilities.

Prefer straightforward Python classes/functions over dependency-injection frameworks.

---

# 5. Python/runtime requirements

Use:

```text
Python >= 3.11
```

Preferred dependency management:

```text
uv
```

Core libraries should include approximately:

```text
pydantic
pyyaml
pandas
pyarrow
google-cloud-storage
tenacity
jupyter
```

LLM SDK dependencies should only be included for providers actually implemented.

Examples:

```text
openai
anthropic
google-genai
```

Do not introduce Airflow, Kubernetes, Ray, Celery, Spark, or a database for the initial implementation.

They are unnecessary for this scale.

---

# 6. Experiment configuration

Experiments must be defined declaratively in YAML.

Example:

```yaml
experiment:
  name: locomo-poc
  seed: 42

benchmark:
  name: locomo
  dataset_path: data/locomo

matrix:
  agent_model:
    - provider: openai
      model: MODEL_NAME

  judge_model:
    - provider: openai
      model: MODEL_NAME

  memory_method:
    - none
    - mem0

  seeds:
    - 1
    - 2
    - 3

execution:
  max_concurrency: 10
  request_timeout_seconds: 120
  max_api_retries: 5

storage:
  backend: gcs
  bucket: agent-memory-experiments
  prefix: experiments/locomo-poc
```

The exact models must remain configuration values rather than hard-coded constants.

---

# 7. Experiment matrix

Implement a deterministic experiment-matrix expansion function.

For example:

```python
runs = expand_matrix(config)
```

Input:

```text
3 models
2 memory methods
3 seeds
```

Output:

```text
18 independent RunSpec objects
```

Each `RunSpec` must contain all parameters required to execute that run.

Example conceptual schema:

```python
RunSpec(
    experiment_name="locomo-poc",
    benchmark="locomo",
    agent_provider="openai",
    agent_model="...",
    judge_provider="openai",
    judge_model="...",
    memory_method="mem0",
    seed=2,
)
```

Ordering must be deterministic.

Running matrix expansion twice against the same configuration must produce identical run ordering and identical run IDs.

---

# 8. Run IDs

Each run must receive a deterministic `run_id`.

Recommended form:

```text
<experiment>-<short-hash>
```

Example:

```text
locomo-poc-a83f2901
```

The hash should derive from a canonical serialized representation of the scientifically meaningful run configuration.

For example:

```text
benchmark
agent model
judge model
memory method
seed
relevant benchmark parameters
```

Do not derive the run ID from timestamps.

This makes retries naturally idempotent.

---

# 9. Cloud Run task mapping

When running under Google Cloud Run Jobs:

```python
run_index = int(os.environ["CLOUD_RUN_TASK_INDEX"])
```

Then:

```python
runs = expand_matrix(config)
run_spec = runs[run_index]
run(run_spec)
```

When running locally:

```bash
python -m memorybench.run --config configs/experiments/poc.yaml --run-index 3
```

must produce equivalent behavior.

Implement a helper such as:

```python
def resolve_run_index(cli_index: int | None) -> int:
    if cli_index is not None:
        return cli_index

    cloud_index = os.getenv("CLOUD_RUN_TASK_INDEX")
    if cloud_index is not None:
        return int(cloud_index)

    return 0
```

Do not otherwise couple experiment execution to Cloud Run.

---

# 10. Concurrency model

Use:

```text
one Cloud Run task = one experiment run
```

Do not initially introduce multiprocessing inside each container.

Parallelism should occur at the infrastructure level:

```text
task 0 -> run 0
task 1 -> run 1
task 2 -> run 2
...
task N -> run N
```

Cloud Run parallelism should be configurable separately from task count.

Example:

```text
100 runs
20 parallel tasks
```

means Cloud Run processes at most 20 experiment runs simultaneously.

This provides a simple mechanism for respecting LLM API rate limits.

---

# 11. LLM interface

Define a minimal provider-neutral interface.

Conceptually:

```python
class LLMClient(Protocol):

    def generate(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 0,
    ) -> LLMResponse:
        ...
```

`LLMResponse` should expose at least:

```python
text: str

provider: str
model: str

input_tokens: int | None
output_tokens: int | None
cached_input_tokens: int | None

latency_seconds: float

raw_metadata: dict
```

Provider-specific responses must be normalized here.

Experiment and benchmark implementations must not directly call provider SDKs.

---

# 12. LLM retries

Retry transient API failures.

Examples include:

```text
HTTP 429
HTTP 500
HTTP 502
HTTP 503
timeouts
temporary transport errors
```

Use exponential backoff with jitter.

Example policy:

```text
maximum attempts: 5
initial backoff: ~1 second
maximum backoff: ~60 seconds
```

Do not blindly retry authentication errors or malformed requests.

Record retry counts as metrics.

---

# 13. Memory-method interface

Memory implementations should expose a small interface.

Conceptually:

```python
class MemorySystem(Protocol):

    def ingest(self, conversation) -> None:
        ...

    def query(self, query: str) -> str:
        ...

    def reset(self) -> None:
        ...
```

The initial implementations should support:

```text
none
mem0/reproduced method
```

Future memory architectures should be addable without modifying benchmark execution logic.

---

# 14. Benchmark interface

Benchmarks should implement a common interface.

Conceptually:

```python
class Benchmark(Protocol):

    def load(self):
        ...

    def examples(self):
        ...

    def evaluate_example(
        self,
        example,
        *,
        agent,
        memory,
        judge,
    ):
        ...
```

Initially prioritize LoCoMo.

Create stubs/interfaces for:

```text
LongMemEval
LoCoMo Plus
```

but do not implement them unless needed for the PoC.

Avoid premature benchmark-generalization.

---

# 15. Result model

Every experiment should produce structured data at two levels:

```text
example-level results
run-level summary
```

## Example-level schema

At minimum:

```text
run_id
experiment_name

benchmark
example_id

agent_provider
agent_model

judge_provider
judge_model

memory_method
seed

question
reference_answer
generated_answer

judge_score
judge_reasoning

agent_input_tokens
agent_output_tokens

judge_input_tokens
judge_output_tokens

agent_latency_seconds
judge_latency_seconds

retry_count

created_at
```

Add benchmark-specific metric fields when required.

---

# 16. Run-level schema

Each completed run must also produce a summary record containing:

```text
run_id

experiment_name
benchmark

agent_provider
agent_model

judge_provider
judge_model

memory_method
seed

num_examples
num_successful_examples
num_failed_examples

primary_score

agent_input_tokens_total
agent_output_tokens_total

judge_input_tokens_total
judge_output_tokens_total

llm_calls_total

wall_time_seconds

started_at
completed_at

git_commit
container_image
config_hash

status
```

Status should be one of:

```text
running
completed
failed
```

---

# 17. Storage layout

Use a hierarchical object layout.

For GCS:

```text
gs://BUCKET/
└── experiments/
    └── locomo-poc/
        ├── manifest/
        │   └── runs.jsonl
        │
        ├── runs/
        │   ├── locomo-poc-a83f2901/
        │   │   ├── run.json
        │   │   ├── examples.parquet
        │   │   ├── summary.parquet
        │   │   └── errors.jsonl
        │   │
        │   └── locomo-poc-b12077ef/
        │       └── ...
        │
        └── aggregate/
            ├── runs.parquet
            └── examples.parquet
```

Do not have multiple workers concurrently append to the same Parquet file.

Each task owns its own unique run directory.

Aggregation happens afterward.

This avoids distributed file locking and corruption issues.

---

# 18. Write semantics

Within each run:

```text
run_id -> unique storage prefix
```

The task should write locally first:

```text
/tmp/results/<run_id>/
```

Then upload completed artifacts to object storage.

Recommended sequence:

```text
1. write examples.parquet locally
2. write summary.parquet locally
3. write run.json locally
4. upload result files
5. upload final completion marker
```

For example:

```text
_SUCCESS
```

The presence of:

```text
runs/<run_id>/_SUCCESS
```

means the run completed successfully.

Aggregation must ignore run directories without `_SUCCESS`.

---

# 19. Idempotency

Before running an experiment, check whether:

```text
runs/<run_id>/_SUCCESS
```

already exists.

Default behavior:

```text
if completed:
    skip
```

Support an explicit override:

```bash
--force
```

Retries must therefore not duplicate completed runs.

This is required because cloud execution platforms may retry failed tasks.

---

# 20. Failure handling

A failure must not destroy partial diagnostic information.

On failure, write:

```text
run.json
errors.jsonl
```

where possible.

`errors.jsonl` records should contain:

```text
timestamp
run_id
example_id
stage
exception_type
message
attempt
```

Avoid writing secrets or complete credential-bearing request headers.

After diagnostic artifacts are persisted, exit with non-zero status so that the execution platform can retry the task.

---

# 21. Manifest

Before cloud execution, generate an explicit manifest.

Example:

```bash
python -m memorybench.manifests \
    --config configs/experiments/poc.yaml \
    --output runs.jsonl
```

Example manifest:

```json
{"run_index": 0, "run_id": "...", "...": "..."}
{"run_index": 1, "run_id": "...", "...": "..."}
{"run_index": 2, "run_id": "...", "...": "..."}
```

The manifest should be uploaded to:

```text
manifest/runs.jsonl
```

This gives a durable record of exactly what the cloud execution intended to run.

---

# 22. Aggregation

Implement:

```bash
python -m memorybench.aggregate \
    --experiment locomo-poc
```

The aggregator should:

```text
discover successful runs
read every summary.parquet
concatenate them
validate schema consistency
write aggregate/runs.parquet

discover example-level Parquet files
concatenate them
validate schema consistency
write aggregate/examples.parquet
```

The final canonical research datasets are:

```text
aggregate/runs.parquet
aggregate/examples.parquet
```

Aggregation must be safe to rerun.

Do not require a database.

---

# 23. Analysis notebook

Implement:

```text
notebooks/01_results_analysis.ipynb
```

The notebook should do only analysis, not experiment execution.

It should:

```python
import pandas as pd

runs = pd.read_parquet(...)
examples = pd.read_parquet(...)
```

Initial analyses should include:

```text
number of completed runs
number of failed/missing runs

mean benchmark score by model
mean score by memory method
mean score by model × memory method

standard deviation / confidence intervals across seeds

token usage by model
token usage by memory method

latency by model
latency by memory method

judge-token consumption

experiment wall-clock time
```

Include basic plots sufficient for validating the PoC.

Keep scientific analysis code separate from execution code.

---

# 24. Storage abstraction

Define:

```python
class ObjectStore(Protocol):

    def exists(self, path: str) -> bool:
        ...

    def upload(self, local_path: Path, remote_path: str) -> None:
        ...

    def download(self, remote_path: str, local_path: Path) -> None:
        ...

    def list(self, prefix: str) -> list[str]:
        ...
```

Implement:

```text
LocalObjectStore
GCSObjectStore
```

Experiment logic should only depend on `ObjectStore`.

This is one of the most important portability boundaries.

Future implementations could include:

```text
S3ObjectStore
PACE filesystem adapter
```

without changing experiments.

---

# 25. Local mode

The entire PoC must work without Google Cloud.

Example:

```bash
uv run python -m memorybench.run \
    --config configs/experiments/poc.yaml \
    --run-index 0
```

Storage config:

```yaml
storage:
  backend: local
  path: ./artifacts
```

Also support local Docker:

```bash
docker build -t memorybench .
```

and:

```bash
docker run \
    --env-file .env \
    -v "$(pwd)/artifacts:/app/artifacts" \
    memorybench \
    python -m memorybench.run \
    --config configs/experiments/poc.yaml \
    --run-index 0
```

Local execution must use the same application entry point as cloud execution.

---

# 26. Docker image

Use a simple production Dockerfile.

Requirements:

```text
Python 3.11+
non-interactive execution
dependencies installed reproducibly
source copied into image
working directory set
unbuffered Python logs
```

Do not run Jupyter inside the experiment container by default.

The container should execute a run and terminate.

Conceptually:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

# install dependencies

COPY src ./src
COPY configs ./configs

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "memorybench.cli"]
```

Use the exact dependency-install syntax appropriate to the chosen package manager.

---

# 27. Logging

Emit structured logs to stdout.

Each important message should contain:

```text
run_id
run_index
benchmark
model
memory_method
stage
```

Example conceptual log:

```json
{
  "event": "example_completed",
  "run_id": "locomo-poc-a83f2901",
  "example_id": "184",
  "agent_latency_seconds": 3.52,
  "judge_latency_seconds": 1.92
}
```

Do not implement a custom remote logging service.

Cloud Run stdout/stderr should naturally flow into Cloud Logging.

Local runs should remain human-readable.

---

# 28. Provenance

Every run must record:

```text
git commit SHA
experiment config hash
run config
Python version
package version
container image identifier when available
benchmark version
model identifier returned by provider where available
timestamp
```

This information belongs in:

```text
run.json
```

and important fields should also appear in `summary.parquet`.

Reproducibility is more important than minimizing metadata size.

---

# 29. API credentials

API keys must be supplied through environment variables from the experiment code's perspective.

Examples:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GOOGLE_API_KEY
```

Do not commit:

```text
.env
credentials
service-account JSON
API keys
```

Local development may use `.env`.

Google Cloud deployment should inject secrets using Secret Manager.

Application code should simply read the normal provider environment variable and remain unaware of Secret Manager.

---

# 30. Google Cloud architecture

The PoC architecture is:

```text
Git repository
      |
      v
Docker build
      |
      v
Artifact Registry
      |
      v
Cloud Run Job
      |
      +------ task 0
      +------ task 1
      +------ task 2
      |        ...
      +------ task N
               |
               v
        external LLM APIs
               |
               v
       Google Cloud Storage
```

Google resources required:

```text
Google Cloud project

Artifact Registry repository

Cloud Storage bucket

Secret Manager secrets

Cloud Run Job

Cloud Run execution service account
```

Avoid unnecessary resources such as:

```text
Cloud SQL
Firestore
Pub/Sub
GKE
Vertex AI
Cloud Functions
```

for the initial implementation.

---

# 31. Google Cloud IAM

Create a dedicated runtime service account, conceptually:

```text
memorybench-runner@PROJECT_ID.iam.gserviceaccount.com
```

Grant it only the permissions required to:

```text
read/write experiment objects in the designated GCS bucket
read configured LLM secrets
run as the Cloud Run Job service identity
```

Do not use downloaded service-account credentials inside Cloud Run.

Use Google Application Default Credentials automatically supplied through the Cloud Run service identity.

---

# 32. GCP deployment script

Implement:

```text
scripts/deploy_gcp.sh
```

It should approximately:

```text
build container
push container to Artifact Registry
create/update Cloud Run Job
configure runtime service account
configure CPU
configure memory
configure timeout
configure retry count
configure GCS bucket
configure secret mappings
```

Suggested initial resources:

```text
CPU: 1 vCPU
RAM: 1–2 GiB
task timeout: 12 hours
task retries: 2
```

Make these configurable rather than buried in source code.

---

# 33. Cloud execution

Implement:

```text
scripts/execute_gcp.sh
```

The script should:

```text
read experiment config
generate manifest
calculate number of runs N
upload manifest
execute Cloud Run Job with N tasks
set configurable parallelism P
```

Conceptually:

```bash
./scripts/execute_gcp.sh \
    configs/experiments/poc.yaml \
    --parallelism 10
```

If the matrix contains 100 runs:

```text
task_count = 100
parallelism = 10
```

Do not manually create 100 separate Cloud Run Jobs.

Use one job execution with 100 indexed tasks.

---

# 34. Rate-limit control

Parallelism must be treated as an experimental deployment parameter.

Example:

```text
task count = 100
parallelism = 5
```

Start conservatively.

Increase to:

```text
10
20
...
```

after observing provider rate limits.

Do not implement a centralized distributed rate limiter for the initial PoC.

Infrastructure-level parallelism plus per-task retry/backoff is sufficient initially.

---

# 35. Development milestones

Implement in this order.

## Milestone 1 — Local single run

Success criterion:

```bash
python -m memorybench.run \
    --config configs/experiments/poc.yaml \
    --run-index 0
```

produces:

```text
examples.parquet
summary.parquet
run.json
_SUCCESS
```

locally.

## Milestone 2 — Deterministic matrix

A config containing multiple models/methods/seeds expands deterministically.

Tests verify:

```text
stable number of runs
stable ordering
stable run IDs
```

## Milestone 3 — Local parallel smoke test

Execute several independent runs locally.

No two runs write into the same directory.

## Milestone 4 — Docker

The same experiment succeeds inside Docker.

Results must be scientifically identical apart from timing/environment metadata.

## Milestone 5 — GCS

Run locally using:

```yaml
storage:
  backend: gcs
```

and verify artifacts appear under the correct prefix.

## Milestone 6 — Cloud Run single task

Deploy the container to Cloud Run Jobs.

Execute one task.

Verify:

```text
LLM call succeeds
results reach GCS
logs reach Cloud Logging
```

## Milestone 7 — Cloud Run array job

Execute:

```text
3–5 tasks
```

in parallel.

Verify correct mapping:

```text
CLOUD_RUN_TASK_INDEX -> RunSpec
```

## Milestone 8 — Full PoC

Run approximately:

```text
10–20 experiment runs
```

with realistic benchmark/API calls.

Aggregate into Parquet and inspect with notebook.

Do not immediately launch all ~100 experiments until this smoke test passes.

---

# 36. Tests

At minimum implement automated tests for:

```text
configuration parsing

experiment matrix expansion

deterministic run IDs

Cloud task-index mapping

local storage

GCS path generation

completed-run skip behavior

aggregation

result schemas
```

A smoke test should use a fake LLM provider.

Example:

```python
FakeLLMClient
```

returning deterministic responses.

Unit tests must not call paid external APIs.

---

# 37. Mock/fake provider

Implement a simple deterministic fake LLM client.

Example behavior:

```python
FakeLLMClient.generate(...)
```

returns:

```text
text = deterministic response
input_tokens = deterministic integer
output_tokens = deterministic integer
latency = near-zero
```

This enables:

```text
CI tests
Docker tests
Cloud Run infrastructure tests
storage tests
```

without spending researcher API credits.

This fake provider is required.

---

# 38. Command-line interface

Target CLI:

```bash
memorybench manifest CONFIG

memorybench run CONFIG --run-index N

memorybench run CONFIG --run-id ID

memorybench aggregate EXPERIMENT

memorybench status EXPERIMENT
```

`status` should report approximately:

```text
expected runs: 100
completed: 73
failed/incomplete: 4
not started: 23
```

Avoid creating a large CLI framework.

Typer or argparse is sufficient.

---

# 39. Status detection

Status should be derived from object storage rather than a central database.

For every expected `run_id`:

```text
_SUCCESS exists -> completed

run directory exists but no _SUCCESS -> incomplete/failed

no run directory -> not started
```

This makes object storage the durable experiment state.

---

# 40. Parquet requirements

Use PyArrow-compatible Parquet.

Avoid arbitrary nested Python objects in important analysis columns.

Use simple types:

```text
string
integer
float
boolean
timestamp
```

Complex provider metadata may be JSON serialized into:

```text
raw_metadata_json
```

Keep core analytical fields first-class typed columns.

---

# 41. Scientific reproducibility requirements

A result must be traceable to:

```text
code version
configuration
benchmark version
model identifier
memory method
seed
judge model
prompt configuration
```

Prompt templates used by benchmarks or judges should either:

```text
live in version-controlled files
```

or be stored in `run.json`.

Do not allow critical prompts to exist only as undocumented strings in notebooks.

---

# 42. Cost instrumentation

Although LLM billing is external to this project, capture enough metadata to estimate it afterward.

For every provider/model combination, store:

```text
input tokens
output tokens
cached tokens when available
number of requests
retries
```

Do not hard-code monetary model prices into the experiment runner.

Cost calculation should happen during analysis from a separate pricing table.

Example:

```text
analysis/model_prices.csv
```

This avoids invalidating old experiment records when provider prices change.

---

# 43. Non-goals

Do not implement the following in the PoC:

```text
model training
GPU scheduling
distributed training

web dashboard
REST API
user authentication

SQL database

Kubernetes

Airflow
Ray
Celery

complex distributed queues

real-time visualization

automatic benchmark downloading unless trivially needed

cross-cloud deployment automation

PACE/Slurm support
```

However, preserve interfaces so that PACE/Slurm or another object-storage backend can be added later.

---

# 44. Important invariants

The implementation should preserve these invariants:

```text
one run = one immutable scientific configuration

one run_id = one experiment output directory

workers never share writable result files

completed runs are idempotently skipped

aggregation only reads completed runs

cloud execution is replaceable

LLM providers are replaceable

storage implementations are replaceable

benchmarks are replaceable

analysis operates only on canonical result files
```

These invariants are more important than the specific class names or folder names suggested above.

---

# 45. PoC experiment

Create a minimal PoC configuration using:

```text
benchmark: LoCoMo subset

agent model: one inexpensive API model

judge model: one inexpensive API model

memory methods:
    none
    mem0/reproduced implementation

seeds:
    1
    2

small benchmark subset:
    enough examples to exercise the complete pipeline
```

Target approximately:

```text
4–8 independent Cloud Run tasks
```

for the first real cloud experiment.

The PoC succeeds when those tasks execute concurrently and the resulting Parquet datasets can be aggregated and analyzed.

---

# 46. Acceptance criteria

The implementation is complete when all of the following hold:

```text
1. A developer can clone the repository and install dependencies.

2. A fake experiment runs locally without cloud credentials.

3. A real LLM API experiment runs locally.

4. The same experiment runs inside Docker.

5. Results can be written to either local storage or GCS through configuration alone.

6. A deterministic config expands into deterministic run IDs.

7. Cloud Run maps task index N to experiment run N.

8. Multiple Cloud Run tasks can execute simultaneously.

9. Retrying a completed run does not duplicate results.

10. Individual workers never write concurrently to the same Parquet file.

11. Every successful run produces:
    run.json
    examples.parquet
    summary.parquet
    _SUCCESS

12. Aggregate execution produces:
    aggregate/runs.parquet
    aggregate/examples.parquet

13. The analysis notebook reads those Parquet datasets without custom preprocessing.

14. Token usage, latency, model identity, memory method, benchmark, score,
    seed, git commit, and run ID are available for downstream analysis.

15. No GCP-specific code exists in benchmark, memory-system, evaluation,
    or LLM-provider scientific logic.
```

---

# 47. Engineering priorities

When choices arise, optimize in this order:

```text
1. scientific reproducibility
2. correctness
3. simplicity
4. observability/debuggability
5. portability
6. cost efficiency
7. scalability
```

The expected workload is on the order of tens to hundreds of independent experiments, not millions of distributed tasks.

Prefer the simplest implementation that remains scientifically reproducible and cloud-portable.

---

# 48. Expected final developer workflow

Local development:

```bash
git clone ...
cd agent-memory-bench

uv sync

memorybench run configs/experiments/poc.yaml --run-index 0
```

Local Docker:

```bash
docker build -t memorybench .

docker run ... memorybench \
    run configs/experiments/poc.yaml \
    --run-index 0
```

Generate experiment plan:

```bash
memorybench manifest configs/experiments/poc.yaml
```

Deploy:

```bash
./scripts/deploy_gcp.sh
```

Run cloud experiment:

```bash
./scripts/execute_gcp.sh \
    configs/experiments/poc.yaml \
    --parallelism 10
```

Inspect:

```bash
memorybench status locomo-poc
```

Aggregate:

```bash
memorybench aggregate locomo-poc
```

Analyze:

```bash
jupyter lab notebooks/01_results_analysis.ipynb
```

The exact shell syntax may evolve, but this workflow and architectural separation should be preserved.
