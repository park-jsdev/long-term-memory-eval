# Cloud Run

How to run one experiment matrix on Google Cloud. Each matrix cell is one Cloud Run task. A cell still answers questions one at a time. The parallel work is independent cells.

Local install and a mock cell are in [runbook.md](runbook.md). The resource list, without the click-path, is in [infra/gcp/README.md](../infra/gcp/README.md).

Do not put an API key, a project id, or a bucket name in a tracked file. Keys live in Secret Manager. On a laptop they live in `.env`.

The Cloud Run job names and the Docker image tag are `memorybench`. Those are the deployed resource names. The process inside the image is `python -m src.experiment_runner`.

## Names for this shell

Set these once. Every command below reads them.

```powershell
$env:PROJECT_ID = "your-gcp-project-id"
$env:REGION = "us-central1"
$env:AR_REPO = "memorybench"
$env:BUCKET = "$($env:PROJECT_ID)-memorybench"
$env:SA = "memorybench-runner"
```

| Thing | Value |
|---|---|
| Project | `$env:PROJECT_ID` |
| Region | `us-central1` |
| Artifact Registry repository | `memorybench` |
| Bucket | `gs://$env:BUCKET` |
| Runner service account | `memorybench-runner@$env:PROJECT_ID.iam.gserviceaccount.com` |
| PoC experiment | `locomo-poc` |
| PoC YAML | `configs/experiments/poc_gcs.yaml` |

`poc_gcs.yaml` is a mock reader and a mock judge, one cell. It writes to the bucket. The same cell with local files is `configs/experiments/poc.yaml`.

## 1. Install gcloud and enable APIs

Install the [Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk), then:

```powershell
gcloud.cmd init
gcloud.cmd config set project $env:PROJECT_ID
gcloud.cmd services enable run.googleapis.com artifactregistry.googleapis.com storage.googleapis.com secretmanager.googleapis.com iam.googleapis.com
```

## 2. Create the registry, bucket, and runner

Skip a create command if `describe` already succeeds.

```powershell
gcloud.cmd artifacts repositories create $env:AR_REPO `
  --repository-format=docker `
  --location=$env:REGION `
  --description="experiment runner images"

gcloud.cmd storage buckets create gs://$env:BUCKET --location=$env:REGION --uniform-bucket-level-access

gcloud.cmd iam service-accounts create $env:SA --display-name="experiment runner"

gcloud.cmd storage buckets add-iam-policy-binding gs://$env:BUCKET `
  --member="serviceAccount:$env:SA@$env:PROJECT_ID.iam.gserviceaccount.com" `
  --role="roles/storage.objectAdmin"
```

Do not add Cloud SQL, Firestore, Pub/Sub, GKE, Vertex AI, Cloud Functions, or a downloaded JSON key in the container.

## 3. Put API keys in Secret Manager

The deploy script mounts four secrets. Create them before the first deploy, including Codex, or the job update fails.

```powershell
foreach ($s in "openai-api-key","codex-api-key","anthropic-api-key","deepseek-api-key") {
  gcloud.cmd secrets create $s --replication-policy=automatic
  gcloud.cmd secrets add-iam-policy-binding $s `
    --member="serviceAccount:$env:SA@$env:PROJECT_ID.iam.gserviceaccount.com" `
    --role="roles/secretmanager.secretAccessor"
}
```

Add each value from a hidden prompt. PowerShell `echo` is a poor way to pass a secret.

```powershell
function Add-SecretValue([string]$name) {
  $value = Read-Host "Value for $name (input hidden)" -AsSecureString
  $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($value)
  )
  $tmp = Join-Path $env:TEMP "$name.txt"
  [System.IO.File]::WriteAllText($tmp, $plain)
  gcloud.cmd secrets versions add $name --data-file=$tmp
  Remove-Item $tmp -Force
}

Add-SecretValue openai-api-key
Add-SecretValue codex-api-key
Add-SecretValue anthropic-api-key
Add-SecretValue deepseek-api-key
```

Deploy maps them onto environment variables as secret references, not plaintext:

```text
OPENAI_API_KEY=openai-api-key:latest
CODEX_API_KEY=codex-api-key:latest
ANTHROPIC_API_KEY=anthropic-api-key:latest
DEEPSEEK_API_KEY=deepseek-api-key:latest
```

To re-attach secrets on jobs that already exist:

```powershell
$secrets = "OPENAI_API_KEY=openai-api-key:latest,CODEX_API_KEY=codex-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"
foreach ($job in "memorybench-qa","memorybench-autorater","memorybench-aggregate","memorybench-collect-full") {
  gcloud.cmd run jobs update $job --region=$env:REGION --set-secrets=$secrets
}
```

## 4. Upload the dataset once

From the repository root, with the local environment installed:

```powershell
python scripts/fetch_locomo.py
gcloud.cmd storage cp data/raw/locomo10.json gs://$env:BUCKET/data/locomo10.json
```

Jobs fail if that object is missing. Upload it again only when the LoCoMo file changes.

## 5. Build the image and create the jobs

Docker Desktop has to be running. Run this from the repository root after any code or YAML change that Cloud Run should pick up. On Windows use the PowerShell script. The shell script breaks under Git Bash on Windows.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1
gcloud.cmd run jobs list --region=$env:REGION
```

The script builds `us-central1-docker.pkg.dev/$env:PROJECT_ID/memorybench/memorybench:<git-sha>`, pushes it, and creates or updates four jobs. It sets `MEMORYBENCH_BUCKET` on the jobs. The runner also accepts `EXPERIMENT_RUNNER_BUCKET`.

| Job | What it runs | Tasks | Parallelism | CPU / memory |
|---|---|---|---|---|
| `memorybench-qa` | `execute-qa` on the experiment YAML | one per cell; the PoC is 1 | 8 | 1 / 4Gi |
| `memorybench-autorater` | `execute-autorater`, after that cell's QA `_SUCCESS` | same count | 8 | 1 / 4Gi |
| `memorybench-aggregate` | `aggregate` | 1 | 1 | 1 / 4Gi |
| `memorybench-collect-full` | `collect-full` | 1 | 1 | 8 / 32Gi |

The default us-central1 quota is 20 vCPU and 40Gi. Keep collect-full at parallelism 1. Eight tasks at 32Gi would ask for 256Gi and be rejected.

The job arguments must name `poc_gcs.yaml` for this PoC, not `poc.yaml`. `poc.yaml` writes to the local disk. If the console still shows `poc.yaml`, the new image did not roll out.

Another experiment YAML:

```powershell
$env:EXPERIMENT_YAML = "configs/experiments/your_matrix_gcs.yaml"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1
```

## 6. Check the console

In project `$env:PROJECT_ID`, region `us-central1`:

1. Cloud Run Jobs. Open `memorybench-qa`. The image tag matches the git sha you just pushed. The service account is `memorybench-runner`. The four API keys are secret references.
2. Artifact Registry has that same tag.
3. Secret Manager has four secrets, each with a version.
4. The bucket is empty of experiment output until the first QA task finishes. After that, look under `experiments/locomo-poc/runs/`.

## 7. Run the four waves

Finish every QA task before any autorater task. The autorater stops immediately if that cell has no QA `_SUCCESS`. It does not wait. Aggregate and collect-full are single tasks. Collect-full is optional: it copies the full packs, including `memory/`. Aggregate writes the thin catalog that analysis reads.

`--async` returns while the job runs. `--wait` holds the shell until that execution ends.

PoC, one mock cell:

```powershell
gcloud.cmd run jobs execute memorybench-qa --region=$env:REGION --tasks=1 --async
gcloud.cmd run jobs execute memorybench-autorater --region=$env:REGION --tasks=1 --async
gcloud.cmd run jobs execute memorybench-aggregate --region=$env:REGION --tasks=1 --async
gcloud.cmd run jobs execute memorybench-collect-full --region=$env:REGION --tasks=1 --async
```

Run the autorater line only after QA has written `_SUCCESS`. Run collect-full only when you want the full dumps.

A larger matrix uses `--tasks` equal to the cell count on QA and on the autorater. Aggregate and collect-full stay at `--tasks=1`. The cell index inside the container is `CLOUD_RUN_TASK_INDEX`.

A successful QA log line looks like `qa complete locomo-poc-… uploaded=N blobs`. The usual failures are a missing `data/locomo10.json` or a runner account without `storage.objectAdmin`.

## 8. Where the files go

The durable copy is the bucket. `/tmp/memorybench-experiments` on the VM is scratch and disappears with the task.

```text
gs://$env:BUCKET/
  data/locomo10.json
  experiments/locomo-poc/
    manifest/runs.jsonl
    runs/<run_id>/
    aggregate/
    collected/
```

| Job | Prefix |
|---|---|
| `memorybench-qa` | `experiments/locomo-poc/runs/<run_id>/` |
| `memorybench-autorater` | `experiments/locomo-poc/runs/<run_id>/autorater/` |
| `memorybench-aggregate` | `experiments/locomo-poc/aggregate/` |
| `memorybench-collect-full` | `experiments/locomo-poc/collected/` |

`<run_id>` is the hashed id for that cell. The same YAML cell keeps the same id. Read it from the QA log or from `manifest/runs.jsonl`.

## 9. Copy results back

Replace `RUN_ID` with the id from the log. These commands overwrite the local copies.

```powershell
New-Item -ItemType Directory -Force -Path experiments\locomo-poc | Out-Null
gcloud.cmd storage cp -r gs://$env:BUCKET/experiments/locomo-poc/aggregate experiments/locomo-poc/
```

That is the thin catalog: parquet, `SUMMARY.md`, and `by_run/`. It does not include `memory/`.

Full packs, after collect-full:

```powershell
gcloud.cmd storage cp -r gs://$env:BUCKET/experiments/locomo-poc/collected experiments/locomo-poc/
```

One cell, including its autorater directory if the judge has already run:

```powershell
gcloud.cmd storage cp -r gs://$env:BUCKET/experiments/locomo-poc/runs/RUN_ID experiments/
```

Every cell:

```powershell
gcloud.cmd storage cp -r gs://$env:BUCKET/experiments/locomo-poc/runs experiments/locomo-poc/
```

Staged reproduction of a full campaign, including when to spend API credits, is in [REPRODUCE.md](REPRODUCE.md).
