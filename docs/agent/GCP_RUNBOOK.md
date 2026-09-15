# GCP runbook — memorybench experiment pipeline

**Audience:** you, running the PoC on Cloud Run from Windows PowerShell.  
**Live 2025 campaign (copy-paste):** `docs/agent/RUNBOOK_2025_LIVE.md`.  
**Resource inventory (what exists, not how to click it):** `infra/gcp/README.md`.  
**Scientific matrix:** `docs/agent/EXPERIMENT_MATRIX_v1.md`.

Jobs always use `configs/experiments/poc_gcs.yaml`. Local mock (no GCP): `configs/experiments/poc.yaml`.

Never put API keys in this file, git, chat, or Cloud Run plaintext env vars. Keys live only in Secret Manager (cloud) or repo-root `.env` (laptop).

---

## Constants (set these once per shell)

Every command below reads these variables, so the runbook works in any project.

```powershell
$env:PROJECT_ID = "your-gcp-project-id"
$env:REGION = "us-central1"
$env:AR_REPO = "memorybench"
$env:BUCKET = "$($env:PROJECT_ID)-memorybench"
$env:SA = "memorybench-runner"
```

| Thing | Value |
|-------|--------|
| Project | `$env:PROJECT_ID` |
| Region | `us-central1` |
| Artifact Registry | `memorybench` |
| Bucket | `gs://$env:BUCKET` |
| Runner SA | `memorybench-runner@$env:PROJECT_ID.iam.gserviceaccount.com` |
| Experiment slug | `locomo-poc` |
| Cloud jobs YAML | `configs/experiments/poc_gcs.yaml` |

---

## 1. One-time: gcloud CLI

Install: https://docs.cloud.google.com/sdk/docs/install-sdk

```powershell
gcloud.cmd init
gcloud.cmd config set project $env:PROJECT_ID
gcloud.cmd config get-value project
```

Enable APIs:

```powershell
gcloud.cmd services enable run.googleapis.com artifactregistry.googleapis.com storage.googleapis.com secretmanager.googleapis.com iam.googleapis.com
```

---

## 2. One-time: create resources

Skip a create if `describe` already succeeds.

```powershell
gcloud.cmd artifacts repositories create $env:AR_REPO `
  --repository-format=docker `
  --location=$env:REGION `
  --description="memorybench experiment images"

gcloud.cmd storage buckets create gs://$env:BUCKET --location=$env:REGION --uniform-bucket-level-access

gcloud.cmd iam service-accounts create $env:SA --display-name="memorybench Cloud Run runner"

gcloud.cmd storage buckets add-iam-policy-binding gs://$env:BUCKET `
  --member="serviceAccount:$env:SA@$env:PROJECT_ID.iam.gserviceaccount.com" `
  --role="roles/storage.objectAdmin"
```

Do **not** create Cloud SQL, Firestore, Pub/Sub, GKE, Vertex AI, Cloud Functions, or a downloaded JSON key inside the container.

---

## 3. One-time: Secret Manager + IAM

Create empty secrets and let the runner SA read them:

```powershell
foreach ($s in "openai-api-key","anthropic-api-key","deepseek-api-key") {
  gcloud.cmd secrets create $s --replication-policy=automatic
  gcloud.cmd secrets add-iam-policy-binding $s `
    --member="serviceAccount:$env:SA@$env:PROJECT_ID.iam.gserviceaccount.com" `
    --role="roles/secretmanager.secretAccessor"
}
```

Add values with hidden input (PowerShell `echo -n` is not reliable):

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
Add-SecretValue anthropic-api-key
Add-SecretValue deepseek-api-key
```

Deploy wires these onto the jobs (not plaintext):

```text
OPENAI_API_KEY=openai-api-key:latest
ANTHROPIC_API_KEY=anthropic-api-key:latest
DEEPSEEK_API_KEY=deepseek-api-key:latest
```

If jobs already exist and you only need to re-attach secrets:

```powershell
$secrets = "OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"
foreach ($job in "memorybench-qa","memorybench-autorater","memorybench-aggregate","memorybench-collect-full") {
  gcloud.cmd run jobs update $job --region=$env:REGION --set-secrets=$secrets
}
```

---

## 4. Confirm what you created

```powershell
gcloud.cmd services list --enabled --filter="name:(run.googleapis.com OR artifactregistry.googleapis.com OR storage.googleapis.com OR secretmanager.googleapis.com)"
gcloud.cmd artifacts repositories describe $env:AR_REPO --location=$env:REGION
gcloud.cmd storage buckets describe gs://$env:BUCKET
gcloud.cmd iam service-accounts describe "$env:SA@$env:PROJECT_ID.iam.gserviceaccount.com"
gcloud.cmd secrets list
gcloud.cmd secrets versions list openai-api-key
gcloud.cmd secrets versions list anthropic-api-key
gcloud.cmd secrets versions list deepseek-api-key
gcloud.cmd run jobs list --region=$env:REGION
```

---

## 5. Dataset in the bucket (once)

From repo root, conda env `distillation`:

```powershell
python scripts/fetch_locomo.py
gcloud.cmd storage cp data/raw/locomo10.json gs://${env:BUCKET}/data/locomo10.json
```

Jobs fail without this blob. Re-upload only if the LoCoMo file changes.

Optional local check (no GCP):

```powershell
conda activate distillation
python -m src.memorybench execute-qa configs/experiments/poc.yaml --run-index 0
```

---

## 6. Deploy / redeploy the image

Docker Desktop must be running. From **repo root**. Redeploy after every code or YAML change that Cloud Run should pick up.

```powershell
cd C:\home\research\Distillation\distillation
$env:PROJECT_ID = "${env:PROJECT_ID}"
$env:REGION = "us-central1"
$env:AR_REPO = "memorybench"
$env:SA = "memorybench-runner"
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_gcp.ps1
gcloud.cmd run jobs list --region=us-central1
```

`deploy_gcp.ps1` builds, pushes `us-central1-docker.pkg.dev/${env:PROJECT_ID}/memorybench/memorybench:<git-sha>`, and create/updates four jobs. On Windows use the `.ps1`, not `deploy_gcp.sh` (LF/`pipefail` breaks under Git Bash).

| Job | Args | Tasks |
|-----|------|--------|
| `memorybench-qa` | `execute-qa configs/experiments/poc_gcs.yaml` | `N` = matrix size; PoC is **1** |
| `memorybench-autorater` | `execute-autorater configs/experiments/poc_gcs.yaml` | same `N`, only after QA `_SUCCESS` |
| `memorybench-aggregate` | `aggregate configs/experiments/poc_gcs.yaml` | **1**; default wave 3 |
| `memorybench-collect-full` | `collect-full configs/experiments/poc_gcs.yaml` | **1**; on-demand wave 3 |

Container args must be `poc_gcs.yaml`, not `poc.yaml`. If you still see `poc.yaml`, the image did not roll out.

---

## 7. Console check

Project `${env:PROJECT_ID}`, region `us-central1`:

1. [Cloud Run Jobs](https://console.cloud.google.com/run/jobs) — four jobs. Open `memorybench-qa`:
   - Image: `us-central1-docker.pkg.dev/${env:PROJECT_ID}/memorybench/memorybench:<git-sha>`
   - Service account: `memorybench-runner@…`
   - Variables & secrets: the three API keys as **secret references**, not plaintext
2. Artifact Registry — that same tag.
3. Secret Manager — three secrets with versions.
4. Bucket — after first QA, `experiments/locomo-poc/runs/<run_id>/`. Empty except IAM is expected **before** execute.

---

## 8. Four-wave pipeline

```text
shared indexes (once, later matrices) → QA (N) → autorater (N) → aggregate (1) → collect-full (1, optional)
```

- Batch **all QA** first. Then **all autorater**. Autorater fail-fasts if that cell’s QA `_SUCCESS` is missing (no poll).
- Autorater is joined to QA by **hashed `run_id`**, not by job execution id. Same YAML cell → same `run_id` (PoC: `locomo-poc-d43c3dda`).
- Default wave 3 is the **thin catalog**. Trigger collect-full only when you want complete `memory/` dumps in one prefix.
- Prefer `--async` (fire-and-forget). `--wait` blocks the shell until that execution ends.

### Mock PoC (current `poc_gcs.yaml`: mock reader/judge, 1 cell)

```powershell
gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=1 --async
# after QA _SUCCESS in the bucket:
gcloud.cmd run jobs execute memorybench-autorater --region=us-central1 --tasks=1 --async
gcloud.cmd run jobs execute memorybench-aggregate --region=us-central1 --tasks=1 --async
# only if you want every dump:
gcloud.cmd run jobs execute memorybench-collect-full --region=us-central1 --tasks=1 --async
```

Blocking variant (first smoke):

```powershell
gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=1 --wait
gcloud.cmd run jobs execute memorybench-autorater --region=us-central1 --tasks=1 --wait
```

Larger matrices: `--tasks=N` on QA and autorater (`N` = cell count). Aggregate and collect-full stay `--tasks=1`.

Logs: [memorybench-qa executions](https://console.cloud.google.com/run/jobs/details/us-central1/memorybench-qa/executions). Success looks like `qa complete locomo-poc-… uploaded=N blobs`. Usual failures: missing `data/locomo10.json`, or the runner SA lacking `storage.objectAdmin`.

---

## 9. Where files land (GCS)

Durable output is the bucket, not the Cloud Run VM (`/tmp/memorybench-experiments` is scratch).

```text
gs://${env:BUCKET}/
  data/locomo10.json
  experiments/locomo-poc/
    manifest/runs.jsonl
    runs/<run_id>/              # QA pack; autorater writes runs/<run_id>/autorater/
    aggregate/                  # default wave 3
    collected/                  # on-demand wave 3
```

| Job | Prefix |
|-----|--------|
| `memorybench-qa` | `…/experiments/locomo-poc/runs/<run_id>/` |
| `memorybench-autorater` | `…/runs/<run_id>/autorater/` |
| `memorybench-aggregate` | `…/experiments/locomo-poc/aggregate/` |
| `memorybench-collect-full` | `…/experiments/locomo-poc/collected/` |

Console: open the [bucket browser](https://console.cloud.google.com/storage/browser) and navigate to `${env:BUCKET}/experiments/locomo-poc/`, where the `runs/`, `aggregate/`, and `collected/` prefixes live.

QA `_SUCCESS` must exist before autorater. After autorater: `autorater/_SUCCESS`. After aggregate: `aggregate/SUMMARY.md`, parquet, `by_run/<run_id>/`.

---

## 10. Pull to local

Prefixes below are **static for `poc_gcs.yaml`**. Re-run the same `cp` anytime; it overwrites local files. Change paths only if you switch experiment YAML or add matrix cells.

| Prefix | Changes when |
|--------|----------------|
| `…/experiments/locomo-poc/aggregate/` | never, for this YAML |
| `…/experiments/locomo-poc/collected/` | never, for this YAML |
| `…/experiments/locomo-poc/runs/` | never, for this YAML |
| `…/runs/locomo-poc-d43c3dda/` | only if cell identity changes (method, seed, reader, subset, experiment name). Same PoC cell keeps `d43c3dda` |

### Catalog (default wave 3)

```powershell
New-Item -ItemType Directory -Force -Path experiments\locomo-poc | Out-Null
gcloud.cmd storage cp -r gs://${env:BUCKET}/experiments/locomo-poc/aggregate experiments/locomo-poc/
```

Local: `experiments/locomo-poc/aggregate/` (parquet, `by_run/<id>/`, `SUMMARY.md`). No `memory/`.

### Full packs (after `memorybench-collect-full`)

```powershell
New-Item -ItemType Directory -Force -Path experiments\locomo-poc | Out-Null
gcloud.cmd storage cp -r gs://${env:BUCKET}/experiments/locomo-poc/collected experiments/locomo-poc/
```

Local: `experiments/locomo-poc/collected/runs/<id>/` (complete pack including `memory/`).

### One QA pack (whole audit tree)

```powershell
gcloud.cmd storage cp -r gs://${env:BUCKET}/experiments/locomo-poc/runs/locomo-poc-d43c3dda experiments/
```

Local: `experiments/locomo-poc-d43c3dda/`. If autorater already ran, `autorater/` is inside this tree.

### One autorater pack only

```powershell
New-Item -ItemType Directory -Force -Path experiments\locomo-poc-d43c3dda | Out-Null
gcloud.cmd storage cp -r gs://${env:BUCKET}/experiments/locomo-poc/runs/locomo-poc-d43c3dda/autorater experiments/locomo-poc-d43c3dda/
```

Local: `experiments/locomo-poc-d43c3dda/autorater/`.

### Every cell

```powershell
gcloud.cmd storage cp -r gs://${env:BUCKET}/experiments/locomo-poc/runs experiments/locomo-poc/
```

Local: `experiments/locomo-poc/runs/<run_id>/`.

---

## 11. Live: 2025 campaign

Copy-paste sequence, pins, cell order, and failure modes: **`docs/agent/RUNBOOK_2025_LIVE.md`**.

Do **not** relaunch `mem0_reader_2024_writers_*` — Claude 3.5 Sonnet returns API `404`. Redeploy between experiments. QA then autorater then aggregate (do not fire all three in one breath).
