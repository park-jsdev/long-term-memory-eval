# GCP resources for the memorybench PoC

Create **only** these. Do not add Cloud SQL, Firestore, Pub/Sub, GKE, Vertex AI, or Cloud Functions.

Set these once in your shell (PowerShell):

```powershell
$env:PROJECT_ID = "YOUR_PROJECT_ID"
$env:REGION = "us-central1"
$env:AR_REPO = "memorybench"
$env:BUCKET = "YOUR_PROJECT_ID-memorybench"
$env:SA = "memorybench-runner"
```

## 1. Project and APIs

1. A Google Cloud **project** you own (`PROJECT_ID`).
2. Enable:
   - `run.googleapis.com` (Cloud Run Jobs)
   - `artifactregistry.googleapis.com`
   - `storage.googleapis.com`
   - `secretmanager.googleapis.com`
   - `iam.googleapis.com`

```bash
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com storage.googleapis.com secretmanager.googleapis.com iam.googleapis.com
```

## 2. Artifact Registry

One **Docker** repository named `memorybench` in `$REGION`.

```bash
gcloud artifacts repositories create $AR_REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="memorybench experiment images"
```

Image URL shape:

```text
$REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/memorybench:TAG
```

## 3. Cloud Storage bucket

One bucket for run artifacts, manifests, aggregates, and **shared indexes**.

```bash
gcloud storage buckets create gs://$BUCKET --location=$REGION
```

Use this prefix layout (workers never append to the same Parquet file):

```text
gs://$BUCKET/
  shared/mem0_locomo10/          # upload once after mem0.run_index
  shared/rag_locomo10/           # upload once after rag.run_index
  data/locomo10.json             # optional; do not bake LoCoMo into the image
  experiments/<experiment_name>/
    manifest/runs.jsonl
    runs/<run_id>/               # each task owns this prefix
    aggregate/
```

## 4. Secret Manager

One secret **per LLM key**. Application code still reads `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY`. Cloud Run maps secrets onto those env vars.

```bash
# after: echo -n KEY | gcloud secrets create openai-api-key --data-file=-
gcloud secrets create openai-api-key --replication-policy=automatic
gcloud secrets create anthropic-api-key --replication-policy=automatic
gcloud secrets create deepseek-api-key --replication-policy=automatic
```

Do **not** download a service-account JSON for Cloud Run.

## 5. Runtime service account

```text
memorybench-runner@$PROJECT_ID.iam.gserviceaccount.com
```

```bash
gcloud iam service-accounts create $SA --display-name="memorybench Cloud Run runner"
```

Grant **only**:

| Role | Why |
|------|-----|
| `roles/storage.objectAdmin` on `gs://$BUCKET` | read/write run prefixes + shared indexes |
| `roles/secretmanager.secretAccessor` on the three secrets | inject API keys as env vars |

Do not grant `roles/owner` or project-wide storage admin.

## 6. Cloud Run Jobs (same image, two jobs)

| Job name | Args | Tasks |
|----------|------|--------|
| `memorybench-qa` | `execute-qa configs/experiments/poc_gcs.yaml` | `N` = matrix size; `CLOUD_RUN_TASK_INDEX` selects the row |
| `memorybench-autorater` | `execute-autorater configs/experiments/poc_gcs.yaml` | same `N`, only after QA `_SUCCESS` |

Suggested first limits (overridable later):

```text
CPU: 1
Memory: 2 GiB
Task timeout: 12 hours
Task retries: 2
Parallelism: 1 for poc, then 5
```

Mount / inject:

- secrets → `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`
- env `PYTHONUNBUFFERED=1`
- dataset: download `locomo10.json` from `gs://$BUCKET/data/` at start, or set `benchmark.dataset_path` to a local copy you copy into the container at job start (do not git the file)

## 7. What you do not create

Cloud SQL, Firestore, Pub/Sub, GKE, Vertex AI endpoints, Cloud Functions, a second job per matrix cell, or a downloaded JSON key inside the container.

## 8. Deploy the image (after secrets exist)

Docker Desktop must be running. From the **repo root**:

PowerShell (recommended on Windows; uses `gcloud.cmd`, no Bash):

```powershell
$env:PROJECT_ID = "YOUR_PROJECT_ID"
$env:REGION = "us-central1"
$env:AR_REPO = "memorybench"
$env:SA = "memorybench-runner"
.\scripts\deploy_gcp.ps1
```

Git Bash / Linux / macOS (`export`, then `./scripts/deploy_gcp.sh`). Shell scripts in this repo use Unix (LF) line endings; `set: pipefail` under `bash` on Windows usually means CRLF. Use the `.ps1` instead.

Jobs run `configs/experiments/poc_gcs.yaml` (downloads `gs://$BUCKET/data/locomo10.json`, writes `gs://$BUCKET/experiments/locomo-poc/runs/<run_id>/`). **Redeploy after that YAML/code exists in the image.**

Execute one QA task (mock; no live LLM):

```powershell
gcloud.cmd run jobs execute memorybench-qa --region=us-central1 --tasks=1 --wait
```

Then autorater (after QA `_SUCCESS` is in the bucket):

```powershell
gcloud.cmd run jobs execute memorybench-autorater --region=us-central1 --tasks=1 --wait
```

### Console after execute

Project `agent-platform-508416` (region `us-central1`):

1. [Cloud Run Jobs](https://console.cloud.google.com/run/jobs?project=agent-platform-508416) → `memorybench-qa` → **Executions** → latest execution. Status **Succeeded** (green). Open **Logs**. You should see `qa complete locomo-poc-… uploaded=N blobs`. Failures are usually missing `data/locomo10.json` or the runner SA lacking `storage.objectAdmin`.
2. Direct executions list: [memorybench-qa executions](https://console.cloud.google.com/run/jobs/details/us-central1/memorybench-qa/executions?project=agent-platform-508416).
3. [Bucket browser](https://console.cloud.google.com/storage/browser/agent-platform-508416-memorybench/experiments/locomo-poc/runs?project=agent-platform-508416) → folder `locomo-poc-<8 hex>/` (current mock cell is `locomo-poc-d43c3dda`). Must contain `_SUCCESS`, `predictions.jsonl`, `TRACE.md`, `reader/traces.jsonl`. After the autorater job: `autorater/_SUCCESS`.
4. Job **Configuration** / container args must be `execute-qa` + `configs/experiments/poc_gcs.yaml` (not `poc.yaml`). If you still see `poc.yaml`, the image was not redeployed.

## 9. Local vs cloud

Laptops: `conda activate distillation`. Image: `uv` or the Dockerfile `pip install -r requirements.txt`. Same CLI:

```bash
python -m src.memorybench execute-qa configs/experiments/poc.yaml --run-index 0
```
