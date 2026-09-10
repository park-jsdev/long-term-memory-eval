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
| `memorybench-qa` | `execute-qa configs/experiments/poc.yaml` | `N` = matrix size; `CLOUD_RUN_TASK_INDEX` selects the row |
| `memorybench-autorater` | `execute-autorater configs/experiments/poc.yaml` | same `N`, only after QA `_SUCCESS` |

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

## 8. Local vs cloud

Laptops: `conda activate distillation`. Image: `uv` or the Dockerfile `pip install -r requirements.txt`. Same CLI:

```bash
python -m src.memorybench execute-qa configs/experiments/poc.yaml --run-index 0
```
