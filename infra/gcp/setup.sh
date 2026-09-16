#!/usr/bin/env bash
# Create the PoC GCP resources. Fill PROJECT_ID first. See infra/gcp/README.md.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-memorybench}"
BUCKET="${BUCKET:-${PROJECT_ID}-memorybench}"
SA="${SA:-memorybench-runner}"

gcloud config set project "${PROJECT_ID}"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com

gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${AR_REPO}" \
      --repository-format=docker \
      --location="${REGION}" \
      --description="memorybench experiment images"

gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}"

gcloud iam service-accounts describe "${SA}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${SA}" --display-name="memorybench Cloud Run runner"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

echo "Create secrets openai-api-key, anthropic-api-key, deepseek-api-key if missing,"
echo "then grant roles/secretmanager.secretAccessor on each to ${SA}@${PROJECT_ID}.iam.gserviceaccount.com"
echo "Next: scripts/deploy_gcp.sh"
