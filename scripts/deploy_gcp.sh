#!/usr/bin/env bash
# Build and push the experiment image; create/update Cloud Run Jobs.
# Requires: gcloud auth, PROJECT_ID, docker.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-memorybench}"
SA="${SA:-memorybench-runner}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/memorybench:${TAG}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker build -t "${IMAGE}" .
docker push "${IMAGE}"

COMMON=(
  --image="${IMAGE}"
  --region="${REGION}"
  --service-account="${SA}@${PROJECT_ID}.iam.gserviceaccount.com"
  --tasks=1
  --parallelism=1
  --task-timeout=12h
  --max-retries=2
  --cpu=1
  --memory=2Gi
  --set-secrets="OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"
)

gcloud run jobs describe memorybench-qa --region="${REGION}" >/dev/null 2>&1 \
  && gcloud run jobs update memorybench-qa "${COMMON[@]}" \
      --args="execute-qa,configs/experiments/poc.yaml" \
  || gcloud run jobs create memorybench-qa "${COMMON[@]}" \
      --args="execute-qa,configs/experiments/poc.yaml"

gcloud run jobs describe memorybench-autorater --region="${REGION}" >/dev/null 2>&1 \
  && gcloud run jobs update memorybench-autorater "${COMMON[@]}" \
      --args="execute-autorater,configs/experiments/poc.yaml" \
  || gcloud run jobs create memorybench-autorater "${COMMON[@]}" \
      --args="execute-autorater,configs/experiments/poc.yaml"

echo "Image ${IMAGE}"
echo "Jobs: memorybench-qa, memorybench-autorater"
