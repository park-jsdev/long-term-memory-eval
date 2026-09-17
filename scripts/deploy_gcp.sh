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
JOB_MEMORY="${JOB_MEMORY:-4Gi}"
# 8 saturates current 4/8-cell waves. Override JOB_PARALLELISM if 429s appear.
JOB_PARALLELISM="${JOB_PARALLELISM:-8}"
EXP_YAML="${EXPERIMENT_YAML:-configs/experiments/poc_gcs.yaml}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker build -t "${IMAGE}" .
docker push "${IMAGE}"

COMMON=(
  --image="${IMAGE}"
  --region="${REGION}"
  --service-account="${SA}@${PROJECT_ID}.iam.gserviceaccount.com"
  --tasks=1
  --parallelism="${JOB_PARALLELISM}"
  --task-timeout=12h
  --max-retries=2
  --cpu=1
  --memory="${JOB_MEMORY}"
  --set-secrets="OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"
  --set-env-vars="PYTHONUNBUFFERED=1,MEMORYBENCH_BUCKET=${BUCKET:-${PROJECT_ID}-memorybench}"
)

gcloud run jobs describe memorybench-qa --region="${REGION}" >/dev/null 2>&1 \
  && gcloud run jobs update memorybench-qa "${COMMON[@]}" \
      --args="execute-qa,${EXP_YAML}" \
  || gcloud run jobs create memorybench-qa "${COMMON[@]}" \
      --args="execute-qa,${EXP_YAML}"

gcloud run jobs describe memorybench-autorater --region="${REGION}" >/dev/null 2>&1 \
  && gcloud run jobs update memorybench-autorater "${COMMON[@]}" \
      --args="execute-autorater,${EXP_YAML}" \
  || gcloud run jobs create memorybench-autorater "${COMMON[@]}" \
      --args="execute-autorater,${EXP_YAML}"

gcloud run jobs describe memorybench-aggregate --region="${REGION}" >/dev/null 2>&1 \
  && gcloud run jobs update memorybench-aggregate "${COMMON[@]}" \
      --args="aggregate,${EXP_YAML}" \
  || gcloud run jobs create memorybench-aggregate "${COMMON[@]}" \
      --args="aggregate,${EXP_YAML}"

gcloud run jobs describe memorybench-collect-full --region="${REGION}" >/dev/null 2>&1 \
  && gcloud run jobs update memorybench-collect-full "${COMMON[@]}" \
      --args="collect-full,${EXP_YAML}" \
  || gcloud run jobs create memorybench-collect-full "${COMMON[@]}" \
      --args="collect-full,${EXP_YAML}"

echo "Image ${IMAGE}"
echo "Experiment YAML ${EXP_YAML}"
echo "Jobs: memorybench-qa, memorybench-autorater, memorybench-aggregate, memorybench-collect-full"
