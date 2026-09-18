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
# collect-full is 1 task. 8×32Gi exceeds us-central1 default 20 vCPU / 40Gi quota.
COLLECT_FULL_MEMORY="${COLLECT_FULL_MEMORY:-32Gi}"
COLLECT_FULL_CPU="${COLLECT_FULL_CPU:-8}"
# 8 saturates current 4/8-cell QA/autorater waves. Aggregate and collect-full stay 1.
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
  --task-timeout=12h
  --max-retries=2
  --set-secrets="OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"
  --set-env-vars="PYTHONUNBUFFERED=1,MEMORYBENCH_BUCKET=${BUCKET:-${PROJECT_ID}-memorybench}"
)

upsert_job() {
  local name="$1" args="$2" cpu="$3" memory="$4" parallelism="$5"
  if gcloud run jobs describe "${name}" --region="${REGION}" >/dev/null 2>&1; then
    gcloud run jobs update "${name}" "${COMMON[@]}" \
      --cpu="${cpu}" --memory="${memory}" --parallelism="${parallelism}" --args="${args}"
  else
    gcloud run jobs create "${name}" "${COMMON[@]}" \
      --cpu="${cpu}" --memory="${memory}" --parallelism="${parallelism}" --args="${args}"
  fi
}

upsert_job memorybench-qa "execute-qa,${EXP_YAML}" 1 "${JOB_MEMORY}" "${JOB_PARALLELISM}"
upsert_job memorybench-autorater "execute-autorater,${EXP_YAML}" 1 "${JOB_MEMORY}" "${JOB_PARALLELISM}"
upsert_job memorybench-aggregate "aggregate,${EXP_YAML}" 1 "${JOB_MEMORY}" 1
upsert_job memorybench-collect-full "collect-full,${EXP_YAML}" "${COLLECT_FULL_CPU}" "${COLLECT_FULL_MEMORY}" 1

echo "Image ${IMAGE}"
echo "Experiment YAML ${EXP_YAML}"
echo "Jobs: memorybench-qa, memorybench-autorater, memorybench-aggregate, memorybench-collect-full"
