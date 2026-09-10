#!/usr/bin/env bash
# Generate the manifest, then execute Cloud Run QA with N tasks.
# Usage: ./scripts/execute_gcp.sh configs/experiments/poc.yaml --parallelism 1
set -euo pipefail

CONFIG="${1:?config yaml}"
shift || true
PARALLELISM=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallelism) PARALLELISM="${2}"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

REGION="${REGION:-us-central1}"

python -m src.memorybench write-manifest "${CONFIG}"
N="$(python -c "
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.load_experiment_yaml import load_experiment_yaml
print(len(expand_run_matrix(load_experiment_yaml('${CONFIG}'))))
")"

echo "Executing ${N} QA tasks, parallelism=${PARALLELISM}"
gcloud run jobs execute memorybench-qa \
  --region="${REGION}" \
  --tasks="${N}" \
  --parallelism="${PARALLELISM}" \
  --wait
