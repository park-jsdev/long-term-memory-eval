#!/usr/bin/env bash
set -euo pipefail
python -m src.experiment_runner write-manifest configs/experiments/poc.yaml
python -m src.experiment_runner execute-qa configs/experiments/poc.yaml --run-index 0
