#!/usr/bin/env bash
set -euo pipefail
python -m src.memorybench write-manifest configs/experiments/poc.yaml
python -m src.memorybench execute-qa configs/experiments/poc.yaml --run-index 0
