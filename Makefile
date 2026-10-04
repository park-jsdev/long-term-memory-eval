# experiment runner image — uv on CI/cloud, conda still fine on laptops.
.PHONY: poc test image

poc:
	python -m src.experiment_runner write-manifest configs/experiments/poc.yaml
	python -m src.experiment_runner execute-qa configs/experiments/poc.yaml --run-index 0

test:
	python -m pytest tests/test_experiment_runner_matrix.py tests/test_experiment_runner_execute_qa.py -q

image:
	docker build -t memorybench .
