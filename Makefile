# memorybench image — uv on CI/cloud, conda still fine on laptops.
.PHONY: poc test image

poc:
	python -m src.memorybench write-manifest configs/experiments/poc.yaml
	python -m src.memorybench execute-qa configs/experiments/poc.yaml --run-index 0

test:
	python -m pytest tests/test_memorybench_matrix.py tests/test_memorybench_execute_qa.py -q

image:
	docker build -t memorybench .
