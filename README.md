# LoCoMo knowledge distillation (minimal scaffold)

Simple research scaffold for **knowledge distillation** with **apples-to-apples LoCoMo QA scoring** (F1 + category rules from [`snap-research/locomo`](https://github.com/snap-research/locomo) `task_eval/evaluation.py`).

Stack: `requirements.txt`, plain YAML, plain JSON loaders, local CSV logs.

## Setup

```bash
conda create -n distillation python=3.11 -y
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py
```

LoCoMo data is **CC BY-NC 4.0** (non-commercial). Pinned commit is recorded in `data/README.md`.

## Layout

```text
configs/default.yaml     # paths, KD knobs, log paths
data/raw/                # locomo10.json (fetched, gitignored)
data/processed/          # flattened JSONL
scripts/fetch_locomo.py
scripts/prepare_data.py
src/data/locomo.py       # JSON load + conversation flatten
src/metrics/locomo_qa.py # official-style QA F1 / categories
src/distill/losses.py    # KD+CE stub
src/train.py             # train stub
src/eval_locomo_qa.py    # score prediction JSON → CSV + summary
logs/                    # CSV logs
experiments/             # run outputs
```

## Prepare examples

```bash
python scripts/prepare_data.py --config configs/default.yaml --split all
```

## Evaluate predictions (apples-to-apples)

Predictions JSON should match LoCoMo sample shape:

```json
[
  {
    "sample_id": "...",
    "qa": [
      {"prediction": "May 7 2023"},
      ...
    ]
  }
]
```

Gold `answer` / `category` are taken from `data/raw/locomo10.json` by `sample_id` order.

```bash
python src/eval_locomo_qa.py --config configs/default.yaml --predictions path/to/preds.json --run-name baseline
```

Reports overall F1 and per-category F1 (ids: `4` single-hop, `1` multi-hop, `2` temporal, `3` open-domain, `5` adversarial), writes `logs/eval_qa.csv` and `experiments/<run>/locomo_qa_scores.json`.

## Train stub

```bash
python src/train.py --config configs/default.yaml
```

Set `model.teacher` / `model.student` in the YAML when you are ready to wire the real KD loop (`src/distill/losses.py`).

## Citation

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
