# LoCoMo multi-teacher memory research

conda + plain YAML + plain JSON + CSV/plots.  
**Now (v0.1):** session-summary baseline QA with a fixed OpenAI answer model.  
**Later:** multi-teacher fusion (C0–C4 sandwich). See `docs/agent/HUMANS.md`.

## Setup

```bash
conda create -n distillation python=3.11 -y
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py
```

Set `$env:OPENAI_API_KEY` for live runs.

## Phase 1 baseline (session summaries → GPT)

Offline smoke:

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --reader mock --max-questions 5 --run-id smoke_mock
```

Live (small):

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --max-questions 3 --run-id smoke_openai
```

Full set:

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --run-id baseline_session_summary
```

Outputs under `experiments/<run_id>/`: `predictions.csv`, `metrics.json`, `plots/`, `run_meta.json`.

Rescore without API:

```bash
python -m src.locomo_eval.evaluate --predictions experiments/<run_id>/predictions.jsonl
```

Tests:

```bash
python -m pytest tests/test_baseline.py -q
```

## Layout

```text
configs/baseline.yaml
prompts/qa_v1.txt
src/locomo_eval/          # baseline pipeline
src/metrics/locomo_qa.py  # official LoCoMo F1
docs/agent/               # SPEC, AGENTS, HUMANS, traces
experiments/<run_id>/     # audit pack
data/raw/locomo10.json    # fetched, gitignored
```

## Docs for humans and agents

- Humans: [`docs/agent/HUMANS.md`](docs/agent/HUMANS.md)
- Agents: [`docs/agent/AGENTS.md`](docs/agent/AGENTS.md)
- Spec: [`docs/agent/SPEC_v1.md`](docs/agent/SPEC_v1.md)
- History: [`docs/agent/traces/`](docs/agent/traces/)

## Citation

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
