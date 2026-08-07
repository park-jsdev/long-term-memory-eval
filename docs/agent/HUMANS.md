# HUMANS.md — how to run and review (v0.1)

**Audience:** you (the researcher).  
**Length target:** 2–3 pages.  
**Update rule:** when workflows or outputs change, snap a copy into `docs/agent/traces/`, then edit this file.

---

## What this project is

You compare **how memory is built** for long multi-session chats (LoCoMo). The eventual design is an experimental **sandwich**:

| Layer | Fixed or variable? | Notes |
|-------|--------------------|--------|
| Top | Fixed | LoCoMo conversations + questions |
| Middle | Variable (later C0–C4) | How candidate memory is constructed / fused |
| Bottom | Fixed | Retrieval budget, answer LLM + prompt, metrics |

**v0.1 goal:** ship the **bottom + I/O skeleton** using LoCoMo’s own **session summaries** as memory, answered by one fixed OpenAI model, scored with LoCoMo-compatible metrics, and with **auditable logs/plots**.

You are not yet running multi-teacher fusion.

---

## One-time setup (conda)

```bash
conda create -n distillation python=3.11 -y
conda activate distillation
pip install -r requirements.txt
python scripts/fetch_locomo.py
```

Data lands at `data/raw/locomo10.json` (not committed; CC BY-NC 4.0).

For live API runs, put the key in a **repo-root `.env`** (gitignored):

```bash
copy .env.example .env
# edit .env → OPENAI_API_KEY=sk-...
```

The pipeline loads `.env` automatically on start. Shell export still works and wins if already set.

---

## Run the baseline

**Offline smoke (no money / no key):**

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --reader mock --max-questions 5 --run-id smoke_mock
```

**Small live check:**

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --max-questions 3 --run-id smoke_openai
```

**Full QA set** (many API calls; cache resumes if interrupted):

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --run-id baseline_session_summary
```

Default model is `gpt-4.1-mini` in `configs/baseline.yaml`. For a stronger fixed answer model (your intended GPT-4.1 class):

```bash
python -m src.locomo_eval.run --config configs/baseline.yaml --model gpt-4.1 --run-id baseline_gpt41
```

Config knobs live only in YAML + CLI overrides — no hidden flags.

---

## Where to look after a run

Everything for one experiment is under `experiments/<run_id>/`:

| File | Why open it |
|------|-------------|
| `predictions.csv` | Spreadsheet audit: Q, gold, pred, scores, memory clip |
| `predictions.jsonl` | Full rows including full memory text |
| `metrics.json` | Overall exact match / token F1 / **LoCoMo F1** |
| `metrics_by_category.csv` | Category breakdown (single-hop, temporal, …) |
| `run_meta.json` | Model, prompt version, data SHA, git hash, time |
| `plots/*.png` | Quick visual of overall + by-category scores |

Recompute metrics without re-calling the API:

```bash
python -m src.locomo_eval.evaluate --predictions experiments/<run_id>/predictions.jsonl
```

---

## How to read the metrics

- **exact_match / token_f1** — simple string overlap (SPEC_v1). Good for debugging.  
- **locomo_f1** — category-aware F1 matching the released LoCoMo eval protocol (what you want for paper comparisons). Category 5 (adversarial) expects phrases like “not mentioned” / “no information available”.

If LoCoMo F1 is low but answers “feel” right, check:

1. session summaries incomplete vs evidence turns,  
2. model verbosity vs short gold,  
3. adversarial category instructions in prompt.

---

## Mental model of the code (reviewable path)

```
locomo10.json
    → dataset.py          # normalize conversation + questions
    → memory.py           # SessionSummaryMemoryBuilder → Memory.text
    → prompts/qa_v1.txt   # fixed template
    → readers.py          # OpenAI (temp=0) or Mock, with disk cache
    → metrics + report    # scores, CSV/JSONL, plots
```

Later swaps should only replace **memory builders** (and eventually teacher/fusion), not the answer prompt or evaluator, if you want clean sandwich comparisons.

---

## Roadmap sketch (not in v0.1)

| ID | Middle layer | Question |
|----|--------------|----------|
| C0 | Raw / chunked dialog | Does structure help? |
| C1 | Single teacher memory | Standard pipeline strength |
| C2 | Top-1 of K teachers | Selection enough? |
| C3 | Aggregate whole memories | Synthesis enough? |
| C4 | Claim-level fusion + validation | Evidence-grounded fusion win? |

Teacher K∈{1,2,3} and utility U_K = Δscore / Δcost come **after** this baseline is trustworthy.

---

## What you should ask agents to do (and not)

**Do:** extend memory builders, improve reports, add Claude reader when you switch fixed answer model, fix bugs, keep docs current.

**Don’t (yet):** multi-LLM fusion, training loops, silent metric changes, deleting cache/data, committing secrets.

---

## Doc versioning

Live: `docs/agent/AGENTS.md`, `docs/agent/HUMANS.md`, `docs/agent/SPEC_v1.md`  
History: `docs/agent/traces/` (dated snapshots when these change)  
