# Notebooks

One study = one numeric prefix. Protocol first, analysis after Cloud Run finishes.

```text
NN_<frozen-axis>_<variable-axis>[_protocol|_analysis].ipynb
```

| File | Experiment | What |
|------|------------|------|
| `01_results_analysis.ipynb` | — | Generic aggregate loader |
| `02_mem0_reader_2024_writers_*` | retired | Claude 3.5 404; do not launch |
| `03_2025_readers_full_context_*` | **1 smoke** | 2025 readers × `full_context`, 1 conversation |
| `04_2025_readers_full_context_*` | **2 baseline** | Same readers × `{full_context, rag}` (Mem0-paper RAG), full LoCoMo |
| `05_mem0_reader_2025_writers_*` | **3 writers** | Frozen `gpt-4o-mini` × 2025 `{summaries, graph}` teachers |
| `06_2025_live_campaign_analysis.ipynb` | campaign concat | Family tables across available packs |

Analysis YAML: `configs/analysis/campaign_2025_live.yaml` (recipes only). Engines: `scripts/analysis/campaign_tables.py` + `campaign_plots.py`. After pulling aggregate:

```bash
python -m src.memorybench report configs/analysis/campaign_2025_live.yaml
```

Operator copy-paste for 1→2→3: `docs/agent/RUNBOOK_2025_LIVE.md`.
