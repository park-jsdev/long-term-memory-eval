# Notebooks

One study = one numeric prefix. Protocol first, analysis after Cloud Run finishes.

```text
NN_<frozen-axis>_<variable-axis>[_protocol|_analysis].ipynb
```

| File | Experiment | What |
|------|------------|------|
| `01_results_analysis.ipynb` | — | Generic aggregate loader |
| `02_mem0_reader_2024_writers_*` | retired | Claude 3.5 404; do not launch |
| `03_2025_readers_full_context_*` | parked 1 smoke | 3-family readers × `full_context` (includes Claude) |
| `04_2025_readers_full_context_*` | parked 2 baseline | 3-family readers × `{full_context, rag}` |
| `05_mem0_reader_2025_writers_*` | parked 3 writers | Frozen `gpt-4o-mini` × 3-family teachers |
| `06_2025_live_campaign_analysis.ipynb` | parked concat | `campaign_2025_live.yaml` |
| `07_2025_readers_openai_deepseek_smoke_*` | **1 smoke** | GPT-5 vs DeepSeek-V3 × `full_context`, 1 conversation |
| `08_2025_readers_openai_deepseek_*` | **2 baseline** | Same readers × `{full_context, rag}` |
| `09_mem0_reader_2025_writers_openai_deepseek_*` | **3 writers** | Frozen `gpt-4o-mini` × `{GPT-5, DeepSeek-V3}` teachers |
| `10_2025_openai_deepseek_campaign_analysis.ipynb` | campaign concat | Family tables across OpenAI vs DeepSeek packs |
| `11_2026_readers_openai_deepseek_smoke_*` | 2026 smoke | GPT-5.6 Terra vs DeepSeek-V4 × `full_context`, 1 conversation |
| `12_2026_readers_openai_deepseek_*` | 2026 baseline | Same readers × `{full_context, rag}` |
| `13_mem0_reader_2026_writers_openai_deepseek_*` | 2026 writers | Frozen `gpt-4o-mini` × `{Terra, DeepSeek-V4}` teachers |
| `14_2026_openai_deepseek_campaign_analysis.ipynb` | 2026 concat | Family tables across 2026 OpenAI vs DeepSeek packs |
| `15_year_family_robustness_analysis.ipynb` | year concat | Robustness, year moves, holes, rank flips, cost/latency/tokens, ROI, saturation |
| `16_locomo_full_context_window_analysis.ipynb` | audit + packs | Full-context injection vs published windows; coverage vs J/latency/USD |
| `17_openai_codex_poc_analysis.ipynb` | Codex PoC | Mini reader/writer PoC: tool audit + Table 2 pins |
| `17_openai_agent_reader_writer_analysis.ipynb` | OpenAI agents | Chat Completions 2024–2026 vs Codex 2026 harness |
| `17_openai_codex_persist_memory_analysis.ipynb` | Persist memory | notes_only vs persist-on vs session summaries |
| `17_openai_mini_vs_codex_writers_analysis.ipynb` | Mini vs Codex writers | Chat Completions vs Codex session summaries / teacher_graph, plus full_context ceiling |
| `17_openai_model_harness_gaps_analysis.ipynb` | Model vs harness gaps | Category J: 4o-mini vs Codex, 2026 vs 2024 model vs harness, plus adversarial refusal |

Year comparison / multi-teacher validity: `python -m src.memorybench report configs/analysis/campaign_year_family.yaml` writes `experiments/_campaign/year_family/analysis/` (mean bars, year-on-x line charts, plus insight CSVs, including cost). Paper Table 2 and gpt-4o-mini clone scores are YAML `pins:`. 2024→2025 pin vs live is not a matched sandwich; 2025→2026 live is. A `rank_flip` on FC vs RAG or teacher_graph vs summaries is a threat to freezing that axis in a multi-teacher middle layer. ROI uses score per second / per $1 / per 1k reasoning tokens; saturation labels `near_ceiling` and `diminishing_returns`.

Full-context vs the model window: `python -m scripts.analysis.context_window` writes `experiments/_campaign/context_window/` from audit dumps + campaign parquet. Not a memorybench job. `full_context` is the conversation (~26k paper / ~28k local), not a filled window.

Codex / OpenAI agent campaign: `python -m src.memorybench report configs/analysis/campaign_openai_codex_poc.yaml`, `campaign_openai_agents.yaml`, `campaign_openai_codex_persist_memory.yaml`, `campaign_openai_mini_vs_codex_writers.yaml`, and `campaign_openai_model_harness_gaps.yaml`. Tool-audit plots (`n_web_search`, `n_mcp`) should be ~0. Workspace J vs LoCoMo F1, gold-id recall bins, hop-to-evidence, and qidx×category are YAML recipes (`scripts/analysis/agent_harness.py` overlays collected traces). Paper pins are not a matched sandwich. Generation is model year; Codex CLI is a 2026 harness. `notes_only` is the persist memory-method cell.

Analysis notebooks are YAML wrappers with a **pre-test** (declared cells, hypotheses, priced volume estimate) and a **post-test** (finished-pack metrics + expected vs actual USD). Helpers: `notebook_pretest` / `notebook_posttest`. Do not put matplotlib or groupby in the notebook. Takeaway metric grids are HTML tables displayed separately from the finding prose (Cursor collapses Markdown tables mixed into a paragraph).

Active analysis YAML: `configs/analysis/campaign_2025_openai_deepseek.yaml` (completed 2025). Next campaign: `configs/analysis/campaign_2026_openai_deepseek.yaml`. Same plot recipes as the parked `campaign_2025_live.yaml`. Cost pins: `configs/models/pricing.yaml`. Engines: `scripts/analysis/campaign_tables.py` + `campaign_plots.py` + `src/memorybench/analysis/cost.py`. After pulling aggregate:

```bash
python -m src.memorybench report configs/analysis/campaign_2026_openai_deepseek.yaml
```

Operator copy-paste: `docs/agent/RUNBOOK_2026_OPENAI_DEEPSEEK.md`. Completed 2025: `docs/agent/RUNBOOK_2025_OPENAI_DEEPSEEK.md`. Parked three-family: `docs/agent/RUNBOOK_2025_LIVE.md`.
