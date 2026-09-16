# Analysis campaign YAML (`analysis_campaign.v1`)

**Recipes vs engines.** `configs/analysis/*.yaml` names the analyses and
visualizations (packs, `group_by`, metrics, plot `kind` / `x` / `hue` / `y`).
The reusable table and figure code lives in `scripts/analysis/campaign_tables.py`
and `scripts/analysis/campaign_plots.py`. A new campaign is a new YAML; do not
fork plot code.

Glue: `src/memorybench/analysis/` (load YAML, read aggregate parquet, write
`analysis/`). CLI: `python -m src.memorybench report configs/analysis/<file>.yaml`.
No LLM.

`campaign: 2025_live` in `configs/analysis/campaign_2025_live.yaml` is the parked three-family recipe (includes Claude). The budget overlay `configs/analysis/campaign_2025_openai_deepseek.yaml` includes that file, remaps pack names, and adds `openai_deepseek_thinking_axis.yaml` (thinking on/off group_by plus reasoning-token vs latency plots). `configs/analysis/campaign_year_family.yaml` is the 2024 vs 2025 vs 2026 family comparison (pins + live packs; 2026 packs are skipped until they exist). The 2026 overlay `configs/analysis/campaign_2026_openai_deepseek.yaml` does the same for GPT-5.6 Terra vs DeepSeek-V4.

## Layout

```yaml
campaign:
  id: 2025_live
  title: 2025 live campaign

defaults:
  source: examples          # examples.parquet (per-question) or runs
  metrics: [locomo_f1, token_f1, exact_match, judge_score]
  output_subdir: analysis

campaign_analyses:          # concat packs listed under experiments:
  - id: reader_family_by_category
    experiments: [smoke, baseline]
    group_by: [model_family, question_category]
    metrics: [locomo_f1, judge_score]
    plots:
      - kind: grouped_bar
        x: question_category
        hue: model_family
        y: locomo_f1

experiments:
  smoke:
    name: locomo-2025-readers-full-context-smoke
    pack: experiments/locomo-2025-readers-full-context-smoke
    family_from: reader     # or writer → model_family column
    analyses:
      - id: by_reader
        group_by: [reader_display_name, memory_method]
        plots:
          - kind: metrics_grouped_bar
```

`plots: [grouped_bar]` (string form) still works; omitted `x` / `hue` / `y`
fall back to `group_by` / `metrics` order. Prefer the explicit form in new YAMLs.

Optional recipe fields:

| YAML | Effect |
|------|--------|
| `where: {memory_method: [full_context]}` | Keep matching rows before grouping |
| `include_pins: true` | Concat campaign-level `pins:` (already-aggregated rows: paper / local_clone) |
| `pins:` (campaign root) | Literature or clone overall scores; set `generation`, `model_family`, `memory_method`, `result_source`, metrics, `n` |
| `pretest:` (per experiment) | Expected `n_cells`, `n_questions`, `scientific_claim`, `hypotheses` for notebook pre-test |
| `cost:` (campaign root) | Price expected vs actual: `pricing`, `scenario`, `volume_from`, `map_models`, `scale`, `parked` |

When `question_category` is **not** in `group_by`, overall means drop LoCoMo category 5 (adversarial) so F1 and judge score match Mem0 J / the 2024 paper clone. Category figures (`group_by` includes `question_category`) keep all five types. `exclude_question_categories: [5]` is then redundant on overall recipes; use it only to drop extra ids.

OpenAI vs DeepSeek overlays add `thinking` (`on`/`off`) to `group_by` and metrics `agent_reasoning_tokens` (reader, per question) / `teacher_reasoning_tokens` (writer, cell total from `cost.json`) so thinking spend can be plotted next to latency. Parked three-family YAML omits that axis.

Prices live in `configs/models/pricing.yaml`. Engine: `src/memorybench/analysis/cost.py`. Notebooks call `notebook_pretest` / `notebook_posttest` (no plot code). Missing actuals stay empty (not $0). Parked models are costed and excluded from the launched total.

Live packs get `generation` from `configs/models/generation_catalog.yaml` (writer model when `family_from: writer`, else `reader_generation`) and `result_source=live`. Do not average paper pins into live question rows: group by `result_source`. Missing 2026 packs are omitted from the axis until you add them; do not invent zero bars.

Missing packs are skipped (experiment 2/3 not pulled yet). Smoke subset is not a scientific claim.

## Outputs

```text
experiments/<experiment_name>/analysis/
  tables/<analysis_id>.csv
  plots/<analysis_id>.png
  tables/cost_by_cell.csv
  tables/cost_by_stage.csv
  tables/cost_parked.csv
  SUMMARY.md
experiments/_campaign/<campaign.id>/
  tables/  plots/  SUMMARY.md
```

`model_family` is derived (OpenAI / Anthropic / DeepSeek) from `reader_provider` or `writer_model`. Do not mix reader-sweep packs into writer-family campaign analyses.

## Plot kinds

| Kind | YAML fields | Engine |
|------|-------------|--------|
| `metrics_grouped_bar` | `hue` (optional; else join `group_by`) | `write_metrics_grouped_bar` — x = metric names |
| `grouped_bar` | `x`, `hue`, `y` | `write_grouped_bar` |
| `bar` | `x`, `y` | `write_bar` (latency unbounded) |

When `question_category` is a group column, tables and plot ticks use official LoCoMo JSON ids from `CATEGORY_NAMES` (`1 multi-hop`, `2 temporal`, `3 open-domain`, `4 single-hop`, `5 adversarial`) — not paper §4.1 prose numbering. Legend is always outside the axes. Overall (non-category) tables do not average category 5.
