# Analysis campaign YAML (`analysis_campaign.v1`)

**Recipes vs engines.** `configs/analysis/*.yaml` names the analyses and
visualizations (packs, `group_by`, metrics, plot `kind` / `x` / `hue` / `y`).
The reusable table and figure code lives in `scripts/analysis/campaign_tables.py`
and `scripts/analysis/campaign_plots.py`. A new campaign is a new YAML; do not
fork plot code.

Glue: `src/memorybench/analysis/` (load YAML, read aggregate parquet, write
`analysis/`). CLI: `python -m src.memorybench report configs/analysis/<file>.yaml`.
No LLM.

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

Missing packs are skipped (experiment 2/3 not pulled yet). Smoke subset is not a scientific claim.

## Outputs

```text
experiments/<experiment_name>/analysis/
  tables/<analysis_id>.csv
  plots/<analysis_id>.png
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

When `question_category` is a group column, tables and plot ticks use official LoCoMo JSON ids from `CATEGORY_NAMES` (`1 multi-hop`, `2 temporal`, `3 open-domain`, `4 single-hop`, `5 adversarial`) — not paper §4.1 prose numbering. Legend is always outside the axes.
