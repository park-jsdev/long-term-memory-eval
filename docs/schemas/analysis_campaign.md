# Analysis campaign YAML (`analysis_campaign.v1`)

**Recipes vs engines.** `configs/analysis/*.yaml` names the analyses and
visualizations (packs, `group_by`, metrics, plot `kind` / `x` / `hue` / `y`).
The reusable table and figure code lives in `scripts/analysis/campaign_tables.py`
and `scripts/analysis/campaign_plots.py`. A new campaign is a new YAML; do not
fork plot code.

Glue: `src/memorybench/analysis/` (load YAML, read aggregate parquet, write
`analysis/`). CLI: `python -m src.memorybench report configs/analysis/<file>.yaml`.
No LLM.

`campaign: 2025_live` in `configs/analysis/campaign_2025_live.yaml` is the parked three-family recipe (includes Claude). The budget overlay `configs/analysis/campaign_2025_openai_deepseek.yaml` includes that file, remaps pack names, and adds `openai_deepseek_thinking_axis.yaml` (thinking on/off group_by plus reasoning-token vs latency plots). `configs/analysis/campaign_year_family.yaml` is the 2024–2026 robustness plane (pins + live packs; notebook 15). Insight CSVs label year moves, family gaps, method-rank flips, thinking deltas, and category holes. Year-family recipes also emit `kind: line` plots (`x: generation`) so each condition is a time series next to the bars. Missing 2026 packs are omitted; do not invent zero bars. The 2026 overlay `configs/analysis/campaign_2026_openai_deepseek.yaml` does the same for GPT-5.6 Terra vs DeepSeek-V4.

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
| `insights:` (campaign root) | Derived CSVs over a mean table (`from:` analysis or prior insight id). Kinds: `year_deltas`, `family_gaps`, `method_ranks`, `rank_flips`, `thinking_deltas`, `category_holes`, `pin_gaps`, `efficiency`, `saturation`. `join_cost: true` attaches priced cell USD. Latency/token/USD year deltas treat **down** as `improving`. No plots. |

When `question_category` is **not** in `group_by`, overall means drop LoCoMo category 5 (adversarial) so F1 and judge score match Mem0 J / the 2024 paper clone. Category figures (`group_by` includes `question_category`) keep all five types. `exclude_question_categories: [5]` is then redundant on overall recipes; use it only to drop extra ids.

OpenAI vs DeepSeek overlays add `thinking` (`on`/`off`) to `group_by` and plot
`hue=thinking` with `x` = family / reader / writer so on vs off is compared
**within** each family (do not average those cells). Metrics
`agent_reasoning_tokens` (reader, per question) / `teacher_reasoning_tokens`
(writer, cell total from `cost.json`) sit next to latency. Parked three-family
YAML omits that axis. Older catalog packs without a parquet `thinking` column
are labeled from `aggregate/by_run/*/run_meta.json` (`teacher_thinking` /
reader thinking) when present.

Mem0 Table 2 latency is **search** p50/p95 (retrieval) and **total** p50/p95
(search + answer generate). Campaign recipes request
`search_latency_seconds_p50` / `total_latency_seconds_p95` (etc.). Generate-only
HTTP time remains `agent_latency_seconds` (successful calls; retries excluded).
Reasoning tokens are a separate thinking-axis metric, not a Mem0 paper column.
Each PNG title is that plot's `y` metric (optional per-plot `title:`), not the
analysis section `title`. Older catalog parquet without
`search_latency_seconds` is filled from hashed-run `predictions.jsonl` when
present. `full_context` search is 0; RAG search stays empty until those times
exist — do not treat missing RAG retrieval as 0. Total latency still adds
generate + 0 when search was never stored, so those bars do not disappear.

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
| `line` | `x`, `y` (`hue` optional) | `write_line` — year-on-x time series; remaining `group_by` columns join as series; thinking-off is dashed |
| `bar` | `x`, `y` | `write_bar` (scores 0–1; latency/tokens autoscale) |

Campaign concat includes `reader_family_by_category` and `writer_family_by_category`
(question category on x; thinking or family as hue). Do not drop the writer
category recipe — sandwich claims need LoCoMo type breakdown, not only overall F1.

When `question_category` is a group column, tables and plot ticks use official LoCoMo JSON ids from `CATEGORY_NAMES` (`1 multi-hop`, `2 temporal`, `3 open-domain`, `4 single-hop`, `5 adversarial`) — not paper §4.1 prose numbering. Legend is always outside the axes. Overall (non-category) tables do not average category 5.
