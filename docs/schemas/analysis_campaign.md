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
fall back to `group_by` / `metrics` order. If YAML sets `x` or `hue` and that
column is missing from the grouped table, skip the figure — do not draw
`model_family`. Prefer the explicit form in new YAMLs.

Optional recipe fields:

| YAML | Effect |
|------|--------|
| `where: {memory_method: [full_context]}` | Keep matching rows before grouping |
| `include_pins: true` | Concat campaign-level `pins:` (already-aggregated rows: paper / local_clone) |
| `pins:` (campaign root) | Literature or clone overall scores; set `generation`, `model_family`, `memory_method`, `result_source`, metrics, `n` |
| `split_by:` (plot) | Write one PNG per distinct value of that column (e.g. one 3-bar paper / local clone / gpt-4o-mini + Codex figure per `paper_method`) |
| `pretest:` (per experiment) | Expected `n_cells`, `n_questions`, `scientific_claim`, `hypotheses` for notebook pre-test |
| `cost:` (campaign root) | Price expected vs actual: `pricing`, `scenario`, `volume_from`, `map_models`, `scale`, `parked` |
| `insights:` (campaign root) | Derived CSVs over a mean table (`from:` analysis or prior insight id). Kinds: `year_deltas`, `family_gaps`, `method_ranks`, `rank_flips`, `thinking_deltas`, `category_holes`, `pin_gaps`, `efficiency`, `saturation`, `j_f1_gap`. `pin_gaps` optional `live_generation:` (`2024` / `2025` / `2026`; default `2025`). `j_f1_gap` is judge J minus LoCoMo F1 on the same answers. `join_cost: true` attaches priced cell USD. Latency/token/USD year deltas treat **down** as `improving`. No plots. |
| `takeaways:` (campaign root) | Harness vs model-reader/writer findings. Each item has `finding` plus optional `from` / `from_right` mean tables, `left` / `right` filters, and `metrics`. `delta` is left minus right. Written first in `SUMMARY.md` and the notebook. Notebook `notebook_show` renders those tables as GitHub-flavored markdown (Cursor does not reliably display pandas HTML; `to_string` inside Markdown collapses columns). |

When `question_category` is **not** in `group_by`, overall means drop LoCoMo category 5 (adversarial) so F1 and judge score match Mem0 J / the 2024 paper clone. `source: runs` pack `locomo_f1` / `token_f1` / `exact_match` include category 5 in `metrics.json`; `load_pack` replaces those with the same cat-5-excluded example means as J. Category figures (`group_by` includes `question_category`) keep all five types. Failure-mode counts (`group_by` includes `failure_mode`) also keep category 5. `exclude_question_categories: [5]` is then redundant on overall recipes; use it only to drop extra ids. Tool-audit means (`n_web_search`, `n_mcp`, `n_retrieval_calls`) attached onto `source: runs` keep category 5 — a web search on an adversarial question is still a policy breach. Count columns autoscale; `used_non_workspace_tools` / `memory_recall` stay 0–1. Older `runs.parquet` without `agent` / `comparison_status` is backfilled from `manifest/runs.jsonl` and `aggregate/by_run/*/run_meta.json`. Stored `comparison_status=harness_failed` from the old any-failed rule is repaired to `incomparable` when `harness_failed_rate` < 1. Workspace recipes derive `answer_n_words`, `recall_bin` (gold `dia_id` coverage across retrieve events), `retrieval_calls_bin`, `hop_bin` (first retrieve step with a gold id), `qidx_bin` (question index in the conversation), `notes_bytes_bin`, `judge_vs_f1`, and `failure_mode_judge` (retrieve/reason split using J, because packed `failure_mode` uses `locomo_f1 > 0`). Older parquet without hop/notes columns is filled from `collected/runs/<id>/agent/trajectory.jsonl` and `notes_ledger.jsonl` by `scripts/analysis/agent_harness.py`. LoCoMo F1 and Mem0 J score the same predicted string; F1 is token overlap on short gold.

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

Live packs get `generation` from `configs/models/generation_catalog.yaml` (writer model when `family_from: writer`, else `reader_generation`) and `result_source=live`. Do not average paper pins into live question rows: group by `result_source`. Notebooks 17 paper-compare plots use `paper_method` (Table 2 ids) with `compare_source` on the full_context figure (`paper` / `local clone` / `live model` / `gpt-4o-mini + Codex` / 2025 / 2026) and `live_source` on the methods figure (`paper` / `live model` / `gpt-4o-mini + Codex`; local-clone pins fill methods this pack did not re-run as Chat Completions). Methods tables also report `locomo_f1`. Session-summaries **paper F1** is Maharana et al. 2024 Table 3 Summary RAG top-5 (0.325 overall, includes adversarial, 50 conversations / n=7512) — not locomo10 and not Mem0 J (`judge_score` stays empty until a live judge run). Persist-off `workspace_files` is the full_context Codex bar. Codex facts map to `mem0` and Codex `teacher_graph` to `mem0g` for literature context only. `mean_table` keeps those group columns (re-annotating if a concat dropped them) so methods are not averaged into one family bar. Missing 2026 packs are omitted from the axis until you add them; do not invent zero bars.

Notebooks 17 (`campaign_openai_codex_poc.yaml`, `campaign_openai_agents.yaml`, `campaign_openai_codex_persist_memory.yaml`, `campaign_openai_mini_vs_codex_writers.yaml`) pin Mem0 Table 2 Full-context / RAG k=2 256 / OpenAI / Mem0 / Mem0g plus local_clone J on the PoC/agents planes. Catalog generation is the **model year**; Codex CLI is a 2026 harness around that model. Sandwich `writer_harness` is `chat_completions` vs `codex` (`campaign_openai_mini_vs_codex_writers.yaml` is that comparison on `teacher_session_summaries` and Mem0g-shaped `teacher_graph`, plus a `memory_lane` x `system_harness` ceiling overlay against stuffed `full_context` and persist-off workspace). Native Codex cells stay audit-only unless `comparison_status=comparable`. Score plots do not group by that
status. A partial workspace-read miss is `harness_failed_rate` / failure-mode
counts, not a cell-level `harness_failed` bar. `notes_only` is the persist
memory-method cell; persist-on with sessions visible is a write ablation.

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
