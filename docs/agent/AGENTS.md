# AGENTS.md — agent operating notes (v0.1)

**Audience:** coding agents working in this repo.  
**Length target:** 2–3 pages.  
**Update rule:** change this file on every meaningful behavior or layout change; snapshot to `docs/agent/traces/` first.

---

## Mission

LoCoMo eval: model-only Chat Completions versus a Codex agent, plus string scores and a separate Mem0 judge. Frozen-reader runs (`experiment.type: frozen_reader`) hold the reader fixed and vary the memory system. One interchangeable writer model. Terms: `docs/glossary.md`. Layout: `docs/LAYOUT.md`.

**Current phase:** `experiment_runner` expands a YAML matrix to hashed run ids and one Cloud Run task per run spec. Scientific code stays in `locomo_eval`. Do **not** claim paper Table 1–2 J from the OSS clones.

`Mem0GraphMemory` is the locked graph schema when a condition uses a graph. The writer is one model (`writer.model` / `--writer-model`). Packs store `writer_model`.

---

## North star (later)

```
Write: conversation → one writer model → memory store
Read:  question → retrieval or workspace → fixed answer LLM → LoCoMo evaluator
```

Condition ids: `raw_chunks`, `session_summaries`, `session_summaries`, `graph`, `full_context`, `rag`, `openai_memory`, `mem0`, `mem0g`, `workspace_files`. Use these ids in logs.

**Now:** Default **reader** is Mem0-parity (`gpt-4o-mini` + `prompts/readers/qa_mem0_v1.txt`). Paper methods: `python -m src.locomo_eval.eval_pipeline --method …` (`rag`, `full_context`, `openai_memory`, `mem0` / `mem0g`). Writer graphs use `graph` (`pool: single`, `configs/writers/openai_mini.yaml`). `session_summaries` is the same one-model writer emitting summaries. Matrices that name `pooled_teacher_*` or `fused_teacher_*` fail. Reader-model or `qa_v1` swaps remain a **separate** robustness axis.

**Agent-level (new):** `workspace_files` + `configs/agents/` (Codex first). Model-only remains one-shot `full_context`. Isolate persist on/off and tools native/controlled. Codex exec always passes `-c web_search="disabled"`, `-c sandbox_permissions=["disk-full-read-access"]`, `--ignore-user-config`, and `--ignore-rules` (no user MCP/execpolicy; workspace `cat`/`read` allowed). Trajectory kinds `web_search` / `mcp` are outside-workspace audit, not retrieve; `agent/metrics.json` rolls up `n_web_search_sum` / `n_mcp_sum` / `used_non_workspace_tools_rate` / `n_harness_failed` / `harness_failed_rate`. Every harness run writes `agent_comparison.v1` to `run_meta.json` and `agent/COMPARISON.md`. Only `status=comparable` supports strict cross-agent claims; native Codex is audit-only until it can enforce shared tool/retrieval limits. A question with no workspace read is `harness_execution_failure`, not `retrieval_failure`. Run `comparison_status` is `harness_failed` only when every question failed; a partial miss is `harness_failed_rate` (report load relabels stored any-failed run specs). Mock smoke does not need the Codex binary. Live GPT-5 + Codex is `configs/presets/agent_codex_gpt5.yaml` after mock+GCP smokes. See `docs/schemas/agent_runtime.md`.

**OpenAI agent designs:** `openai_agent_{readers,writers}.yaml` is the
thinking-off 2024/2025/2026 all-condition plane; `openai_codex_readers.yaml`
is no-memory workspace answering; `openai_codex_writers.yaml` uses Codex as a
summary/fact/graph writer with frozen GPT-4o-mini reader;
`openai_codex_end_to_end.yaml` permits persistent notes with sessions still
visible; `openai_codex_persist_memory.yaml` is the 3-condition memory-method test
(persist-off / persist-on+full / persist-on+notes_only);
`openai_mini_writers_structured.yaml` is the 2-condition GPT-4o-mini Chat Completions
writer twin of PoC Codex summaries/graph. Operator instructions:
`docs/agent/RUNBOOK_OPENAI_AGENTS.md`, `docs/agent/RUNBOOK_OPENAI_CODEX_PERSIST.md`,
and `docs/agent/RUNBOOK_OPENAI_MINI_VS_CODEX_WRITERS.md`. The focused reader
comparison uses `openai_mini_codex_readers_analysis.yaml`: six run specs with the
same rendered reader payload across model-only, Codex persist-off, and Codex
persist-on. The finished pack is pinned at
`experiments/locomo-openai-mini-codex-readers-analysis-v2`. The design YAML
`configs/analysis/design_openai_mini_codex_readers_analysis.yaml` reads that
pack, and `notebooks/17_openai_mini_codex_readers_analysis.ipynb` is the
published walkthrough. Operator steps:
`docs/runbook_mini_vs_codex_readers.md`.

---

## Repo map (v0.1)

| Path | Role |
|------|------|
| `configs/` | Composable YAML: `writers/`, `readers/`, `agents/`, `layouts/`, `autoraters/`, `stacks/`, `presets/`, `experiments/` — see `configs/README.md` |
| `configs/presets/mem0_baseline.yaml` | CLI default: Mem0-parity reader + `session_summaries` writer |
| `configs/presets/agent_codex_gpt5.yaml` | Codex + GPT-5 harness (minimal model knobs); `--reader mock` for smoke |
| `configs/writers/` | Memory methods (`raw_chunks`, `mem0`, `graph`, `workspace_files`, …) |
| `configs/readers/` | Answer LLM request controls |
| `configs/agents/` | Harness adapters / persist / tools (`codex`, later `claude_code`, `opencode`, `pi`) |
| `configs/layouts/` | Answer prompt + message layout |
| `configs/autoraters/` | Judge configs (QA does not include these) |
| `configs/models/generation_catalog.yaml` | Fillable `api_model_id` / `model_snapshot` table |
| `configs/models/pricing.yaml` | List-price pins for campaign cost estimates (not invoices) |
| `configs/models/context_windows.yaml` | Published context windows for utilization analysis (not a runtime cap) |
| `prompts/` | Role folders matching configs: `readers/`, `agents/`, `writers/`, `autoraters/` — see `prompts/README.md` |
| `prompts/readers/qa_mem0_v1.txt` | Pinned released Mem0 answer prompt for baseline parity |
| `prompts/agents/qa_workspace_v1.txt` | Harness prompt: retrieve from conversation files, JSON `answer` |
| `prompts/readers/qa_v1.txt` | Alternate short prompt; override only as a separate bottom-layer axis |
| `prompts/writers/session_summary_v1.txt` | Writer session-summary prompt |
| `prompts/writers/graph_v1.txt` | Writer → entity/relation JSON (Mem0g-shaped) |
| `prompts/autoraters/autorater_mem0_v1.txt` | Mem0 LLM-as-a-Judge prompt |
| `prompts/writers/mem0_extract_v1.txt` / `prompts/writers/mem0_update_v1.txt` | Mem0 fact extract + ADD/UPDATE/DELETE/NONE (pin: mem0 @ ece7ff6b) |
| `prompts/writers/mem0g_*.txt` | Entity / relation / conflict (pin: mem0 graph @ 69a832dc) |
| `docs/LAYOUT.md` | Where models, memory, and trajectories live |
| `docs/reports/` | Local writeups — **gitignored** |
| `docs/schemas/memory_runtime.md` | Runtime `{memory}` audit |
| `docs/schemas/preprocess_runtime.md` | Session-block preprocess schema (`preprocess_io.v1`) |
| `docs/schemas/experiment_pack.md` | Dump/load contract for one experiment pack (`audit_pack.v3` claim audit) |
| `docs/schemas/mem0_index.md` | Write-index dump schema (`mem0_index.v1`) |
| `docs/schemas/rag_index.md` | RAG chunk dump (`rag_index.v1`) |
| `docs/schemas/openai_memory_index.md` | Privileged extract-all dump |
| `src/locomo_eval/` | Baseline package |
| `scripts/compare_full_runs.py` | Comparison report for finished run packs; infers frozen prompt from run metadata |
| `scripts/analysis/` | Reusable analyses + plots (`run_benchmark` calls the autorater API unless mock) |
| `scripts/analysis/campaign_tables.py` / `campaign_plots.py` | YAML-driven campaign/experiment means + bars and year-on-x lines (legend outside; category names) |
| `scripts/analysis/campaign_insights.py` | Year-family labels: deltas, family gaps, rank flips, thinking, category holes |
| `scripts/analysis/agent_harness.py` | Overlay hop / notes / writes from collected agent traces onto campaign parquet |
| `scripts/analysis/compare_predictions.py` | Two prediction JSONLs → paired LoCoMo F1 boxplot + histograms (kernel used by compare_full_runs) |
| `scripts/analysis/run_benchmark.py` | Finished prediction pack → Mem0 F1/BLEU-1/J + literature tables, histograms, boxplots, latency plots |
| `scripts/analysis/compare_to_paper.py` | Offline paper Table 2 J vs best local autorater J (grouped bars) |
| `scripts/analysis/context_window.py` | Offline: billed input vs published windows, coverage vs J/latency/USD (notebook 16) |
| `scripts/analysis/verify_experiments.py` | Offline pack verifier (TRACE/prompts/logs) + graph year diagnosis |
| `src/metrics/locomo_qa.py` | Official LoCoMo category F1 |
| `data/raw/locomo10.json` | Dataset (gitignored; fetch) |
| `experiments/<run_id>/` | Human-auditable run pack (gitignored, except the pinned `locomo-openai-mini-codex-readers-analysis-v2` reference) |
| `configs/experiments/*.yaml` | Harness matrices (`poc`, frozen reader, agent, longitudinal, ablation) |
| `src/experiment_runner/` | Thin orchestrator: expand matrix, hashed ids, QA then autorater, Parquet |
| `docs/agent/SPEC_v2.md` | Cloud-portable experiment runner requirements — **local only** |
| `docs/agent/EXPERIMENT_MATRIX_v1.md` | Scientific matrix + skip vs regenerate — **local only** |
| `infra/gcp/README.md` | Exact GCP resources to create |
| `docs/gcp.md` | Cloud Run: infra, deploy, execute, pull from GCS |
| `docs/agent/RUNBOOK_2025_LIVE.md` | Parked three-family 2025 campaign (includes Claude) |
| `docs/agent/RUNBOOK_2025_OPENAI_DEEPSEEK.md` | Budget 2025: GPT-5 vs DeepSeek-V3 |
| `docs/agent/RUNBOOK_2026_OPENAI_DEEPSEEK.md` | 2026: GPT-5.6 Terra vs DeepSeek-V4 |
| `docs/agent/RUNBOOK_OPENAI_AGENTS.md` | OpenAI Chat Completions vs Codex campaign operator steps |
| `docs/agent/RUNBOOK_OPENAI_CODEX_PERSIST.md` | 3-condition persist-as-memory GCS copy-paste |
| `docs/agent/RUNBOOK_OPENAI_MINI_VS_CODEX_WRITERS.md` | 2-condition mini Chat Completions vs Codex summary/graph writers |
| `docs/runbook_mini_vs_codex_readers.md` | Six-run-spec GPT-4o-mini/Codex reader analysis: model, persist-off, and persist-on for full context and dataset summaries |
| `configs/analysis/design_2025_live.yaml` | Three-family analysis plane (tables/plots, no LLM) |
| `configs/analysis/design_2025_openai_deepseek.yaml` | OpenAI vs DeepSeek packs + thinking on/off designs |
| `configs/analysis/openai_deepseek_thinking_axis.yaml` | Within-family thinking on/off; separate token / generate / search / total plots |
| `configs/analysis/design_year_family.yaml` | 2024–2026 robustness + year-move story (pins + live; notebook 15) |
| `configs/analysis/design_openai_codex_poc.yaml` | Mini Codex PoC: tool audit + Table 2 pins + 2024 mini vs Codex vs 2025/2026 model-only J (notebook 17) |
| `configs/analysis/design_openai_agents.yaml` | Chat Completions 2024–2026 vs Codex 2026 harness (notebook 17) |
| `configs/analysis/design_openai_codex_persist_memory.yaml` | Persist-off / persist-on / notes_only vs summaries (notebook 17) |
| `configs/analysis/design_openai_mini_vs_codex_writers.yaml` | Mini Chat Completions vs Codex summaries/graph writers plus full_context ceiling overlay (notebook 17) |
| `configs/analysis/design_openai_mini_codex_readers_analysis.yaml` | GPT-4o-mini/Codex reader analysis; reads the pinned `locomo-openai-mini-codex-readers-analysis-v2` pack; notebook 17 |
| `configs/analysis/design_2026_openai_deepseek.yaml` | Same designs, 2026 Terra vs V4 packs |
| `notebooks/` | Local notebooks — **gitignored**, except `17_openai_mini_codex_readers_analysis.ipynb` |
| `docs/agent/SPEC_v1.md` | Phase 1 requirements — **local only** |
| `README.md` | Landing page: harness diagram, local quickstart, license, citation |
| `docs/loop.md` | Teaching note: one question from config to notebook |
| `docs/architecture.md` | Architecture: harness and experiment runner |
| `docs/runbook.md` | Install, build, and run |
| `docs/REPRODUCE.md` | Staged reproduction runbook (offline stages 0–3, paid stages 4–7) |
| `LICENSE` / `NOTICE.md` | MIT code license + LoCoMo (CC BY-NC) and Mem0 (Apache-2.0) terms |
| `docs/agent/HUMANS.md` | Human-facing brief — **local only** |
| `docs/agent/traces/` | Doc version history — **local only** |

### Package modules (`src/locomo_eval/`)

| File | Responsibility |
|------|----------------|
| `schemas.py` | Conversation, Session, Turn, SessionBlock, Question, Memory, Prediction |
| `dataset.py` | Load LoCoMo JSON → Conversation (read path; preprocess is separate) |
| `preprocess/` | HLD (i): `DataIngestor` + `PreprocessingPipeline` + session-document join + deterministic `run_index` dump |
| `model_orchestrator.py` | One writer model → locked `Mem0GraphMemory` (passthrough if no model) |
| `memory.py` | MemoryBuilder interface + raw_chunks / session_summaries / graph / full_context / rag / openai_memory / mem0 / mem0g |
| `rag/` | Paper RAG: tiktoken chunk, embed dump, cosine top-k, `full_context` builder |
| `openai_memory/` | Privileged extract-all dump + retrieve-all (not ChatGPT Memory product) |
| `mem0/` | Write-index: ingest pairs, extract, update, vector store, locked `GraphMemory`, dump, `run_index` CLI |
| `writer_callers.py` | Write-path LLM clients (via `writer_model.py` → orchestrator; not the reader) |
| `reasoning_extractor.py` | Shared reasoning-text helpers (reader + teacher callers) |
| `eval_pipeline.py` | Index → QA → optional autorater seeds for one `--method` |
| `agents/` | Harness eval: workspace files, Codex/mock adapters, retrieval trajectory |
| `stats.py` | Mean ± std, 95% CI, Wilcoxon, McNemar (no API) |
| `prompts.py` | Load/render prompt text |
| `readers.py` | OpenAI + Mock readers, temp=0 |
| `models.py` | Model ids / families / Chat Completions kwargs |
| `writer_model.py` | Session-summary + graph writer models (openai / anthropic / deepseek / mock) |
| `ping_writers.py` | Cheap-model plumbing ping (not QA) |
| `autorater.py` | Mem0-style CORRECT/WRONG LLM judge; GPT-4o + mock |
| `mem0_metrics.py` | Mem0 lexical F1/BLEU-1 and latency summaries |
| `mem0_baselines.py` | Published Mem0 Tables 1–2 values (literature pins, not re-runs) |
| `metrics.py` | EM, token F1, LoCoMo F1 |
| `report.py` | JSONL/CSV/plots |
| `run.py` | CLI: one memory YAML → one audit pack (`run_locomo_pipeline_with_memory_config`; compare is a separate script) |
| `experiment_pack/` | Experiment-pack I/O: `audit_layout` + `claim_audit` (lineage/cost/SUMMARY) + `audit_writer` vs `audit_loader` + `prompt_bundle` (`TRACE.md`) + `verify_pack` / `verify_graph_years` (offline log verifier). Output stays `experiments/<run_id>/` |
| `audit_pack.py` | Compat shim re-exporting `audit_writer` / `audit_layout` |
| `offline_evaluate.py` | CLI: rescore stored predictions with string metrics only (no API, not an LLM autorater) |

### Package modules (`src/experiment_runner/`)

Purpose-named files (no generic `run.py` / `config.py`). Wraps locomo_eval; does not reimplement metrics or memory.

| File | Responsibility |
|------|----------------|
| `expand_run_matrix.py` | YAML → ordered `ExperimentRunSpec` list |
| `hashed_run_id.py` | Deterministic `<experiment>-<8 hex>` |
| `execute_qa_run.py` | One run spec → locomo_eval QA + Parquet + `_SUCCESS` |
| `execute_autorater_run.py` | Separate judge job on stored predictions |
| `analysis/` | Load analysis YAML + write report dirs; tables/plots come from `scripts/analysis/campaign_*`; cost from `analysis/cost.py`; `context_window.py` is a standalone coverage/window report (not a job) |
| `aggregate_successful_runs.py` | Third wave: collect all run specs → `experiments/<name>/aggregate/` |
| `completed_run_skip.py` | Skip if `_SUCCESS` unless `--force` |
| `open_configured_store.py` / `local_object_store.py` / `gcs_object_store.py` / `gcs_run_workspace.py` | Portable storage; GCS download/upload for Cloud Run (dataset + `shared/<index_run_id>/`) |
| `experiment_cli.py` | `write-manifest`, `execute-qa`, `execute-autorater`, `aggregate`, `collect-full`, `report`, `status` |

---

## Commands agents should use

```bash
conda activate <your-env-name>
pip install -r requirements.txt
python scripts/fetch_locomo.py

# Experiment harness (matrix → one run spec). Mock PoC, no API:
python -m src.experiment_runner write-manifest configs/experiments/poc.yaml
python -m src.experiment_runner execute-qa configs/experiments/poc.yaml --run-index 0
python -m src.experiment_runner execute-autorater configs/experiments/poc.yaml --run-index 0
python -m src.experiment_runner aggregate configs/experiments/poc.yaml
python -m src.experiment_runner collect-full configs/experiments/poc.yaml
python -m src.experiment_runner status configs/experiments/poc.yaml
python -m src.experiment_runner report configs/analysis/design_2025_openai_deepseek.yaml --experiment smoke
python -m src.experiment_runner report configs/analysis/design_year_family.yaml
python -m src.experiment_runner report configs/analysis/design_openai_codex_poc.yaml
python -m src.experiment_runner report configs/analysis/design_openai_agents.yaml
python -m src.experiment_runner report configs/analysis/design_openai_codex_persist_memory.yaml
python -m src.experiment_runner report configs/analysis/design_openai_mini_vs_codex_writers.yaml
python -m src.experiment_runner report configs/analysis/design_openai_mini_codex_readers_analysis.yaml

# Cloud Run PoC (GCS). Operator steps: docs/gcp.md
# Live 2025 campaign, OpenAI vs DeepSeek (budget): docs/agent/RUNBOOK_2025_OPENAI_DEEPSEEK.md
# Live 2026 campaign, Terra vs DeepSeek-V4: docs/agent/RUNBOOK_2026_OPENAI_DEEPSEEK.md
# Parked three-family (includes Claude): docs/agent/RUNBOOK_2025_LIVE.md

# Offline smoke (no API key)
python -m src.locomo_eval.run --config configs/presets/mem0_baseline.yaml --reader mock --max-questions 5 --run-id smoke_mock

# Agent harness mock smoke (no Codex binary, no API)
python -m src.locomo_eval.run --config configs/presets/agent_codex_gpt5.yaml --reader mock --max-questions 3 --run-id smoke_agent_codex
python -m src.experiment_runner write-manifest configs/experiments/openai_codex_persist_memory.yaml
python -m src.experiment_runner execute-qa configs/experiments/openai_codex_persist_memory.yaml --run-index 0
python -m src.experiment_runner write-manifest configs/experiments/openai_mini_writers_structured.yaml
python -m src.experiment_runner execute-qa configs/experiments/openai_mini_writers_structured.yaml --run-index 0
# GCP mock: configs/experiments/agent_codex_poc_gcs.yaml (Codex not required in the image)
# Live Codex + GPT-5 (needs `codex` on PATH + CODEX_API_KEY or CLI login; do not pass --reader mock)
# python -m src.locomo_eval.run --config configs/presets/agent_codex_gpt5.yaml --max-questions 3 --run-id live_agent_codex_gpt5

# Live OpenAI (needs OPENAI_API_KEY in repo-root .env or shell)
python -m src.locomo_eval.run --config configs/presets/mem0_baseline.yaml --max-questions 3 --run-id smoke_openai

# Full baseline (costly)
python -m src.locomo_eval.run --config configs/presets/mem0_baseline.yaml --run-id mem0_baseline_session_summary

# Offline rescore (string metrics only; not an LLM autorater)
python -m src.locomo_eval.offline_evaluate --predictions experiments/<run_id>/predictions.jsonl

# Online Mem0 judge over a finished run (GPT-4o-mini; skips category 5)
python -m scripts.analysis.run_benchmark --run experiments/<run_id>

# Same component, offline plumbing smoke
python -m scripts.analysis.run_benchmark --run experiments/<run_id> --autorater mock

# Evaluation pipeline (one method). Mock smoke, then live + judge seeds.
# --max-samples caps index and QA; --max-questions then takes file-order pairs.
# Subset smokes write experiments/<run_id>_index (they do not overwrite rag_locomo10).
python -m src.locomo_eval.eval_pipeline --method full_context --reader mock --max-questions 5 --autorater mock --n-judge-runs 2 --run-id smoke_eval_full_context
python -m src.locomo_eval.eval_pipeline --method rag --reader mock --embedder mock --max-samples 1 --max-questions 5 --autorater mock --run-id smoke_eval_rag
python -m src.locomo_eval.eval_pipeline --method openai_memory --reader mock --extractor mock --max-samples 1 --max-questions 5 --autorater mock --run-id smoke_eval_openai_memory

# RAG / full-context / OpenAI-memory as separate index then QA (same as eval_pipeline steps)
python -m src.locomo_eval.rag.run_index --config configs/writers/rag.yaml --embedder mock --max-samples 1 --run-id smoke_rag_index
python -m src.locomo_eval.openai_memory.run_index --extractor mock --max-samples 1 --run-id smoke_openai_memory_index

# Mem0 / Mem0g write-index (no QA). Mock smoke; live gpt-4o-mini is costly.
python -m src.locomo_eval.mem0.run_index --config configs/writers/mem0.yaml --extractor mock --embedder mock --max-samples 1 --run-id smoke_mem0_index
python -m src.locomo_eval.mem0.run_index --config configs/writers/mem0g.yaml --run-id mem0g_locomo10

# Multi-seed J aggregate (no API; packs already judged)
python -m scripts.analysis.aggregate_seeds --packs experiments/<run>/autorater_seeds/seed_00 experiments/<run>/autorater_seeds/seed_01 --out experiments/<run>/autorater

# Paper Table 2 J vs best local live-judge seed (no API; packs already judged)
python -m scripts.analysis.compare_to_paper --runs experiments/full_context_qa experiments/rag_k2_256_qa experiments/mem0_qa --out experiments/compare_paper_vs_local

# Full-context injection vs published windows (offline; audit dumps + campaign parquet)
python -m scripts.analysis.context_window --out experiments/_design/context_window

# Offline pack verifier (configs, prompts, schemas, logs; no API). Exit 1 if invalid.
python -m scripts.analysis.verify_experiments experiments/<run_id>
python -m scripts.analysis.verify_experiments --graph-years experiments/locomo-mem0-reader-2025-writers-openai-deepseek experiments/locomo-mem0-reader-2026-writers-openai-deepseek

# Deterministic preprocess write-index (no LLM). Full locomo10, then 10 reader calls per sanity memory:
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess
python -m src.locomo_eval.preprocess.run_index --run-id locomo_preprocess --eval-questions 10
python -m src.locomo_eval.preprocess.run_index --eval-questions 10 --eval-reader mock --run-id smoke_preprocess

# Unit tests — preprocess (HLD i) + Mem0 index + evaluation (HLD iv) + controlled-comparison regression locks
# pytest.ini disables pytest-asyncio (not used; old plugin + pytest 9 fails collection).
python -m pytest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_preprocess_index.py tests/test_mem0_index.py tests/test_rag_index.py tests/test_openai_memory.py tests/test_stats.py tests/test_eval_pipeline.py tests/test_evaluation_pipeline.py tests/test_regressions.py tests/test_autorater_sanity.py tests/test_compare_to_paper.py tests/test_integration_sanity.py tests/test_run_isolation.py tests/test_model_orchestrator.py tests/test_memory_writer.py tests/test_experiment_pack.py tests/test_claim_audit.py tests/test_experiment_runner_matrix.py tests/test_experiment_runner_execute_qa.py tests/test_experiment_runner_aggregate.py tests/test_config_includes.py tests/test_prompt_bundle.py tests/test_gcs_run_workspace.py tests/test_analysis_design.py tests/test_campaign_cost.py tests/test_campaign_insights.py tests/test_context_window.py tests/test_verify_experiments.py tests/test_agent_harness.py -q
# or (file path avoids a site-packages module named `tests` shadowing this folder)
python -m unittest tests/test_preprocessing_pipeline.py tests/test_session_documents.py tests/test_preprocess_index.py tests/test_mem0_index.py tests/test_rag_index.py tests/test_openai_memory.py tests/test_stats.py tests/test_eval_pipeline.py tests/test_evaluation_pipeline.py tests/test_regressions.py tests/test_autorater_sanity.py tests/test_compare_to_paper.py tests/test_integration_sanity.py tests/test_run_isolation.py tests/test_model_orchestrator.py tests/test_memory_writer.py tests/test_experiment_pack.py tests/test_claim_audit.py tests/test_experiment_runner_matrix.py tests/test_experiment_runner_execute_qa.py tests/test_experiment_runner_aggregate.py tests/test_config_includes.py tests/test_prompt_bundle.py tests/test_gcs_run_workspace.py tests/test_analysis_design.py tests/test_campaign_cost.py tests/test_campaign_insights.py tests/test_context_window.py tests/test_verify_experiments.py tests/test_agent_harness.py

# Compare two prediction sets (offline; LoCoMo F1 boxplot + histograms)
python -m scripts.analysis.compare_predictions --a experiments/cmp_raw_chunks --b experiments/cmp_session_summaries --out experiments/compare_raw_chunks_session_summaries
python scripts/compare_full_runs.py --runs experiments/cmp_raw_chunks experiments/cmp_session_summaries --out experiments/compare_raw_chunks_session_summaries
python scripts/compare_cross_model.py --runs experiments/session_summaries_mini experiments/session_summaries_luna --axis reader --out experiments/compare_reader_mini_luna


# Session-document tables + dataset histograms (no API; needs locomo10.json)
python scripts/export_session_documents.py --data data/raw/locomo10.json --out data/processed
```

Set API keys via repo-root `.env` (`copy .env.example .env`) or shell:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`.
Never commit keys or `.env`. `src/locomo_eval/env.py` loads `.env` at run start / teacher-caller init.

# Writer plumbing ping (cheap models; needs keys unless --mock)
python -m src.locomo_eval.ping_writers --mock
python -m src.locomo_eval.ping_writers --providers openai,anthropic,deepseek

# Mock writer graph (no API). One model. --writer-model swaps it.
python -m src.locomo_eval.run --config configs/writers/graph.yaml --reader mock --writer mock --max-questions 3 --run-id smoke_graph
python -m src.locomo_eval.run --config configs/writers/session_summaries.yaml --reader mock --writer mock --writer-model gpt-4o-mini --max-questions 3 --run-id smoke_writer_summaries
# Writer thinking on by default except GPT-5.x / GPT-6 catalog overlays (off + 8192) when matrix.thinking is absent. OpenAI vs DeepSeek designs set matrix.thinking on/off. Frozen GPT-5.6 reader stays reasoning_effort=none unless --reader-thinking on; gpt-5 off-settings use minimal.
python -m src.locomo_eval.run --config configs/writers/graph.yaml --thinking off --reader mock --writer mock --max-questions 3 --run-id smoke_thinking_off

# Live writer (needs the provider key for writer.model). Do not pass --writer mock.
python -m src.locomo_eval.run --config configs/writers/graph.yaml --max-questions 3 --run-id live_teacher_graph

---

## Run audit package (always write)

Each run under `experiments/<run_id>/` must include:

- `TRACE.md` — config include chain → prompt files → jsonl outputs
- `prompts/` — snapshot copies of every `*_prompt_path` in that YAML
- `predictions.jsonl` — one row per question (incl. memory text); also copied to `reader/`
- `predictions.csv` — spreadsheet-friendly + scores + memory preview
- `metrics.json` — overall + by-category
- `metrics_by_category.csv`
- `run_meta.json` — model, **teacher_model**, prompt, data hash, git hash, timestamp, `audit_layout`
- `config.source.yaml` / `config.resolved.yaml` — source YAML copy + loaded YAML with CLI overrides
- `agent/` — harness traces, retrieval trajectory, conversation workspaces (when `answer_mode: agent`)
- `cost.json` — reader/teacher token totals and pinned-USD rollup
- `SUMMARY.md` — human claim-audit report (lineage pointers, teacher quality, ingest)
- `ATTRIBUTION.md` / `attribution.jsonl` — LLM call → pipeline role → claims made
- `plots/` — overall + category bars
- `reader/` — answer-LLM traces (`traces.jsonl`) + LoCoMo predictions
- `memory/` — `{memory}` payload; `lineage.jsonl` (question → item → writer); `retrieve_ranks.jsonl` (losers included); `memory/writer/` when a writer ran (calls, session text, quality.json); `memory/graph/` when graph memory was built (`ingest.jsonl`)

Each QA run also writes `ATTRIBUTION.md` (human) and `attribution.jsonl` (machine): every LLM call, the role it played (reader / teacher / graph), and the claims that call produced (triples, summaries, predicted answers), joined to fusion `kept` and lineage injection when those files exist.

Agents must not silently skip CSV/plots when code paths change.

An autorater pack under `experiments/<run_id>/autorater/` must include
`autorater_verdicts.jsonl`, `traces.jsonl` (judge reasoning), `schema.json`,
`autorater_metrics.json`, `run_meta.json`,
`SUMMARY.md`, `tables/`, and `plots/`. It is a snapshot: every invocation
clears these generated artifacts and regenerates from one prediction file.
Never append autorater analyses.

A Mem0 write-index under `experiments/<run_id>/mem0_index/` must include
`run_meta.json`, `schema.json`, `index.jsonl`, and `by_sample/<id>/speaker_a.json`
+ `speaker_b.json` (and `graph.json` when `enable_graph`). Reusing a run id
clears that dump and regenerates every sample.

A deterministic preprocess dump under `experiments/<run_id>/preprocess/` must include
`run_meta.json`, `schema.json`, `index.jsonl`, and `by_sample/<id>/sessions.jsonl`
+ `documents.jsonl`. No LLM. Gold answers stay out of the dump.

---

## Design rules for agents

1. **Frozen reader vs other designs:** frozen-reader YAMLs (`experiment.type: frozen_reader`) freeze reader+prompt and vary the memory system. Default frozen reader remains `gpt-4o-mini` + `qa_mem0_v1`; mem0 writer stays `gpt-4o-mini` extract for shared indexes. Sweeps/ablations are separate YAMLs (`experiment.type`). **Agent eval** (`type: agent`) freezes the harness model and varies adapter / persist / tools; `workspace_files` is not a stuffed-context memory claim. Do not mix a reader sweep into a frozen-reader claim. One writer model per run spec (`tests/test_memory_writer.py`). GPT-5.6 frozen readers stay `reasoning_effort=none`; hosted `gpt-5` off-settings use `minimal` (API rejects `none`). OpenAI vs DeepSeek designs set `matrix.thinking: [off, on]` for **both** families as readers (sweep) and writers (frozen-reader experiments). DeepSeek on/off always sends `extra_body.thinking` `{type: enabled|disabled}` (`deepseek-chat` defaults off; `deepseek-v4-flash` defaults on). Headroom: reader 256/8192, teacher 8192/32768 (overrides catalog `teacher:`). `--reader-thinking` is the reader-sweep flag; `--thinking` remains write-path. Parked three-family YAML omits the axis; catalog GPT-5/5.6/6 `teacher:` stays off + 8192 there. Overall analysis F1/J drop LoCoMo category 5 (Mem0); category plots keep it. Reader-analysis `adversarial_refusal` / `false_refusal` keep both classes: category-5 refusal uses the LoCoMo phrase rule, and false refusal is that phrase on categories 1–4 (`*_n` counts the defined rows). Gold-token drop stays conditional omission on categories 1–4 and is not plotted (it complements kept). `reader_usd` is pinned list price per question, judge excluded. Reasoning tokens (`agent_reasoning_tokens` / `teacher_reasoning_tokens`) are first-class campaign metrics next to latency. Each figure title is its `y` metric (tokens ≠ generate ≠ search p50 ≠ total p95). If YAML `y` is missing from the table, skip that figure — do not draw locomo_f1 under a judge_score filename. If YAML plot `x` or `hue` is missing from the grouped table, skip the figure — do not fall back to `model_family`. `mean_table` re-annotates `paper_method` / `live_source` / `compare_source` rather than dropping them from `group_by` (that drop averaged every memory method into one OpenAI bar). `runs.parquet` stores pack F1 including category 5 and has no J; `load_pack` copies cat-5-excluded mean `judge_score` / `locomo_f1` / `token_f1` / `exact_match` from examples, and mean tool-audit columns (`n_web_search`, `n_mcp`, `used_non_workspace_tools`, `n_retrieval_calls`, `memory_recall`) over all questions (audit keeps category 5). Count metrics autoscale. Workspace J vs LoCoMo F1 is the same predicted string (F1 = token overlap on short gold; J = Mem0 paraphrase judge). Notebook `notebook_show` / takeaways render GitHub-flavored markdown tables (Cursor does not reliably display pandas HTML). Gold-id `recall_bin` is union over retrieve events; `evidence_retrieved` is all-or-nothing. `hop_bin` is the first retrieve step that hit a gold `dia_id` (writes do not count). Persist-off workspace prompts never mention notes; persist-on appends structured `- (dia_id) speaker: fact` lines; `notes_only` hides `sessions/` after ingest and is the harness memory-method condition (not persist-on with the haystack still on disk). Campaign YAML `takeaways:` contrast harness reader vs Chat Completions reader vs frozen-reader writer (`delta` = left − right). Notebooks 17 pin Mem0 Table 2 / local_clone J and plot catalog **model year** separately from the 2026 Codex CLI harness. PoC `reader_vs_paper_j` is paper / local clone / gpt-4o-mini + Codex on `full_context` (persist-off workspace, not a second Chat Completions live clone); `methods_vs_paper_j` repeats that 3-bar for each Table 2 method and also reports LoCoMo F1 (session-summaries paper F1 is Maharana et al. 2024 Table 3 Summary RAG top-5, not Mem0 J). Mini-vs-Codex `methods_vs_full_context` is a ceiling overlay (`memory_lane` x `system_harness`): stuffed Chat Completions `full_context`, persist-off Codex workspace, and frozen-reader summaries/graph writers. It is not a ranking of memory systems. Missing RAG search stays empty, not 0; `full_context` search is 0.
2. **Orchestrator is software**, not one giant LLM call (`ModelOrchestrator`). The **experiment runner** (`src/experiment_runner`) only expands matrices and launches one locomo_eval run spec per task.
3. **Prefer small pure functions** over frameworks.
4. **Keep metrics dual-reported:** SPEC token F1/EM *and* LoCoMo category F1.
   Autorater reports additionally use Mem0 F1/BLEU-1/J and must label these
   separately; Mem0 category 5 is excluded from J and from design-level overall
   F1/EM (`source: runs` pack F1 is replaced with the cat-5-excluded example mean).
   Mock autorater output is plumbing-only (`mock_sanity_not_llm_judge`), must
   not occupy the literature J column, and must never log a live model id.
5. **Runs are self-contained.** A bare `locomo_eval.run` invocation still clears and regenerates. **`experiment_runner execute-qa` skips** when `_SUCCESS` exists (Cloud Run retries). `--force` regenerates. Do not add response stores or per-question resume. Shared Mem0/RAG indexes (`mem0_locomo10`, `rag_locomo10`) are built once and reused across reader run specs.
6. **Plain YAML**, plain JSON loaders, local CSV — no Hydra/W&B. Components live under `configs/{writers,readers,layouts,autoraters,teachers}/` and compose with `includes:` (later keys win). `pipeline.memory` is still a builder id, not a path to another YAML.
7. **Update docs:** after behavior change, copy previous AGENTS/HUMANS into `docs/agent/traces/YYYY-MM-DD_topic.md`, then edit live files. Traces stay local (gitignored); keep writing them anyway.
8. **Eval vs write split:** analysis of finished dumps imports `src.locomo_eval.experiment_pack.audit_loader` (and `audit_layout`). Do not add eval loops to `run.py` or import `teachers` / `experiment_pack.audit_writer` from an eval branch. On-disk contract: `docs/schemas/experiment_pack.md`.
9. **Published vs local.** The repo is shared with outside researchers, so anything tracked must be reproduction-relevant and account-agnostic. Traces, specs, `HUMANS.md`, `claim_audit_status.md`, notebooks, and `papers/` are gitignored — edit them freely, never `git add -f` them. The published notebook is `notebooks/17_openai_mini_codex_readers_analysis.ipynb`. The published run pack is `experiments/locomo-openai-mini-codex-readers-analysis-v2`. Full-context rows store each conversation's memory text and Codex task-prompt prefix once; see `REFERENCE_PIN.json`. Cloud ids come from `$env:PROJECT_ID` / `$MEMORYBENCH_BUCKET`; do not hardcode a project id, bucket, or service-account email into a tracked config, script, or doc.

---

## Code, tests, and comments

From review. Follow these when adding or renaming code.

**Names.** Unambiguous, justified, and kept current. If a name no longer matches the behavior, rename it (e.g. a generic orchestrator must not be called `run_baseline` when “baseline” is a config). Do not overload research terms across different mechanisms (audit JSONL vs a live store; baseline YAML vs `session_summaries`). Condition ids are descriptive (`raw_chunks`), not numbered (`C0`).

**Comments.** Explain *why* and the surrounding context, not a restatement of the next line. Ambiguous helpers need a one-liner on what they pin for later reproduction (`_git_hash`, `_file_sha256`). Distinguish lookalike layers (`offline_evaluate.py` string scoring vs `autorater.py` online LLM judging).

**Schemas / data-model classes.** The module docstring should map how types connect and which pipeline step uses them (load → memory → reader → prediction → score). Each class gets a short “what it is / who consumes it” note. Label gold answers as scorer-only (never in the reader prompt).

**Tests.**
- One unit test focuses on one function (`exact_match` tests stay separate from `token_f1` tests).
- Test names include the behavior **and** the expected outcome, e.g. `test_exact_match_returns_one_when_answers_match_after_normalization`.
- Group related cases in a `TestCase` per function or class; do not pile unrelated functions into one method.
- Claim-audit joins: lock sample/session isolation, retrieve-loser exclusion, and **optional** fields/filters (`None` = no restriction; missing layers = empty, not invented teachers). See `tests/test_claim_audit.py`, `tests/test_experiment_pack.py`, and `tests/test_regressions.py`.

**Experiments.** One locomo_eval YAML per `run_locomo_pipeline_with_memory_config` call. The harness (`configs/experiments/*.yaml`) expands many run specs and calls that once per `--run-index`. QA and autorater are separate jobs. Compare packs with `scripts/compare_full_runs.py` as before.

---

## Out of scope (v0.1)

Mem0 Platform / Neo4j / Qdrant, claiming paper Table 1–2 J from this OSS clone, mixing a reader-prompt swap into a graph/store claim, validator loop, retrieval budgets as experiments, training/distillation loop, web UI, event-summarization / multimodal tasks. Claim-level fusion and LLM validators are future work; current `resolve_*` policies are baseline heuristics only. A-Mem / LangMem / Zep / MemGPT are literature pins only.

---

## Acceptance checklist for agent PRs

- [ ] Dataset loads without editing source JSON  
- [ ] Default config remains Mem0-parity (GPT-4o-mini reader/judge, pinned prompts, released message/request shape); overrides do not mutate defaults
- [ ] One-question and full-run share the same command  
- [ ] Predictions JSONL deterministic fields  
- [ ] Metrics include EM, token F1, LoCoMo F1 by category  
- [ ] Autorater includes Mem0 F1/BLEU-1/J, category-5 exclusion, fresh tables, and fresh plots (never appends)
- [ ] Memory builder swappable without changing reader/evaluator  
- [ ] Tests for parse, memory, preprocess session blocks, session-document join, Mem0 index (mock), normalize  
- [ ] Model-integration sanity (`test_integration_sanity`: reader swap, teacher family swap, LoCoMo vs SPEC scorer)  
- [ ] `tests/test_autorater_sanity.py` stays green (mock only; no API)
- [ ] `tests/test_compare_to_paper.py` stays green (offline paper vs local J; no API)
- [ ] `tests/test_rag_index.py`, `tests/test_openai_memory.py`, `tests/test_stats.py`, `tests/test_eval_pipeline.py` stay green (mock only; no API)
- [ ] `tests/test_mem0_index.py`, `tests/test_regressions.py`, `tests/test_run_isolation.py`, `tests/test_model_orchestrator.py`, `tests/test_memory_writer.py`, and `tests/test_claim_audit.py` stay green (mock only; no API)
- [ ] `tests/test_experiment_runner_matrix.py`, `tests/test_experiment_runner_execute_qa.py`, and `tests/test_experiment_runner_aggregate.py` stay green (mock only)
- [ ] `tests/test_agent_harness.py` stays green (mock harness, trajectory failure modes, matrix skip)
- [ ] `tests/test_verify_experiments.py` stays green (offline pack verifier + graph year diagnosis)
- [ ] AGENTS.md + HUMANS.md updated + trace snapshot  

---

## Traceability

- Spec: `docs/agent/SPEC_v1.md`, `docs/agent/SPEC_v2.md`  
- LoCoMo pin: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`  
- Paper: Maharana et al., arXiv:2402.17753  
- Mem0 protocol: Chhikara et al., arXiv:2504.19413 (architecture clone; not Platform v2 numbers)
- Autorater protocol/baselines: Chhikara et al., arXiv:2504.19413
- Autorater prompt source: [Mem0 `ACCURACY_PROMPT` (pinned code)](https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py), also reproduced in paper Appendix A
