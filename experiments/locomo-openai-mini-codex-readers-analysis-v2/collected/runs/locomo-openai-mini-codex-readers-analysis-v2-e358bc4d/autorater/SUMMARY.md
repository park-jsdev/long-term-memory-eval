# Mem0 autorater benchmark

Judge protocol: Mem0 LLM-as-a-Judge (Chhikara et al., arXiv:2504.19413).
Paper: https://arxiv.org/abs/2504.19413
Prompt source: https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py

This pack scores **already generated** answers. It does not run Mem0
extract/update. Category 5 (adversarial) is skipped, matching Mem0.
It reads the source prediction pack but does not modify its files.
Each invocation clears prior analysis and regenerates from source predictions.

## This run

- method: `session_summaries`
- autorater_provider: `openai`
- autorater_model: `gpt-4o-mini`
- Live LLM judge result; comparable only when protocol/model/subset match.
- n_judged: 1540 (skipped category 5: 446)
- Mem0 F1: **31.08** (paper scale, %)
- BLEU-1: **24.92**
- LLM-as-a-Judge J: **49.03**
- LoCoMo F1 (judged subset): 0.314
- reader latency p50/p95 (s): 1.6874 / 2.4516
- autorater latency p50/p95 (s): 0.523 / 0.6691
- mean reader prompt tokens: 11343.4468
- total reader tokens (stored API usage): 17484047
- total autorater tokens: 610229
- mean memory chars: 18444.4104

## By category

- temporal (n=321): F1=34.6, B1=27.48, J=32.4
- open_domain (n=96): F1=19.61, B1=14.83, J=41.67
- multi_hop (n=282): F1=29.38, B1=21.17, J=52.13
- single_hop (n=841): F1=31.62, B1=26.36, J=55.17

## Literature pins (not a re-run of those systems)

Compare `tables/vs_literature.csv` to Chhikara et al., arXiv:2504.19413 Table 2.
Headline published overall J: Mem0=66.88, Mem0g=68.44, Zep=65.99,
full-context=72.90, OpenAI=52.90, best RAG (k=2, 256)=60.97.

## Tables

- overall: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/tables/overall.csv`
- by_category: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/tables/by_category.csv`
- vs_literature: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/tables/vs_literature.csv`
- vs_literature_by_category: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/tables/vs_literature_by_category.csv`
- per_question: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/tables/per_question.csv`

## Plots

- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/plots/judge_verdict_barplot.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/plots/mem0_f1_histogram.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/plots/lexical_scores_boxplot.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/plots/reader_latency_histogram.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/plots/latency_boxplot.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/plots/by_category_bars.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/plots/j_vs_literature.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-e358bc4d/autorater/plots/latency_vs_literature.png`
