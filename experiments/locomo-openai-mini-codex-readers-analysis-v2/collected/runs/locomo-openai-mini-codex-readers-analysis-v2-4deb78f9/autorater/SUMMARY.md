# Mem0 autorater benchmark

Judge protocol: Mem0 LLM-as-a-Judge (Chhikara et al., arXiv:2504.19413).
Paper: https://arxiv.org/abs/2504.19413
Prompt source: https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py

This pack scores **already generated** answers. It does not run Mem0
extract/update. Category 5 (adversarial) is skipped, matching Mem0.
It reads the source prediction pack but does not modify its files.
Each invocation clears prior analysis and regenerates from source predictions.

## This run

- method: `full_context`
- autorater_provider: `openai`
- autorater_model: `gpt-4o-mini`
- Live LLM judge result; comparable only when protocol/model/subset match.
- n_judged: 1540 (skipped category 5: 446)
- Mem0 F1: **43.59** (paper scale, %)
- BLEU-1: **35.74**
- LLM-as-a-Judge J: **74.09**
- LoCoMo F1 (judged subset): 0.4382
- reader latency p50/p95 (s): 2.06 / 7.9825
- autorater latency p50/p95 (s): 0.516 / 0.658
- mean reader prompt tokens: 34996.6883
- total reader tokens (stored API usage): 53911755
- total autorater tokens: 612558
- mean memory chars: 104442.8169

## By category

- temporal (n=321): F1=35.67, B1=28.48, J=54.52
- open_domain (n=96): F1=21.5, B1=16.0, J=53.12
- multi_hop (n=282): F1=33.56, B1=25.19, J=65.96
- single_hop (n=841): F1=52.49, B1=44.3, J=86.68

## Literature pins (not a re-run of those systems)

Compare `tables/vs_literature.csv` to Chhikara et al., arXiv:2504.19413 Table 2.
Headline published overall J: Mem0=66.88, Mem0g=68.44, Zep=65.99,
full-context=72.90, OpenAI=52.90, best RAG (k=2, 256)=60.97.

## Tables

- overall: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/tables/overall.csv`
- by_category: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/tables/by_category.csv`
- vs_literature: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/tables/vs_literature.csv`
- vs_literature_by_category: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/tables/vs_literature_by_category.csv`
- per_question: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/tables/per_question.csv`

## Plots

- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/plots/judge_verdict_barplot.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/plots/mem0_f1_histogram.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/plots/lexical_scores_boxplot.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/plots/reader_latency_histogram.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/plots/latency_boxplot.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/plots/by_category_bars.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/plots/j_vs_literature.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-4deb78f9/autorater/plots/latency_vs_literature.png`
