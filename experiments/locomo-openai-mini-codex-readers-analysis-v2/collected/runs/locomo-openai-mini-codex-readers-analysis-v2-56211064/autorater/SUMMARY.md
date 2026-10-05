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
- Mem0 F1: **33.9** (paper scale, %)
- BLEU-1: **27.85**
- LLM-as-a-Judge J: **48.83**
- LoCoMo F1 (judged subset): 0.3441
- reader latency p50/p95 (s): 0.527 / 0.803
- autorater latency p50/p95 (s): 0.522 / 0.6631
- mean reader prompt tokens: 4071.9539
- total reader tokens (stored API usage): 6279739
- total autorater tokens: 608156
- mean memory chars: 18444.4104

## By category

- temporal (n=321): F1=37.99, B1=31.53, J=36.14
- open_domain (n=96): F1=18.93, B1=13.63, J=42.71
- multi_hop (n=282): F1=30.35, B1=21.78, J=47.87
- single_hop (n=841): F1=35.23, B1=30.1, J=54.7

## Literature pins (not a re-run of those systems)

Compare `tables/vs_literature.csv` to Chhikara et al., arXiv:2504.19413 Table 2.
Headline published overall J: Mem0=66.88, Mem0g=68.44, Zep=65.99,
full-context=72.90, OpenAI=52.90, best RAG (k=2, 256)=60.97.

## Tables

- overall: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/tables/overall.csv`
- by_category: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/tables/by_category.csv`
- vs_literature: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/tables/vs_literature.csv`
- vs_literature_by_category: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/tables/vs_literature_by_category.csv`
- per_question: `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/tables/per_question.csv`

## Plots

- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/plots/judge_verdict_barplot.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/plots/mem0_f1_histogram.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/plots/lexical_scores_boxplot.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/plots/reader_latency_histogram.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/plots/latency_boxplot.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/plots/by_category_bars.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/plots/j_vs_literature.png`
- `/tmp/memorybench-experiments/locomo-openai-mini-codex-readers-analysis-v2-56211064/autorater/plots/latency_vs_literature.png`
