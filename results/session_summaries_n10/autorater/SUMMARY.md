# Mem0 autorater benchmark

Judge protocol: Mem0 LLM-as-a-Judge (Chhikara et al., arXiv:2504.19413).
Paper: https://arxiv.org/abs/2504.19413
Prompt source: https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py

This pack scores **already generated** answers. It does not run Mem0
extract/update. Category 5 (adversarial) is skipped, matching Mem0.
It reads the source prediction pack but does not modify its files.
Evaluation caches are prohibited. Cached source rows are rejected, and
each invocation clears prior analysis and regenerates from source predictions.

## This run

- method: `session_summaries`
- autorater_provider: `openai`
- autorater_model: `gpt-4o-mini`
- Live LLM judge result; comparable only when protocol/model/subset match.
- n_judged: 10 (skipped category 5: 0)
- Mem0 F1: **26.08** (paper scale, %)
- BLEU-1: **17.29**
- LLM-as-a-Judge J: **80.0**
- LoCoMo F1 (judged subset): 0.3136
- reader latency p50/p95 (s): 1.608 / 4.6799
- autorater latency p50/p95 (s): 0.6625 / 1.1221
- mean reader prompt tokens: 3918.8
- total reader tokens (stored API usage): 39384
- total autorater tokens: 4035
- mean memory chars: 20863.0

## By category

- temporal (n=6): F1=26.44, B1=17.61, J=83.33
- open_domain (n=1): F1=14.29, B1=9.09, J=100.0
- multi_hop (n=3): F1=29.3, B1=19.39, J=66.67

## Literature pins (not a re-run of those systems)

Compare `tables/vs_literature.csv` to Chhikara et al., arXiv:2504.19413 Table 2.
Headline published overall J: Mem0=66.88, Mem0g=68.44, Zep=65.99,
full-context=72.90, OpenAI=52.90, best RAG (k=2, 256)=60.97.

## Tables

- overall: `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\tables\overall.csv`
- by_category: `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\tables\by_category.csv`
- vs_literature: `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\tables\vs_literature.csv`
- vs_literature_by_category: `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\tables\vs_literature_by_category.csv`
- per_question: `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\tables\per_question.csv`

## Plots

- `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\plots\judge_verdict_barplot.png`
- `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\plots\mem0_f1_histogram.png`
- `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\plots\lexical_scores_boxplot.png`
- `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\plots\reader_latency_histogram.png`
- `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\plots\latency_boxplot.png`
- `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\plots\by_category_bars.png`
- `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\plots\j_vs_literature.png`
- `C:\home\research\Distillation\distillation\experiments\session_summaries_n10\autorater\plots\latency_vs_literature.png`
