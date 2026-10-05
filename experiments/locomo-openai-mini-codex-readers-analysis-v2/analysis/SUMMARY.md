# experiment readers (locomo-openai-mini-codex-readers-analysis-v2)

All six cells render qa_mem0_v1 with the same memory text and question. Model-only and Codex persist-off share that payload exactly. Codex persist-on appends a harness-only note instruction and may reuse evidence-grounded notes; payload and final-task hashes stay separate. Prompt-delivered evidence is not workspace retrieval. Web search and user MCP are disabled or ignored and must remain zero in audit metrics.

Config: `C:/home/research/Distillation/distillation/configs/analysis/campaign_openai_mini_codex_readers_analysis.yaml`

## Takeaways — harness vs model reader/writer

table: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/tables/takeaways.csv`

### Full context — model vs Codex
claim: `full_context_harness_delta`

The model and persist-off Codex receive the same rendered reader payload. The remaining difference is Chat Completions versus Codex orchestration.

| left_label | right_label | metric | n_left | n_right | left_value | right_value | delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| model_full_context | codex_full_context_persist_off | judge_score | 1540 | 1540 | 0.7474 | 0.7409 | 0.0065 |
| model_full_context | codex_full_context_persist_off | locomo_f1 | 1540 | 1540 | 0.5383 | 0.4382 | 0.1001 |

### Session summaries — model vs Codex
claim: `summaries_harness_delta`

The LoCoMo summaries and question are identical reader payloads before either system answers.

| left_label | right_label | metric | n_left | n_right | left_value | right_value | delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| model_session_summaries | codex_session_summaries_persist_off | judge_score | 1540 | 1540 | 0.4883 | 0.4903 | -0.0019 |
| model_session_summaries | codex_session_summaries_persist_off | locomo_f1 | 1540 | 1540 | 0.3441 | 0.314 | 0.0301 |

### Persisted Codex is a stateful reader condition
claim: `persisted_agent_state`

Persist-on adds an explicitly logged Codex note instruction after the shared reader payload. It is an auditable stateful-agent condition, not exact end-to-end request parity.

## reader_conditions — Matched reader payload conditions
table: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/tables/reader_conditions.csv`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/reader_conditions_memory_method_prompt_parity_condition_judge_score_grouped_bar.png`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/reader_conditions_memory_method_prompt_parity_condition_locomo_f1_grouped_bar.png`

| memory_method | prompt_parity_condition | n | judge_score | locomo_f1 | token_f1 | exact_match | answer_n_words |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_context | 4o-mini | 1540 | 0.7474 | 0.5383 | 0.5319 | 0.211 | 4.0994 |
| full_context | 4o-mini + Codex (persist off) | 1540 | 0.7409 | 0.4382 | 0.4336 | 0.1188 | 6.3721 |
| full_context | 4o-mini + Codex (persist on) | 1540 | 0.7636 | 0.3367 | 0.3304 | 0.0338 | 14.1221 |
| session_summaries | 4o-mini | 1540 | 0.4883 | 0.3441 | 0.3359 | 0.1143 | 4.013 |
| session_summaries | 4o-mini + Codex (persist off) | 1540 | 0.4903 | 0.314 | 0.308 | 0.0883 | 5.1442 |
| session_summaries | 4o-mini + Codex (persist on) | 1540 | 0.5149 | 0.2407 | 0.2344 | 0.0234 | 9.2844 |

## reader_by_category — Matched payload by LoCoMo category
table: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/tables/reader_by_category.csv`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/reader_by_category_question_category_prompt_parity_condition_judge_score_grouped_bar.png`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/reader_by_category_question_category_prompt_parity_condition_locomo_f1_grouped_bar.png`

| question_category | memory_method | prompt_parity_condition | n | locomo_f1 | judge_score |
| --- | --- | --- | --- | --- | --- |
| 1 multi-hop | full_context | 4o-mini | 282 | 0.3785 | 0.6596 |
| 1 multi-hop | full_context | 4o-mini + Codex (persist off) | 282 | 0.3127 | 0.6596 |
| 1 multi-hop | full_context | 4o-mini + Codex (persist on) | 282 | 0.2663 | 0.7234 |
| 2 temporal | full_context | 4o-mini | 321 | 0.516 | 0.5576 |
| 2 temporal | full_context | 4o-mini + Codex (persist off) | 321 | 0.3664 | 0.5452 |
| 2 temporal | full_context | 4o-mini + Codex (persist on) | 321 | 0.2847 | 0.5265 |
| 3 open-domain | full_context | 4o-mini | 96 | 0.2453 | 0.4792 |
| 3 open-domain | full_context | 4o-mini + Codex (persist off) | 96 | 0.2216 | 0.5312 |
| 3 open-domain | full_context | 4o-mini + Codex (persist on) | 96 | 0.1505 | 0.5625 |
| 4 single-hop | full_context | 4o-mini | 841 | 0.6339 | 0.8799 |
| 4 single-hop | full_context | 4o-mini + Codex (persist off) | 841 | 0.5324 | 0.8668 |
| 4 single-hop | full_context | 4o-mini + Codex (persist on) | 841 | 0.4014 | 0.8906 |
| 5 adversarial | full_context | 4o-mini | 446 | 0.0202 | 0 |
| 5 adversarial | full_context | 4o-mini + Codex (persist off) | 446 | 0.0067 | 0 |
| 5 adversarial | full_context | 4o-mini + Codex (persist on) | 446 | 0.0112 | 0 |
| 1 multi-hop | session_summaries | 4o-mini | 282 | 0.3016 | 0.4787 |
| 1 multi-hop | session_summaries | 4o-mini + Codex (persist off) | 282 | 0.2837 | 0.5213 |
| 1 multi-hop | session_summaries | 4o-mini + Codex (persist on) | 282 | 0.2295 | 0.5461 |
| 2 temporal | session_summaries | 4o-mini | 321 | 0.3839 | 0.3614 |
| 2 temporal | session_summaries | 4o-mini + Codex (persist off) | 321 | 0.3536 | 0.324 |
| 2 temporal | session_summaries | 4o-mini + Codex (persist on) | 321 | 0.2553 | 0.3489 |
| 3 open-domain | session_summaries | 4o-mini | 96 | 0.1961 | 0.4271 |
| 3 open-domain | session_summaries | 4o-mini + Codex (persist off) | 96 | 0.2061 | 0.4167 |
| 3 open-domain | session_summaries | 4o-mini + Codex (persist on) | 96 | 0.1603 | 0.4792 |
| 4 single-hop | session_summaries | 4o-mini | 841 | 0.3601 | 0.547 |
| 4 single-hop | session_summaries | 4o-mini + Codex (persist off) | 841 | 0.3214 | 0.5517 |
| 4 single-hop | session_summaries | 4o-mini + Codex (persist on) | 841 | 0.248 | 0.5719 |
| 5 adversarial | session_summaries | 4o-mini | 446 | 0.1278 | 0 |
| 5 adversarial | session_summaries | 4o-mini + Codex (persist off) | 446 | 0.0202 | 0 |
| 5 adversarial | session_summaries | 4o-mini + Codex (persist on) | 446 | 0.0426 | 0 |

## reader_performance — Reader latency and list-price cost
table: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/tables/reader_performance.csv`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/reader_performance_memory_method_prompt_parity_condition_total_latency_seconds_p50_grouped_bar.png`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/reader_performance_memory_method_prompt_parity_condition_reader_usd_grouped_bar.png`

| memory_method | prompt_parity_condition | n | total_latency_seconds_p50 | total_latency_seconds_p95 | agent_latency_seconds_p50 | agent_input_tokens | agent_output_tokens | reader_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_context | 4o-mini | 1540 | 0.6505 | 1.5561 | 0.6505 | 27840.6883 | 6.1325 | 0.0042 |
| full_context | 4o-mini + Codex (persist off) | 1540 | 2.06 | 7.9825 | 2.06 | 34996.6883 | 10.9448 | 0.0053 |
| full_context | 4o-mini + Codex (persist on) | 1540 | 2.2518 | 8.5688 | 2.2518 | 36132.2825 | 23.4071 | 0.0054 |
| session_summaries | 4o-mini | 1540 | 0.527 | 0.803 | 0.527 | 4071.9539 | 5.7987 | 0.0006 |
| session_summaries | 4o-mini + Codex (persist off) | 1540 | 1.6874 | 2.4516 | 1.6874 | 11343.4468 | 9.8305 | 0.0017 |
| session_summaries | 4o-mini + Codex (persist on) | 1540 | 1.7805 | 2.8257 | 1.7805 | 12047.4766 | 17.5896 | 0.0018 |

## gold_token_stages — Gold-token coverage and answer retention
table: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/tables/gold_token_stages.csv`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/gold_token_stages_memory_method_prompt_parity_condition_gold_memory_recall_grouped_bar.png`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/gold_token_stages_memory_method_prompt_parity_condition_gold_answer_recall_grouped_bar.png`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/gold_token_stages_memory_method_prompt_parity_condition_gold_answer_precision_grouped_bar.png`

| memory_method | prompt_parity_condition | n | gold_memory_recall | gold_answer_recall | gold_answer_precision | gold_kept_given_memory | gold_dropped_given_memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_context | 4o-mini | 1540 | 0.9283 | 0.5947 | 0.5773 | 0.613 | 0.387 |
| full_context | 4o-mini + Codex (persist off) | 1540 | 0.9283 | 0.5886 | 0.427 | 0.6113 | 0.3887 |
| full_context | 4o-mini + Codex (persist on) | 1540 | 0.9283 | 0.6294 | 0.2799 | 0.6549 | 0.3451 |
| session_summaries | 4o-mini | 1540 | 0.7335 | 0.3753 | 0.3832 | 0.44 | 0.56 |
| session_summaries | 4o-mini + Codex (persist off) | 1540 | 0.7335 | 0.384 | 0.335 | 0.4556 | 0.5444 |
| session_summaries | 4o-mini + Codex (persist on) | 1540 | 0.7335 | 0.3988 | 0.2127 | 0.4735 | 0.5265 |

## gold_token_by_category — Gold-token prompt coverage by category
table: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/tables/gold_token_by_category.csv`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/gold_token_by_category.png`

| question_category | memory_method | n | gold_memory_recall |
| --- | --- | --- | --- |
| 1 multi-hop | full_context | 282 | 0.9213 |
| 2 temporal | full_context | 321 | 0.8645 |
| 3 open-domain | full_context | 96 | 0.6281 |
| 4 single-hop | full_context | 841 | 0.9893 |
| 1 multi-hop | session_summaries | 282 | 0.7004 |
| 2 temporal | session_summaries | 321 | 0.7822 |
| 3 open-domain | session_summaries | 96 | 0.3793 |
| 4 single-hop | session_summaries | 841 | 0.7664 |

## adversarial_refusal — Adversarial refusal and false refusal
table: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/tables/adversarial_refusal.csv`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/adversarial_refusal_memory_method_prompt_parity_condition_adversarial_refusal_grouped_bar.png`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/adversarial_refusal_memory_method_prompt_parity_condition_false_refusal_grouped_bar.png`

| memory_method | prompt_parity_condition | n | adversarial_refusal | adversarial_refusal_n | false_refusal | false_refusal_n |
| --- | --- | --- | --- | --- | --- | --- |
| full_context | 4o-mini | 1986 | 0.0202 | 446 | 0.0026 | 1540 |
| full_context | 4o-mini + Codex (persist off) | 1986 | 0.0067 | 446 | 0.0013 | 1540 |
| full_context | 4o-mini + Codex (persist on) | 1986 | 0.0112 | 446 | 0.0006 | 1540 |
| session_summaries | 4o-mini | 1986 | 0.1278 | 446 | 0.0468 | 1540 |
| session_summaries | 4o-mini + Codex (persist off) | 1986 | 0.0202 | 446 | 0.0097 | 1540 |
| session_summaries | 4o-mini + Codex (persist on) | 1986 | 0.0426 | 446 | 0.0078 | 1540 |

## codex_audit — Codex tool and persistence audit
table: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/tables/codex_audit.csv`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/codex_audit_memory_method_agent_persist_n_web_search_grouped_bar.png`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/codex_audit_memory_method_agent_persist_n_mcp_grouped_bar.png`
plot: `C:/home/research/Distillation/distillation/experiments/locomo-openai-mini-codex-readers-analysis-v2/analysis/plots/codex_audit_memory_method_agent_persist_n_write_events_grouped_bar.png`

| memory_method | agent_persist | n | n_web_search | n_mcp | used_non_workspace_tools | n_write_events | notes_retrieved | notes_bytes | harness_failed_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_context | false | 1 | 0 | 0 | 0 | 0 | 0 |  | 0 |
| full_context | true | 1 | 0 | 0 | 0 | 0.0191 | 0.002 | 268.3087 | 0 |
| session_summaries | false | 1 | 0 | 0 | 0 | 0 | 0 |  | 0 |
| session_summaries | true | 1 | 0 | 0 | 0 | 0.0131 | 0.0161 | 204.929 | 0 |
