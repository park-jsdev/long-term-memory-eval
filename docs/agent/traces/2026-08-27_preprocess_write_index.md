# Trace: 2026-08-27 — before deterministic preprocess write-index

Preprocess dump existed as `write_conversation_run_log` (sessions.jsonl only)
and was unwired from run.py. raw_chunks / session_summaries always built from
Conversation. Live files add `python -m src.locomo_eval.preprocess.run_index`
(full-dataset, no LLM) plus dump-backed retrieve/format and `--eval-questions`.
