"""Reusable experiment analyses.

Figure 1 (single run): predicted + reference → LoCoMo string scorer → metrics.
  That lives in ``src/locomo_eval/metrics.py`` / ``offline_evaluate.py``.

Figure 2 (two runs): experimental output A + B → ``compare_predictions`` →
  paired LoCoMo F1 plots. Reader / teacher / memory / rater are *which* two
  packs you pass in, not extra diagram nodes.

Dataset histograms / naive-retrieval bars: ``dataset_stats`` (implementation
in ``src.locomo_eval.preprocess.dataset_stats``; CLI via
``scripts/export_session_documents.py``).

Online benchmark: finished predictions → ``run_benchmark`` → Mem0 lexical
metrics + LLM autorater + literature/latency tables and plots.
"""
