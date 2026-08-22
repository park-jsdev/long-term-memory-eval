"""Reusable offline analysis helpers (no API, not an LLM autorater).

Figure 1 (single run): predicted + reference → LoCoMo string scorer → metrics.
  That lives in ``src/locomo_eval/metrics.py`` / ``offline_evaluate.py``.

Figure 2 (two runs): experimental output A + B → ``compare_predictions`` →
  paired LoCoMo F1 plots. Reader / teacher / memory / rater are *which* two
  packs you pass in, not extra diagram nodes.
"""
