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

Paper vs local J (offline): finished autorater packs → ``compare_to_paper`` →
grouped Table 2 bars using the best live seed per method.

Campaign / experiment reports (offline): YAML in ``configs/analysis/`` names
the table and plot; ``campaign_tables`` / ``campaign_plots`` draw them.
``python -m src.memorybench report configs/analysis/<campaign>.yaml``.

Context window vs coverage (offline, not a job): billed reader input against
published windows + memory-method coverage vs J/latency/USD.
``python -m scripts.analysis.context_window`` (notebook 16).

Pack validity (offline, not a job): TRACE / prompt snapshots / quality.json /
graph dumps. ``python -m scripts.analysis.verify_experiments``. ``--graph-years``
labels technical vs scientific reasons teacher_graph did not move.
"""
