"""Experiment runner around ``locomo_eval``.

Expands a YAML matrix into hashed run ids and launches one eval cell per
task. A cell still answers questions one at a time. Parallelism is
independent cells, not a faster model. Scoring stays in ``src.locomo_eval``.
"""

from .experiment_run_spec import ExperimentRunSpec, ReaderModelRef

__all__ = ["ExperimentRunSpec", "ReaderModelRef"]
