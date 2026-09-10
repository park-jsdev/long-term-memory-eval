"""Thin experiment harness around ``locomo_eval``.

This package expands a YAML matrix, assigns hashed run ids, and executes
one LoCoMo QA or autorater job. Scientific logic (memory builders, readers,
metrics, Mem0 clone) stays in ``src.locomo_eval``.
"""

from .experiment_run_spec import ExperimentRunSpec, ReaderModelRef

__all__ = ["ExperimentRunSpec", "ReaderModelRef"]
