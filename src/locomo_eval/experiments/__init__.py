"""Experiment pack I/O split so eval can land in another branch.

- ``layout`` / ``load``: read-only contract for ``experiments/<run_id>/``
- ``write``: used only by ``run.py`` and the autorater CLI

Eval code should not import ``write``, ``run``, or ``teachers``.
"""

from .layout import AUDIT_LAYOUT_VERSION, PackPaths, audit_layout_meta, teacher_dir_name
from .load import ExperimentPack, load_experiment_pack, load_qa_pack, predictions_jsonl

__all__ = [
    "AUDIT_LAYOUT_VERSION",
    "ExperimentPack",
    "PackPaths",
    "audit_layout_meta",
    "load_experiment_pack",
    "load_qa_pack",
    "predictions_jsonl",
    "teacher_dir_name",
]
