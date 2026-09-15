"""Compat shim. Prefer ``src.locomo_eval.experiments.audit_writer`` / ``.audit_layout``."""

from .experiments.audit_layout import AUDIT_LAYOUT_VERSION, audit_layout_meta, teacher_dir_name
from .experiments.audit_writer import (
    audit_graph_dict,
    write_autorater_traces,
    write_claim_audit,
    write_frozen_config,
    write_graph_module,
    write_reader_module,
    write_teacher_module,
)

__all__ = [
    "AUDIT_LAYOUT_VERSION",
    "audit_graph_dict",
    "audit_layout_meta",
    "teacher_dir_name",
    "write_autorater_traces",
    "write_claim_audit",
    "write_frozen_config",
    "write_graph_module",
    "write_reader_module",
    "write_teacher_module",
]
