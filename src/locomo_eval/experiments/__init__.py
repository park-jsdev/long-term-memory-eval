"""On-disk sandwich audit: path contract, dump during a run, load for analysis.

A LoCoMo *run* is one YAML / ``--method`` producing ``experiments/<run_id>/``.
That directory is the sandwich audit (schema ``audit_pack.v2``): reader,
teachers, memory graph, claim lineage, and optional autorater mapped to
one condition's results.

```
conversation + question
        │
        ▼
  MemoryBuilder / TeacherOrchestrator     ← variable middle
        │
        ▼
  audit_writer  →  experiments/<run_id>/   ← dump (run.py, autorater)
        │
        ▼
  audit_loader  →  metrics / traces         ← analysis (compare scripts)
```

| Module | Role | Who imports it |
|--------|------|-----------------|
| ``audit_layout`` | Folder/file names only | both sides |
| ``audit_writer`` | Write reader / teacher / graph / judge files | ``run.py``, autorater CLI |
| ``audit_loader`` | Read a finished sandwich audit | ``scripts/analysis/``, compare CLIs |
| ``verify_pack`` / ``verify_graph_years`` | Offline validity from dumps | ``scripts/analysis/verify_experiments`` |

Eval code should not import ``audit_writer``, ``run``, or ``teachers``.
"""

from .audit_layout import AUDIT_LAYOUT_VERSION, AuditPaths, audit_layout_meta, teacher_dir_name
from .audit_loader import SandwichAudit, load_qa_pack, load_sandwich_audit, predictions_jsonl
from .verify_pack import PackReport, verify_pack

__all__ = [
    "AUDIT_LAYOUT_VERSION",
    "AuditPaths",
    "PackReport",
    "SandwichAudit",
    "audit_layout_meta",
    "load_qa_pack",
    "load_sandwich_audit",
    "predictions_jsonl",
    "teacher_dir_name",
    "verify_pack",
]
