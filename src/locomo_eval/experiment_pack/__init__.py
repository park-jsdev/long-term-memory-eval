"""On-disk sandwich pack: write during a run, load for analysis, verify offline.

The output folder stays ``experiments/<run_id>/``. This package is the
code that writes and reads that folder. Only ``verify_pack`` and
``verify_graph_years`` check a finished pack; the rest dump and load it.
The old name ``experiments`` collided with that folder and with
``configs/experiments/``.

A LoCoMo *run* is one YAML / ``--method`` producing ``experiments/<run_id>/``.
That directory is the sandwich audit (schema ``audit_pack.v3``): reader,
writer, memory graph, claim lineage, and optional autorater mapped to
one condition's results.

```
conversation + question
        │
        ▼
  MemoryBuilder / ModelOrchestrator        ← variable middle
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
| ``audit_writer`` | Write reader / writer / graph / judge files | ``run.py``, autorater CLI |
| ``audit_loader`` | Read a finished sandwich audit | ``scripts/analysis/``, compare CLIs |
| ``verify_pack`` / ``verify_graph_years`` | Offline validity from dumps | ``scripts/analysis/verify_experiments`` |

Eval code should not import ``audit_writer``, ``run``, or the writer model.
"""

from .audit_layout import AUDIT_LAYOUT_VERSION, AuditPaths, audit_layout_meta, writer_dir_name
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
    "writer_dir_name",
    "verify_pack",
]
