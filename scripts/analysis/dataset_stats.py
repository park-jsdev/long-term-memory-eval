"""CLI wrapper. Prefer: python scripts/export_session_documents.py

Implementation lives in src.locomo_eval.preprocess.dataset_stats so tests do
not import the scripts/ folder (conda often shadows a package named scripts).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.locomo_eval.preprocess.dataset_stats import (  # noqa: E402
    retrieval_report,
    write_dataset_analysis,
)

__all__ = ["retrieval_report", "write_dataset_analysis"]
