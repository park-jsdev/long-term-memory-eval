"""Build LocalObjectStore or GCSObjectStore from experiment YAML storage:."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.memorybench.gcs_object_store import GCSObjectStore
from src.memorybench.local_object_store import LocalObjectStore

ROOT = Path(__file__).resolve().parents[2]


def open_configured_store(cfg: dict[str, Any]):
    storage = cfg.get("storage") or {}
    backend = str(storage.get("backend") or "local").lower()
    if backend == "local":
        path = storage.get("path") or "experiments"
        root = Path(path)
        if not root.is_absolute():
            root = ROOT / root
        return LocalObjectStore(root)
    if backend == "gcs":
        bucket = (
            os.environ.get("MEMORYBENCH_BUCKET")
            or storage.get("bucket")
        )
        if not bucket:
            raise ValueError(
                "storage.bucket or MEMORYBENCH_BUCKET is required when backend is gcs"
            )
        prefix = str(storage.get("prefix") or "")
        return GCSObjectStore(bucket=str(bucket), prefix=prefix)
    raise ValueError(f"Unknown storage.backend {backend!r} (use local or gcs)")


def local_experiments_root(cfg: dict[str, Any]) -> Path:
    """Directory locomo_eval uses as ``--output-dir`` (shared indexes live here too)."""
    storage = cfg.get("storage") or {}
    path = storage.get("path") or storage.get("local_path") or "experiments"
    root = Path(path)
    if not root.is_absolute():
        root = ROOT / root
    return root
