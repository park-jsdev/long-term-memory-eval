"""Load a standalone experiment YAML (optional model catalog for reader ids)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config import load_config
from src.memorybench.experiment_types import require_known_type

ROOT = Path(__file__).resolve().parents[2]


def load_experiment_yaml(path: str | Path) -> dict[str, Any]:
    cfg = load_config(path)
    if "experiment" not in cfg:
        raise ValueError(f"Missing experiment: block in {path}")
    require_known_type(cfg["experiment"].get("type") or "sweep")
    catalog_rel = cfg.get("model_catalog")
    if catalog_rel:
        catalog_path = Path(catalog_rel)
        if not catalog_path.is_file():
            catalog_path = ROOT / catalog_rel
        cfg["_model_catalog"] = _load_catalog(catalog_path)
    else:
        cfg["_model_catalog"] = {}
    cfg["_source_path"] = str(Path(path))
    return cfg


def _load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    models = raw.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError(f"model catalog must be a mapping under models: ({path})")
    return {str(key): dict(value) for key, value in models.items()}
