"""Load one YAML file as a dict. Optional ``includes:`` deep-merges other files first.

Not Hydra: no interpolation, no defaults list, no overrides CLI. Later keys in the
current file win. Nested mappings merge; lists and scalars replace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_yaml(Path(path), stack=())


def list_config_chain(path: str | Path) -> list[Path]:
    """YAML files merged for this config: included files first, then the entry file.

    Order matches ``load_config`` (depth-first includes, later keys win).
    """
    return _list_chain(Path(path), stack=())


def deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)
        for key, value in overlay.items():
            if key in out:
                out[key] = deep_merge(out[key], value)
            else:
                out[key] = value
        return out
    return overlay


def _load_yaml(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(p) for p in stack + (path,))
        raise ValueError(f"Config include cycle: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    includes = cfg.get("includes") or []
    if includes and not isinstance(includes, list):
        raise ValueError(f"includes must be a list: {path}")
    merged: dict[str, Any] = {}
    for rel in includes:
        merged = deep_merge(merged, _load_yaml(_resolve_include(path, str(rel)), stack + (path,)))
    local = {k: v for k, v in cfg.items() if k != "includes"}
    return deep_merge(merged, local)


def _list_chain(path: Path, stack: tuple[Path, ...]) -> list[Path]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(p) for p in stack + (path,))
        raise ValueError(f"Config include cycle: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    includes = cfg.get("includes") or []
    if includes and not isinstance(includes, list):
        raise ValueError(f"includes must be a list: {path}")
    chain: list[Path] = []
    for rel in includes:
        chain.extend(_list_chain(_resolve_include(path, str(rel)), stack + (path,)))
    chain.append(path)
    return chain


def _resolve_include(base_file: Path, rel: str) -> Path:
    candidate = Path(rel)
    if candidate.is_absolute():
        return candidate
    from_root = ROOT / rel
    if from_root.is_file():
        return from_root
    return (base_file.parent / rel).resolve()
