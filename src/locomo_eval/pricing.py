"""Pinned list prices for cost rollups (campaign analysis + per-run cost.json).

USD per 1M tokens from ``configs/models/pricing.yaml``. Unknown models stay
token-only. This is not an invoice.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICING_PATH = ROOT / "configs" / "models" / "pricing.yaml"


@dataclass(frozen=True)
class ModelRates:
    """What it is: list price for one API id. Who consumes it: estimate_usd."""

    input: float
    output: float
    peak_input: float | None = None
    peak_output: float | None = None


@dataclass(frozen=True)
class PricingTable:
    """Loaded pricing YAML. ``as_of`` is the pin date, not a live fetch."""

    as_of: str
    note: str
    models: dict[str, ModelRates]
    source_path: Path


def load_pricing(path: str | Path | None = None) -> PricingTable:
    dest = Path(path) if path is not None else DEFAULT_PRICING_PATH
    if not dest.is_file():
        raise FileNotFoundError(f"pricing YAML not found: {dest}")
    raw = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
    models: dict[str, ModelRates] = {}
    for key, block in (raw.get("models") or {}).items():
        if not isinstance(block, dict):
            continue
        models[str(key)] = ModelRates(
            input=float(block["input"]),
            output=float(block.get("output") or 0.0),
            peak_input=_opt_float(block.get("peak_input")),
            peak_output=_opt_float(block.get("peak_output")),
        )
    return PricingTable(
        as_of=str(raw.get("pricing_as_of") or ""),
        note=str(raw.get("note") or "").strip(),
        models=models,
        source_path=dest.resolve(),
    )


@lru_cache(maxsize=4)
def _cached_pricing(path_str: str) -> PricingTable:
    return load_pricing(path_str)


def default_pricing() -> PricingTable:
    return _cached_pricing(str(DEFAULT_PRICING_PATH.resolve()))


def rates_for(
    model: str | None,
    *,
    scenario: str = "off_peak",
    table: PricingTable | None = None,
) -> ModelRates | None:
    if not model:
        return None
    table = table or default_pricing()
    rates = table.models.get(str(model).strip())
    if rates is None:
        return None
    if scenario == "peak" and rates.peak_input is not None:
        return ModelRates(
            input=rates.peak_input,
            output=float(rates.peak_output if rates.peak_output is not None else rates.output),
        )
    return rates


def estimate_usd(
    model: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    scenario: str = "off_peak",
    table: PricingTable | None = None,
) -> float | None:
    """Pinned list-price USD; ``None`` if the model is unpriced."""
    rates = rates_for(model, scenario=scenario, table=table)
    if rates is None:
        return None
    return round(
        (prompt_tokens * rates.input + completion_tokens * rates.output) / 1_000_000.0,
        6,
    )


def add_costs(*values: float | None) -> float | None:
    """Sum known USD pins; ``None`` if every input was unpriced (not $0)."""
    known = [v for v in values if v is not None]
    if not known:
        return None
    return round(sum(known), 6)


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
