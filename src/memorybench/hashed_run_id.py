"""Deterministic hashed run ids. No timestamps.

Form: ``<experiment>-<8 hex chars>`` from a canonical JSON of QA identity.
Retries of the same scientific cell reuse the same directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def short_config_hash(obj: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:8]


def slug_experiment_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(name).strip()).strip("-").lower()
    return slug or "experiment"


def hashed_run_id(experiment_name: str, identity: dict[str, Any]) -> str:
    return f"{slug_experiment_name(experiment_name)}-{short_config_hash(identity)}"
