"""Load LoCoMo JSON into Conversation objects without mutating the source file.

Wraps dataset.py so HLD (i) has a named data ingestor. Does not fork session parsing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..dataset import load_conversations, load_raw, parse_sample
from ..schemas import Conversation


class DataIngestor:
    """Read locomo10.json (or a fixture) into Conversation objects.

    Source JSON stays in data/raw/. This class never writes that path.
    """

    def load(self, path: str | Path) -> list[Conversation]:
        return load_conversations(path)

    def ingest_sample(self, sample: dict[str, Any]) -> Conversation:
        return parse_sample(sample)

    def load_raw(self, path: str | Path) -> list[dict[str, Any]]:
        """Raw dicts only — for tests that check the file was not rewritten."""
        return load_raw(path)
