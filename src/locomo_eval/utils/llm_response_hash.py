"""Content-addressed disk memo of **LLM** responses (not run checkpoints).

Keyed by ``llm_request_hash`` of the request (stage, provider, model,
sampling, prompt), not by run or question id.

This utility is retained for historical experiments, but response caching is
prohibited in evaluation. Reader, teacher, and autorater factories reject a
non-null store.

Call sites must put ``pipeline_stage`` in the key payload so stages cannot collide:

  - ``answer_reader`` — historical answer-stage key
  - ``teacher`` — historical teacher-stage key
  - ``autorater`` — historical judge-stage key

Not this store:
  - Run audit output: ``experiments/<run_id>/predictions.jsonl``
  - Offline string metrics: ``offline_evaluate.py`` (no LLM)
  - Distinctness sanity: ``llm_request_hash`` / compare scripts (no disk)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm_request_hash import (
    PIPELINE_STAGE_ANSWER_READER,
    PIPELINE_STAGE_AUTORATER,
    PIPELINE_STAGE_TEACHER,
)

__all__ = [
    "DEFAULT_LLM_RESPONSE_HASH_DIR",
    "LlmResponseHash",
    "PIPELINE_STAGE_ANSWER_READER",
    "PIPELINE_STAGE_AUTORATER",
    "PIPELINE_STAGE_TEACHER",
    "llm_response_hash_dir_from_run_cfg",
]


DEFAULT_LLM_RESPONSE_HASH_DIR = "experiments/cache"


def llm_response_hash_dir_from_run_cfg(run_cfg: dict | None) -> str:
    """YAML ``run.llm_response_hash_dir``, with older cache-dir keys as fallback."""
    cfg = run_cfg or {}
    return (
        cfg.get("llm_response_hash_dir")
        or cfg.get("llm_response_cache_dir")
        or cfg.get("cache_dir")
        or DEFAULT_LLM_RESPONSE_HASH_DIR
    )


class LlmResponseHash:
    """JSON-file memo of one LLM completion, keyed by ``llm_request_hash``.

    Who writes/reads when wired: OpenAIReader; teacher / autorater later, same
    class, different ``pipeline_stage``. MockReader does not use this (no API).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
