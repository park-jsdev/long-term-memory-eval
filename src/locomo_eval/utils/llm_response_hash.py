"""Content-addressed disk memo of **LLM** responses (not run checkpoints).

Keyed by ``llm_request_hash`` of the request (stage, provider, model,
sampling, prompt), not by run or question id.

Future optimization: reusable at any LLM call site so identical requests are
not re-billed. **Not wired** into ``run_locomo_pipeline_with_memory_config``
while the end-to-end pipeline is being validated.

To re-enable on the answer reader:
  1. Construct ``LlmResponseHash(llm_response_hash_dir_from_run_cfg(cfg["run"]))``
  2. Pass it as ``llm_response_hash=`` to ``get_reader`` / ``OpenAIReader``
  3. Set ``run.llm_response_hash_dir`` in the YAML (default ``experiments/cache/``)

Call sites must put ``pipeline_stage`` in the key payload so stages cannot collide:

  - ``answer_reader`` — ``OpenAIReader.answer`` (hook exists; run.py does not pass a store)
  - ``teacher`` — reserved for a write-path teacher LLM
  - ``autorater`` — reserved for a future LLM-as-judge (not ``offline_evaluate.py``)

Not this store:
  - Per-run resume: ``experiments/<run_id>/predictions.jsonl``
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
