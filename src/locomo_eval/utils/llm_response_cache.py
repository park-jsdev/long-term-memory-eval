"""Content-addressed disk memo of **LLM** responses (not run checkpoints).

Future optimization: reusable at any LLM call site so identical requests are
not re-billed. **Not wired** into ``run_locomo_pipeline_with_memory_config``
while the end-to-end pipeline is being validated.

To re-enable on the answer reader:
  1. Construct ``LlmResponseCache(llm_response_cache_dir_from_run_cfg(cfg["run"]))``
  2. Pass it as ``llm_response_cache=`` to ``get_reader`` / ``OpenAIReader``
  3. Set ``run.llm_response_cache_dir`` in the YAML (default ``experiments/cache/``)

Call sites must put ``pipeline_stage`` in the key payload so stages cannot collide:

  - ``answer_reader`` — ``OpenAIReader.answer`` (hook exists; run.py does not pass a cache)
  - ``teacher`` — reserved for a write-path teacher LLM
  - ``autorater`` — reserved for a future LLM-as-judge (not ``offline_evaluate.py``)

Keyed by a hash of the request (stage, provider, model, sampling, prompt), not
by run or question id.

Not this store:
  - Per-run resume: ``experiments/<run_id>/predictions.jsonl``
  - Offline string metrics: ``offline_evaluate.py`` (no LLM)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Values for payload["pipeline_stage"]. New LLM call sites must pick a new id.
PIPELINE_STAGE_ANSWER_READER = "answer_reader"
PIPELINE_STAGE_TEACHER = "teacher"
PIPELINE_STAGE_AUTORATER = "autorater"

DEFAULT_LLM_RESPONSE_CACHE_DIR = "experiments/cache"


def llm_response_cache_dir_from_run_cfg(run_cfg: dict | None) -> str:
    """YAML ``run.llm_response_cache_dir``, with old ``run.cache_dir`` as fallback."""
    cfg = run_cfg or {}
    return (
        cfg.get("llm_response_cache_dir")
        or cfg.get("cache_dir")
        or DEFAULT_LLM_RESPONSE_CACHE_DIR
    )


class LlmResponseCache:
    """JSON-file memo of one LLM completion, keyed by the full request hash.

    Who writes/reads when wired: OpenAIReader; teacher / autorater later, same
    class, different ``pipeline_stage``. MockReader does not use this (no API).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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
