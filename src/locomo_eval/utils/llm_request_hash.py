"""SHA-256 of one LLM request payload.

This is not a disk lookup. Evaluation factories prohibit
``LlmResponseHash``. We still hash the same dict ``OpenAIChatCaller`` would
use, so tests and compare scripts can check that two runs asked the model
different things.

Lifecycle
---------
1. **Live call:** ``OpenAIChatCaller.complete`` records the request hash for
   audit and distinctness checks only. No response-store lookup is permitted.
2. **Offline rebuild:** ``llm_request_hash_from_prediction`` re-renders the
   QA prompt from stored ``memory_text`` + ``question`` and hashes it with
   the logged reader model / temperature / max_tokens.
3. **Callers that consume the hash (sanity only):**
   - ``scripts/compare_full_runs.py`` — want
     ``fraction_same_llm_request_hash ≈ 0`` when memory changed the prompt.
   - ``scripts/compare_cross_model.py`` — reader axis: want ≈ 0 (model in
     the payload changed); teacher axis: hashes differ only if memory text
     differs (reader is frozen).
   - ``tests/test_evaluation_pipeline.py`` — raw vs session-summary confound lock.
   - ``tests/test_integration_sanity.py`` — reader-model swap lock.

Equal hashes mean the same intended Chat Completions request. They do not
mean the stored answers matched, and they do not mean a disk hit occurred.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..models import ModelSpec, resolve_model
from ..prompts import render_qa_prompt

# Values for payload["pipeline_stage"]. New LLM call sites must pick a new id.
PIPELINE_STAGE_ANSWER_READER = "answer_reader"
PIPELINE_STAGE_TEACHER = "teacher"
PIPELINE_STAGE_AUTORATER = "autorater"


def llm_request_hash(payload: dict[str, Any]) -> str:
    """SHA-256 of canonical JSON. Filename if ``LlmResponseHash`` is used."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def llm_request_payload(
    *,
    spec: ModelSpec,
    temperature: float,
    max_tokens: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Canonical request dict hashed by ``llm_request_hash``.

    Shared by the live OpenAI caller (reader and teacher) so the offline
    hash cannot drift from the payload a future store would use.
    ``extra`` is caller-specific (role, prompt, pipeline_stage, …).
    """
    payload = {
        "provider": "openai",
        "model": spec.model_id,
        "family": spec.family,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "max_tokens_field": spec.max_tokens_field,
        "reasoning_effort": spec.reasoning_effort,
        **extra,
    }
    if "pipeline_stage" not in payload:
        payload["pipeline_stage"] = (
            PIPELINE_STAGE_TEACHER
            if payload.get("role") == "teacher"
            else PIPELINE_STAGE_ANSWER_READER
        )
    return payload


def llm_request_hash_for_reader(
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt: str,
) -> str:
    """SHA-256 of an answer-reader request (filled QA prompt + model knobs)."""
    spec = resolve_model(model)
    payload = llm_request_payload(
        spec=spec,
        temperature=temperature,
        max_tokens=max_tokens,
        extra={
            "role": "reader",
            "prompt": prompt,
            "pipeline_stage": PIPELINE_STAGE_ANSWER_READER,
        },
    )
    return llm_request_hash(payload)


def llm_request_hash_from_prediction(
    row: dict[str, Any],
    meta: dict[str, Any],
    template: str,
) -> str:
    """Rebuild the reader ``llm_request_hash`` from a stored prediction (no API).

    ``row`` is one ``predictions.jsonl`` object; ``meta`` is ``run_meta.json``.
    Used by compare scripts after both runs have finished.
    """
    model = row.get("reader_model") or meta.get("reader_model")
    if not model:
        raise ValueError("reader_model missing from prediction row and run_meta")
    prompt = render_qa_prompt(
        template, row.get("memory_text") or "", row.get("question") or ""
    )
    return llm_request_hash_for_reader(
        model=str(model),
        temperature=float(meta.get("temperature", 0.0) or 0.0),
        max_tokens=int(meta.get("max_tokens", 64) or 64),
        prompt=prompt,
    )
