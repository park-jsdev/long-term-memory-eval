"""Shared helpers that are not a pipeline step.

``llm_response_hash`` is a disk memo of LLM replies, keyed by
``llm_request_hash``. Implemented here but not wired into ``run.py``
until E2E validation is done (future optimization).

``llm_request_hash`` is the SHA-256 of an LLM request payload.
Compare scripts use it for distinctness sanity, not as a live store lookup.
"""
