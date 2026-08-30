"""LLM-as-a-Judge autorater (Mem0 prompt). Online evaluation of stored answers.

This is **not** ``offline_evaluate.py`` (string EM / token F1 / LoCoMo F1) and
**not** the answer reader (which must not see gold). The autorater sees
question + gold + predicted answer and returns a binary CORRECT/WRONG label.

Protocol follows Mem0 (Chhikara et al., arXiv:2504.19413, Appendix A):
generous topic/date match, JSON ``{"label": "CORRECT"|"WRONG"}``, skip LoCoMo
category 5. Default live model is GPT-4o. Mem0's extract/update memory
algorithm is **not** implemented here.

Prompt source (pinned Mem0 code):
https://github.com/mem0ai/mem0/blob/ece7ff6b/evaluation/metrics/llm_judge.py

    Prediction JSONL  →  Autorater.rate()  →  llm_score 0/1
    (same rows the LoCoMo string scorer already uses)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .mem0_baselines import ADVERSARIAL_CATEGORY
from .models import DEFAULT_AUTORATER_MODEL
from .prompts import load_prompt_template, render_autorater_prompt
from .readers import OpenAIChatCaller

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTORATER_PROMPT = ROOT / "prompts" / "autorater_mem0_v1.txt"

# Released Mem0 judge does not set a completion-token limit.
DEFAULT_AUTORATER_MAX_TOKENS: int | None = None


@dataclass
class AutoraterVerdict:
    """One judge decision. Consumed by ``scripts.analysis.run_benchmark``.

    ``llm_score`` is 1 for CORRECT, 0 for WRONG. ``skipped`` is True when
    Mem0 would drop the item (category 5) so it is not mixed into J.
    """

    label: str
    llm_score: int
    raw_text: str
    latency_s: float
    usage: dict[str, Any]
    model: str
    provider: str
    skipped: bool = False
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Autorater(ABC):
    """Sandwich-adjacent evaluator: grades a predicted answer against gold.

    Called from ``run_benchmark`` after the answer LLM has already run.
    """

    model_name: str
    provider: str

    @abstractmethod
    def rate(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> AutoraterVerdict:
        """Return a CORRECT/WRONG verdict. Must see gold (unlike the reader)."""
        ...


def parse_judge_label(text: str) -> tuple[str, str | None]:
    """Extract CORRECT/WRONG from a judge completion.

    Prefers a JSON object with key ``label`` (Mem0 ``response_format``).
    Falls back to a unique CORRECT or WRONG token in free text.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty autorater response.")

    obj, reasoning = _try_json_label(raw)
    if obj is not None:
        return obj, reasoning

    blob = _first_json_object(raw)
    if blob:
        obj, reasoning = _try_json_label(blob)
        if obj is not None:
            return obj, reasoning

    tokens = set(re.findall(r"\b(?:CORRECT|WRONG|INCORRECT)\b", raw.upper()))
    labels = {
        "WRONG" if token in {"WRONG", "INCORRECT"} else "CORRECT"
        for token in tokens
    }
    if len(labels) == 1:
        return labels.pop(), None
    raise ValueError(f"Could not parse judge label from: {raw[:240]!r}")


def _normalize_label(value: Any) -> str:
    label = str(value).strip().upper()
    if label in {"CORRECT", "RIGHT", "YES", "TRUE", "1"}:
        return "CORRECT"
    if label in {"WRONG", "INCORRECT", "NO", "FALSE", "0"}:
        return "WRONG"
    raise ValueError(f"Unknown judge label {value!r}")


def _try_json_label(text: str) -> tuple[str | None, str | None]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(obj, dict) or "label" not in obj:
        return None, None
    reasoning = obj.get("reasoning") or obj.get("explanation")
    return _normalize_label(obj["label"]), (str(reasoning) if reasoning else None)


def _first_json_object(text: str) -> str | None:
    match = re.search(r"\{[^{}]*\}", text)
    return match.group(0) if match else None


def should_skip_category(category: int | str | None, skip_category: int = ADVERSARIAL_CATEGORY) -> bool:
    """Mem0 drops LoCoMo category 5 (adversarial / unanswerable)."""
    if category is None:
        return False
    try:
        return int(category) == int(skip_category)
    except (TypeError, ValueError):
        return False


class MockAutorater(Autorater):
    """Offline plumbing check. Token overlap is not a valid LLM judge score."""

    provider = "mock"

    def __init__(self, model_name: str = "mock"):
        # Never impersonate the configured live model in audit artifacts.
        self.model_name = "mock"

    def rate(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> AutoraterVerdict:
        gold_toks = {t for t in _mock_tokens(gold_answer) if len(t) >= 3}
        pred_toks = set(_mock_tokens(generated_answer))
        correct = bool(gold_toks & pred_toks) or (
            _mock_tokens(gold_answer) == _mock_tokens(generated_answer)
            and bool(_mock_tokens(gold_answer))
        )
        label = "CORRECT" if correct else "WRONG"
        return AutoraterVerdict(
            label=label,
            llm_score=1 if correct else 0,
            raw_text=json.dumps({"label": label}),
            latency_s=0.0,
            usage={},
            model=self.model_name,
            provider=self.provider,
            skipped=False,
            reasoning="mock token overlap" if correct else "mock no overlap",
        )


def _mock_tokens(text: str) -> list[str]:
    from .mem0_metrics import simple_tokenize

    return simple_tokenize(text)


class OpenAIAutorater(Autorater):
    """Live GPT-4o (or override) Mem0 judge via Chat Completions."""

    provider = "openai"

    def __init__(
        self,
        model: str = DEFAULT_AUTORATER_MODEL,
        temperature: float = 0.0,
        max_tokens: int | None = DEFAULT_AUTORATER_MAX_TOKENS,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
        max_retries: int = 8,
        min_request_interval_s: float = 0.0,
        max_wait_s: float = 3600.0,
        prompt_path: str | Path | None = None,
    ):
        self._chat = OpenAIChatCaller(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
        )
        self.model_name = self._chat.model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        path = Path(prompt_path) if prompt_path else DEFAULT_AUTORATER_PROMPT
        self.prompt_version, self.prompt_template = load_prompt_template(path)

    def rate(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> AutoraterVerdict:
        prompt = render_autorater_prompt(
            self.prompt_template,
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )
        messages = [{"role": "user", "content": prompt}]
        text, meta = self._chat.complete(
            messages,
            create_extra={"response_format": {"type": "json_object"}},
        )
        label, reasoning = parse_judge_label(text)
        return AutoraterVerdict(
            label=label,
            llm_score=1 if label == "CORRECT" else 0,
            raw_text=text,
            latency_s=float(meta.get("latency_s") or 0.0),
            usage=dict(meta.get("usage") or {}),
            model=str(meta.get("model") or self.model_name),
            provider=self.provider,
            skipped=False,
            reasoning=reasoning,
        )


def skipped_verdict(*, model: str, provider: str, category: int) -> AutoraterVerdict:
    """Placeholder row so the verdict audit stays 1:1 with predictions."""
    return AutoraterVerdict(
        label="SKIPPED",
        llm_score=0,
        raw_text="",
        latency_s=0.0,
        usage={},
        model=model,
        provider=provider,
        skipped=True,
        reasoning=f"Mem0 skips LoCoMo category {category}",
    )


def get_autorater(
    name: str,
    model: str = DEFAULT_AUTORATER_MODEL,
    temperature: float = 0.0,
    max_tokens: int | None = DEFAULT_AUTORATER_MAX_TOKENS,
    max_retries: int = 8,
    min_request_interval_s: float = 0.0,
    max_wait_s: float = 3600.0,
    prompt_path: str | Path | None = None,
) -> Autorater:
    """Build a mock or OpenAI autorater."""
    name = (name or "").lower()
    if name == "mock":
        return MockAutorater()
    if name == "openai":
        return OpenAIAutorater(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            min_request_interval_s=min_request_interval_s,
            max_wait_s=max_wait_s,
            prompt_path=prompt_path,
        )
    raise ValueError(f"Unknown autorater '{name}'. Use openai or mock.")
