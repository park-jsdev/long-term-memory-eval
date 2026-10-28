"""Render scoped hypotheses and a truncated conventional summary.

H2 compares the two texts at the same whitespace-token budget. The
summary arm reuses the idea of session summaries already in
MemoryBuilder. This module does not call that builder, so a fixture
can be scored without a LoCoMo Conversation object.
"""

from __future__ import annotations

from .schema import HypothesisRecord

_TRUNCATION_MARK = "[... truncated to token budget ...]"


def render_hypothesis(record: HypothesisRecord) -> str:
    """One record as text the response model, or a score, can read.

    Evidence ids are written out so the PoC coverage check can see them.
    A later extractor keeps the ids in the record even if the prose drops
    them.
    """
    exceptions = ", ".join(record.exceptions) if record.exceptions else "none"
    evidence = ", ".join(record.evidence_ids)
    sessions = ", ".join(str(s) for s in record.session_ids)
    return (
        f"[{record.record_id}] Who: {record.who_basis} speaker {record.speaker}\n"
        f"What ({record.what_type}): {record.what}\n"
        f"Scope: {record.scope}\n"
        f"Exceptions: {exceptions}\n"
        f"Evidence: {evidence}\n"
        f"Sessions: {sessions}"
    )


def render_hypotheses(records: list[HypothesisRecord]) -> str:
    if not records:
        return "(No scoped behavioral hypotheses.)"
    return "\n\n".join(render_hypothesis(record) for record in records)


def render_memory_for_reply(record: HypothesisRecord) -> str:
    """Who/What/Why for a reply. Record ids stay in the audit files.

    The reply model does not need the id to follow the constraint, and
    the id would show up in answers as a bookkeeping confound. "This user"
    is the person speaking. "Speaker A" was read as someone else.
    """
    exceptions = ", ".join(record.exceptions) if record.exceptions else "none"
    why = record.uncertainty.strip() or "none stated"
    return (
        f"Who: {record.who_basis}, this user\n"
        f"What ({record.what_type}): {record.what}\n"
        f"Scope: {record.scope}\n"
        f"Exceptions: {exceptions}\n"
        f"Why: {why}"
    )


def render_record_for_decision(record: HypothesisRecord) -> str:
    """Record text for Stage 1 and Stage 2. The id stays for the selector.

    The reply rendering drops the id. This one keeps it so Stage 1 can
    copy it into selected_ids. "This user" blocks the other-person abstain.
    """
    exceptions = ", ".join(record.exceptions) if record.exceptions else "none"
    evidence = ", ".join(record.evidence_ids)
    return (
        f"[{record.record_id}] Who: {record.who_basis}, this user\n"
        f"What ({record.what_type}): {record.what}\n"
        f"Scope: {record.scope}\n"
        f"Exceptions: {exceptions}\n"
        f"Evidence: {evidence}"
    )


def whitespace_tokens(text: str) -> list[str]:
    """Split on whitespace. This budget is not the model's tokenizer."""
    return str(text).split()


def truncate_to_token_budget(text: str, budget: int) -> str:
    """Keep the tail. Recent sessions are the ones a summary should retain.

    A negative budget is a caller bug. Zero means the arm is empty on
    purpose, which is how a matched-budget test can force a short summary.
    """
    if budget < 0:
        raise ValueError("token budget must be >= 0")
    tokens = whitespace_tokens(text)
    if len(tokens) <= budget:
        return text.strip()
    mark_tokens = _TRUNCATION_MARK.split()
    room = budget - len(mark_tokens)
    if room <= 0:
        return _TRUNCATION_MARK
    kept = " ".join(tokens[-room:])
    return f"{_TRUNCATION_MARK}\n{kept}"


def conventional_summary(
    session_summaries: list[tuple[str, str]],
    budget: int,
) -> str:
    """Join session summaries, then cut them to the hypothesis budget.

    session_summaries are (session id, text) in chronological order.
    The full version should build this with the same model that writes
    hypotheses. The PoC only needs the truncation seam.
    """
    chunks = []
    for session_id, text in session_summaries:
        body = str(text).strip()
        if not body:
            continue
        chunks.append(f"[Session {session_id}]\n{body}")
    joined = "\n\n".join(chunks) if chunks else "(No session summaries.)"
    return truncate_to_token_budget(joined, budget)


def hypothesis_token_budget(records: list[HypothesisRecord]) -> int:
    """Budget the summary must match: the rendered hypothesis length."""
    return len(whitespace_tokens(render_hypotheses(records)))


def _contains(text: str, needle: str) -> bool:
    return needle.casefold() in text.casefold()


def h2_scores(
    records: list[HypothesisRecord],
    hypothesis_text: str,
    summary_text: str,
    unsupported_phrases: list[str] | None = None,
) -> dict[str, object]:
    """Mechanical H2 numbers for a fixture. Meaning judgments come later.

    Evidence coverage is whether each evidence id appears in the text.
    Hypothesis rendering includes those ids, so this arm scores high by
    construction until a prose-only rendering is introduced.
    """
    phrases = list(unsupported_phrases or [])
    evidence_ids = [eid for record in records for eid in record.evidence_ids]
    scopes = [record.scope for record in records if record.scope.strip()]

    def coverage(text: str) -> float:
        if not evidence_ids:
            return 0.0
        found = sum(1 for eid in evidence_ids if eid in text)
        return found / len(evidence_ids)

    def scopes_present(text: str) -> bool:
        return bool(scopes) and all(_contains(text, scope) for scope in scopes)

    def unsupported(text: str) -> int:
        return sum(1 for phrase in phrases if _contains(text, phrase))

    return {
        "hypothesis_evidence_coverage": coverage(hypothesis_text),
        "summary_evidence_coverage": coverage(summary_text),
        "hypothesis_scope_present": scopes_present(hypothesis_text),
        "summary_scope_present": scopes_present(summary_text),
        "hypothesis_unsupported": unsupported(hypothesis_text),
        "summary_unsupported": unsupported(summary_text),
        "hypothesis_tokens": len(whitespace_tokens(hypothesis_text)),
        "summary_tokens": len(whitespace_tokens(summary_text)),
        "token_budget": hypothesis_token_budget(records),
    }
