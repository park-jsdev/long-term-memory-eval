"""Tiny live/mock ping for teacher providers (OpenAI, Anthropic, DeepSeek).

Not a QA run. Use this to confirm keys and cheap models before indexing.

    python -m src.locomo_eval.ping_teachers
    python -m src.locomo_eval.ping_teachers --mock
    python -m src.locomo_eval.ping_teachers --providers openai,anthropic,deepseek
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.locomo_eval.env import load_env
from src.locomo_eval.models import (
    SANITY_TEACHER_ANTHROPIC,
    SANITY_TEACHER_DEEPSEEK,
    SANITY_TEACHER_OPENAI,
)
from src.locomo_eval.teachers import get_teacher

DEFAULT_ROSTER = (
    ("openai", SANITY_TEACHER_OPENAI),
    ("anthropic", SANITY_TEACHER_ANTHROPIC),
    ("deepseek", SANITY_TEACHER_DEEPSEEK),
)

PING_MAX_TOKENS = 64


def format_ping_error(exc: BaseException) -> str:
    """Human-readable failure line. Some SDKs raise with empty str(exc)."""
    msg = str(exc).strip() or repr(exc)
    name = type(exc).__name__
    if msg.startswith(name):
        return msg
    return f"{name}: {msg}"


def ping_teachers(
    providers: list[str],
    *,
    mock: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    wanted = {p.strip().lower() for p in providers if p.strip()}
    for provider, model in DEFAULT_ROSTER:
        if wanted and provider not in wanted:
            continue
        name = "mock" if mock else provider
        teacher = get_teacher(
            name, model=model, teacher_id=provider, max_tokens=PING_MAX_TOKENS, thinking=False
        )
        try:
            text, meta = teacher.ping()
            ok = "pong" in (text or "").casefold()
            error = None
            if not ok and not (text or "").strip():
                error = (
                    "empty completion (no exception). "
                    "Ping forces thinking off; if this persists, raise max_tokens."
                )
            rows.append(
                {
                    "provider": provider,
                    "model": teacher.model_name,
                    "ok": ok,
                    "text": (text or "")[:80],
                    "latency_s": meta.get("latency_s"),
                    "error": error,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "ok": False,
                    "text": "",
                    "latency_s": None,
                    "error": format_ping_error(exc),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    loaded = load_env()
    if loaded is not None:
        print(f"Loaded env from {loaded}")
    parser = argparse.ArgumentParser(description="Ping cheap teacher models (plumbing check)")
    parser.add_argument(
        "--providers",
        default="openai,anthropic,deepseek",
        help="Comma-separated: openai,anthropic,deepseek",
    )
    parser.add_argument("--mock", action="store_true", help="No API; MockTeacher replies pong")
    args = parser.parse_args(argv)
    providers = [p.strip() for p in str(args.providers).split(",") if p.strip()]
    rows = ping_teachers(providers, mock=bool(args.mock))
    n_ok = 0
    for row in rows:
        status = "ok" if row["ok"] else "FAIL"
        if row["ok"]:
            n_ok += 1
        extra = row["error"] or row["text"]
        print(f"  [{status}] {row['provider']}/{row['model']}: {extra}")
    print(f"{n_ok}/{len(rows)} teachers replied pong")
    return 0 if n_ok == len(rows) and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
