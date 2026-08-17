"""Load secrets from a repo-root .env file (never commit .env)."""

from __future__ import annotations

from pathlib import Path

# src/locomo_eval/env.py → repo root is parents[2]
ROOT = Path(__file__).resolve().parents[2]

_LOADED = False


def load_env(dotenv_path: str | Path | None = None, override: bool = False) -> Path | None:
    """Load .env once. Existing real env vars win unless override=True.

    Returns the path that was loaded, or None if no file / dotenv missing.
    """
    global _LOADED
    if _LOADED and dotenv_path is None:
        return ROOT / ".env" if (ROOT / ".env").is_file() else None

    path = Path(dotenv_path) if dotenv_path else ROOT / ".env"
    if not path.is_file():
        _LOADED = True
        return None

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            "python-dotenv is required to load .env. "
            "Install with: pip install python-dotenv"
        ) from exc

    load_dotenv(path, override=override)
    _LOADED = True
    return path
