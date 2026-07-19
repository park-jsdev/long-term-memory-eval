"""Download LoCoMo locomo10.json at a pinned commit."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

DEFAULT_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
DEFAULT_URL = (
    f"https://raw.githubusercontent.com/snap-research/locomo/{DEFAULT_COMMIT}/data/locomo10.json"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/raw/locomo10.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        digest = sha256_file(out)
        print(f"Already present: {out}")
        print(f"SHA256: {digest}")
        return

    print(f"Downloading {args.url}")
    resp = requests.get(args.url, timeout=120)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    digest = sha256_file(out)
    print(f"Wrote {out} ({len(resp.content)} bytes)")
    print(f"SHA256: {digest}")
    print("License: CC BY-NC 4.0 — research / non-commercial use only.")
    print(f"Pinned commit: {DEFAULT_COMMIT}")


if __name__ == "__main__":
    main()
