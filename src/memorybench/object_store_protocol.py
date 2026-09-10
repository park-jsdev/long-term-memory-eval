"""Portable object-store protocol. Experiment logic depends only on this."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    def exists(self, path: str) -> bool: ...

    def upload(self, local_path: Path, remote_path: str) -> None: ...

    def download(self, remote_path: str, local_path: Path) -> None: ...

    def list(self, prefix: str) -> list[str]: ...
