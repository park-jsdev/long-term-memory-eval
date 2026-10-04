"""Filesystem object store. Remote paths are relative to a local root."""

from __future__ import annotations

import shutil
from pathlib import Path


class LocalObjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _full(self, path: str) -> Path:
        return self.root / path.replace("\\", "/").lstrip("/")

    def exists(self, path: str) -> bool:
        return self._full(path).exists()

    def upload(self, local_path: Path, remote_path: str) -> None:
        dest = self._full(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def download(self, remote_path: str, local_path: Path) -> None:
        src = self._full(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)

    def list(self, prefix: str) -> list[str]:
        base = self._full(prefix)
        if not base.exists():
            return []
        out: list[str] = []
        for path in base.rglob("*"):
            if path.is_file():
                out.append(path.relative_to(self.root).as_posix())
        return sorted(out)
