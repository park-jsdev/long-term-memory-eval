"""Download LoCoMo + upload audit packs when ``storage.backend`` is ``gcs``.

locomo_eval still reads/writes a local directory. This module is the only
place that copies that directory to/from the bucket.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memorybench.gcs_object_store import gcs_run_prefix
from src.memorybench.open_configured_store import open_configured_store


def gcs_enabled(cfg: dict[str, Any]) -> bool:
    storage = cfg.get("storage") or {}
    return str(storage.get("backend") or "local").lower() == "gcs"


def experiment_name(cfg: dict[str, Any]) -> str:
    return str((cfg.get("experiment") or {}).get("name") or "experiment")


def remote_run_prefix(cfg: dict[str, Any], run_id: str) -> str:
    return gcs_run_prefix(experiment_name(cfg), run_id)


def dataset_object_name(cfg: dict[str, Any]) -> str:
    storage = cfg.get("storage") or {}
    return str(storage.get("dataset_object") or "data/locomo10.json")


def ensure_dataset_local(cfg: dict[str, Any], *, store: Any | None = None) -> Path:
    """Return a local LoCoMo JSON path. Downloads from GCS when backend is gcs."""
    if not gcs_enabled(cfg):
        raw = (cfg.get("benchmark") or {}).get("dataset_path") or "data/raw/locomo10.json"
        return Path(raw)
    from src.memorybench.open_configured_store import local_experiments_root

    dest = local_experiments_root(cfg) / "data" / "locomo10.json"
    if dest.is_file():
        return dest
    handle = store or open_configured_store(cfg)
    obj = dataset_object_name(cfg)
    try:
        handle.download(obj, dest)
    except Exception as exc:
        raise SystemExit(
            f"Could not download gs dataset object {obj!r} to {dest}. "
            "Upload locomo10.json to that blob first."
        ) from exc
    return dest


def ensure_shared_index_local(
    cfg: dict[str, Any],
    index_run_id: str,
    dump_name: str,
    out_root: Path,
    *,
    store: Any | None = None,
) -> Path:
    """Ensure ``out_root/<index_run_id>/<dump_name>/`` exists.

    On GCS, download ``shared/<index_run_id>/`` (upload the local
    ``experiments/<index_run_id>/`` tree once after run_index).
    """
    dest = out_root / index_run_id / dump_name
    if dest.is_dir() and any(dest.iterdir()):
        return dest
    if not gcs_enabled(cfg):
        return dest
    handle = store or open_configured_store(cfg)
    n = download_tree(handle, f"shared/{index_run_id}", out_root / index_run_id)
    if n == 0 or not dest.is_dir() or not any(dest.iterdir()):
        raise SystemExit(
            f"Could not download gs://…/shared/{index_run_id}/ ({dump_name}). "
            f"Build locally then: gcloud storage cp -r experiments/{index_run_id} "
            "gs://$BUCKET/shared/"
        )
    return dest


def gcs_blob_exists(cfg: dict[str, Any], remote_path: str, *, store: Any | None = None) -> bool:
    handle = store or open_configured_store(cfg)
    return bool(handle.exists(remote_path))


def qa_success_remote(cfg: dict[str, Any], run_id: str) -> str:
    return f"{remote_run_prefix(cfg, run_id)}/_SUCCESS"


def autorater_success_remote(cfg: dict[str, Any], run_id: str) -> str:
    return f"{remote_run_prefix(cfg, run_id)}/autorater/_SUCCESS"


def aggregate_prefix(cfg: dict[str, Any]) -> str:
    """Thin catalog: ``experiments/<name>/aggregate/`` (default third wave)."""
    return f"experiments/{experiment_name(cfg)}/aggregate"


def collected_prefix(cfg: dict[str, Any]) -> str:
    """Full packs: ``experiments/<name>/collected/`` (on-demand third wave)."""
    return f"experiments/{experiment_name(cfg)}/collected"


def download_experiment_runs(
    cfg: dict[str, Any],
    run_ids: list[str],
    out_root: Path,
    *,
    store: Any | None = None,
) -> dict[str, int]:
    """Pull each run prefix into ``out_root/<run_id>/``. Local backend is a no-op."""
    if not gcs_enabled(cfg):
        return {}
    handle = store or open_configured_store(cfg)
    counts: dict[str, int] = {}
    for run_id in run_ids:
        counts[run_id] = download_tree(
            handle, remote_run_prefix(cfg, run_id), out_root / run_id
        )
    return counts


def upload_aggregate_dir(
    cfg: dict[str, Any], agg_dir: Path, *, store: Any | None = None
) -> int:
    if not gcs_enabled(cfg):
        return 0
    handle = store or open_configured_store(cfg)
    return upload_tree(handle, agg_dir, aggregate_prefix(cfg))


def upload_collected_dir(
    cfg: dict[str, Any], collected_dir: Path, *, store: Any | None = None
) -> int:
    if not gcs_enabled(cfg):
        return 0
    handle = store or open_configured_store(cfg)
    return upload_tree(handle, collected_dir, collected_prefix(cfg))


def upload_tree(store: Any, local_dir: Path, remote_prefix: str) -> int:
    """Copy every file under local_dir to remote_prefix/. Returns file count."""
    if not local_dir.is_dir():
        return 0
    n = 0
    prefix = remote_prefix.strip("/")
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        store.upload(path, f"{prefix}/{rel}")
        n += 1
    return n


def download_tree(store: Any, remote_prefix: str, local_dir: Path) -> int:
    marker = remote_prefix.strip("/").replace("\\", "/")
    names = store.list(marker)
    n = 0
    for name in names:
        key = str(name).replace("\\", "/")
        if key.endswith("/"):
            continue
        if key == marker:
            continue
        if key.startswith(marker + "/"):
            rel = key[len(marker) + 1 :]
        else:
            continue
        dest = local_dir / rel
        store.download(key, dest)
        n += 1
    return n


def upload_run_dir(cfg: dict[str, Any], run_id: str, run_dir: Path, *, store: Any | None = None) -> int:
    if not gcs_enabled(cfg):
        return 0
    handle = store or open_configured_store(cfg)
    return upload_tree(handle, run_dir, remote_run_prefix(cfg, run_id))


def download_run_dir(cfg: dict[str, Any], run_id: str, run_dir: Path, *, store: Any | None = None) -> int:
    if not gcs_enabled(cfg):
        return 0
    handle = store or open_configured_store(cfg)
    return download_tree(handle, remote_run_prefix(cfg, run_id), run_dir)
