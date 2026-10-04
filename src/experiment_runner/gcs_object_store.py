"""GCS object store. Imported only when storage.backend is gcs.

Application code never mentions Secret Manager; ADC supplies credentials.
"""

from __future__ import annotations

from pathlib import Path


class GCSObjectStore:
    def __init__(self, bucket: str, prefix: str = "") -> None:
        try:
            from google.cloud import storage  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "google-cloud-storage is required for storage.backend: gcs. "
                "Install extras: pip install google-cloud-storage"
            ) from exc
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)
        self.prefix = prefix.strip("/")

    def _blob_name(self, path: str) -> str:
        rel = path.replace("\\", "/").lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{rel}"
        return rel

    def exists(self, path: str) -> bool:
        return self._bucket.blob(self._blob_name(path)).exists()

    def upload(self, local_path: Path, remote_path: str) -> None:
        self._bucket.blob(self._blob_name(remote_path)).upload_from_filename(
            str(local_path)
        )

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._bucket.blob(self._blob_name(remote_path)).download_to_filename(
            str(local_path)
        )

    def list(self, prefix: str) -> list[str]:
        full = self._blob_name(prefix)
        names: list[str] = []
        for blob in self._client.list_blobs(self._bucket, prefix=full):
            name = blob.name
            if self.prefix:
                name = name[len(self.prefix) :].lstrip("/")
            names.append(name)
        return names


def gcs_run_prefix(experiment_name: str, run_id: str) -> str:
    """Path under the experiment prefix: runs/<run_id>/..."""
    return f"experiments/{experiment_name}/runs/{run_id}"
