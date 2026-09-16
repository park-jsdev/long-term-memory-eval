"""GCS workspace helpers (fake store; no GCP)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.gcs_run_workspace import (
    download_tree,
    ensure_shared_index_local,
    gcs_enabled,
    remote_run_prefix,
    upload_tree,
)
from src.memorybench.load_experiment_yaml import load_experiment_yaml


class FakeStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def exists(self, path: str) -> bool:
        return path.replace("\\", "/") in self.blobs

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.blobs[remote_path.replace("\\", "/")] = Path(local_path).read_bytes()

    def download(self, remote_path: str, local_path: Path) -> None:
        key = remote_path.replace("\\", "/")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.blobs[key])

    def list(self, prefix: str) -> list[str]:
        marker = prefix.strip("/").replace("\\", "/")
        out = []
        for key in self.blobs:
            if key == marker or key.startswith(marker + "/"):
                out.append(key)
        return sorted(out)


class TestGcsWorkspaceHelpers(unittest.TestCase):
    def test_upload_then_download_roundtrips_files(self):
        store = FakeStore()
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "run"
            (src / "reader").mkdir(parents=True)
            (src / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            (src / "reader" / "traces.jsonl").write_text("{}\n", encoding="utf-8")
            n = upload_tree(store, src, "experiments/locomo-poc/runs/cell-1")
            self.assertEqual(n, 2)
            dest = Path(raw) / "restored"
            m = download_tree(store, "experiments/locomo-poc/runs/cell-1", dest)
            self.assertEqual(m, 2)
            self.assertEqual((dest / "_SUCCESS").read_text(encoding="utf-8"), "ok\n")
            self.assertTrue((dest / "reader" / "traces.jsonl").is_file())

    def test_poc_gcs_yaml_keeps_one_mock_full_context_cell(self):
        cfg = load_experiment_yaml(ROOT / "configs" / "experiments" / "poc_gcs.yaml")
        self.assertTrue(gcs_enabled(cfg))
        self.assertEqual(cfg["storage"]["dataset_object"], "data/locomo10.json")
        specs = expand_run_matrix(cfg)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].memory_method, "full_context")
        prefix = remote_run_prefix(cfg, specs[0].run_id)
        self.assertTrue(prefix.startswith("experiments/locomo-poc/runs/"))

    def test_ensure_shared_index_downloads_rag_dump_from_shared_prefix(self):
        store = FakeStore()
        cfg = {"storage": {"backend": "gcs"}}
        with tempfile.TemporaryDirectory() as raw:
            src = Path(raw) / "idx"
            (src / "rag_index").mkdir(parents=True)
            (src / "rag_index" / "schema.json").write_text("{}\n", encoding="utf-8")
            upload_tree(store, src, "shared/rag_locomo10")
            out = Path(raw) / "experiments"
            dest = ensure_shared_index_local(
                cfg, "rag_locomo10", "rag_index", out, store=store
            )
            self.assertTrue((dest / "schema.json").is_file())

    def test_ensure_shared_index_raises_when_shared_prefix_is_empty(self):
        store = FakeStore()
        cfg = {"storage": {"backend": "gcs"}}
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(SystemExit):
                ensure_shared_index_local(
                    cfg,
                    "rag_locomo10",
                    "rag_index",
                    Path(raw) / "experiments",
                    store=store,
                )

    def test_poc_gcs_include_does_not_drop_mock_execution(self):
        cfg = load_config(ROOT / "configs" / "experiments" / "poc_gcs.yaml")
        self.assertEqual(cfg["execution"]["reader_provider"], "mock")
        self.assertEqual(cfg["storage"]["backend"], "gcs")


if __name__ == "__main__":
    unittest.main()
