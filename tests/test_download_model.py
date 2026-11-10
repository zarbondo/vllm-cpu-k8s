"""Unit tests of scripts/download_model.py with the Hugging Face API mocked out."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_model.py"
spec = importlib.util.spec_from_file_location("download_model", SCRIPT)
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)

MODEL = "Org/Tiny-Instruct"


class FakeHub:
    def __init__(self, files: dict[str, bytes], sha: str = "abc123"):
        self.files, self.sha, self.downloads = files, sha, 0

    def model_info(self, repo_id, revision=None, token=None):
        assert repo_id == MODEL
        return SimpleNamespace(sha=self.sha)

    def snapshot_download(self, repo_id, revision, local_dir, allow_patterns, token, max_workers):
        assert revision == self.sha  # the exact commit, not the moving branch
        self.downloads += 1
        for name, data in self.files.items():
            (Path(local_dir) / name).write_bytes(data)
        return str(local_dir)


GOOD = {
    "config.json": b"{}",
    "tokenizer.json": b"{}",
    "model.safetensors": b"\0" * 64,
}


@pytest.fixture
def hub(monkeypatch, tmp_path):
    def install(files=GOOD, sha="abc123"):
        fake = FakeHub(files, sha)
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=fake.model_info))
        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
        return fake

    monkeypatch.setenv("MODEL_ID", MODEL)
    monkeypatch.setenv("MODELS_ROOT", str(tmp_path))
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("MODEL_REVISION", raising=False)
    return install


def test_first_start_downloads_and_writes_marker(hub, tmp_path):
    fake = hub()
    assert dm.main() == 0
    marker = json.loads((tmp_path / "Org--Tiny-Instruct" / dm.MARKER).read_text())
    assert marker["model_id"] == MODEL and marker["revision"] == "main"
    assert marker["commit_sha"] == "abc123" and marker["files"] == 3
    assert fake.downloads == 1


def test_restart_uses_volume_without_network(hub, monkeypatch):
    hub()
    assert dm.main() == 0

    def no_network(*a, **kw):
        raise AssertionError("network must not be used when the model is cached")

    monkeypatch.setattr(huggingface_hub, "HfApi", no_network)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_network)
    assert dm.main() == 0


def test_pinned_commit_matches_marker(hub, monkeypatch):
    hub(sha="deadbeef")
    assert dm.main() == 0
    monkeypatch.setenv("MODEL_REVISION", "deadbeef")
    monkeypatch.setattr(huggingface_hub, "HfApi", None)  # would crash if called
    assert dm.main() == 0


def test_revision_change_replaces_old_files(hub, tmp_path, monkeypatch):
    hub(files={**GOOD, "old-only.safetensors": b"x"}, sha="v1")
    assert dm.main() == 0
    monkeypatch.setenv("MODEL_REVISION", "v2")
    fake = hub(sha="v2")
    assert dm.main() == 0
    model_dir = tmp_path / "Org--Tiny-Instruct"
    assert not (model_dir / "old-only.safetensors").exists()
    assert json.loads((model_dir / dm.MARKER).read_text())["commit_sha"] == "v2"
    assert fake.downloads == 1


def test_incomplete_checkpoint_fails_without_marker(hub, tmp_path):
    files = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
        ).encode(),
        "model-00001-of-00002.safetensors": b"x",
    }
    hub(files=files)
    assert dm.main() == 1
    assert not (tmp_path / "Org--Tiny-Instruct" / dm.MARKER).exists()


def test_empty_token_is_not_sent(hub, monkeypatch):
    seen = {}
    fake = hub()

    def model_info(repo_id, revision=None, token=None):
        seen["token"] = token
        return SimpleNamespace(sha=fake.sha)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=model_info))
    monkeypatch.setenv("HF_TOKEN", "")
    assert dm.main() == 0
    assert seen["token"] is None
