#!/usr/bin/env python3
"""Download model weights from Hugging Face into the persistent volume, once.

Runs as the init container of the vLLM pod (and from the vLLM entrypoint when
MODEL_AUTO_DOWNLOAD=true, e.g. for a plain `docker run`).

* <MODEL_DIR>/.download-complete exists and was written for the same MODEL_ID
  and MODEL_REVISION -> exit at once, without touching the network.
* otherwise: resolve the revision to a commit sha, download that exact commit
  (resumable), check that config, tokenizer and every weight shard are in place,
  then write the marker atomically. A crash before the marker is written means
  the next start simply resumes the download.

A file lock in the volume keeps two pods from downloading into the same
directory at the same time.

Environment:
  MODEL_ID              Hugging Face repo id, e.g. Qwen/Qwen2.5-1.5B-Instruct (required)
  MODEL_REVISION        branch, tag or commit sha (default: main)
  MODELS_ROOT           mount point of the volume (default: /models)
  MODEL_DIR             target directory (default: <MODELS_ROOT>/<org>--<name>)
  MODEL_ALLOW_PATTERNS  comma-separated globs to fetch (default: configs, tokenizer, safetensors)
  HF_TOKEN              token for gated/private repos (optional, from a k8s Secret)
  DOWNLOAD_MAX_WORKERS  parallel downloads (default: 8)
  DOWNLOAD_LOCK_TIMEOUT seconds to wait for another pod's download (default: 3600)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MARKER = ".download-complete"
DEFAULT_PATTERNS = "*.json,*.safetensors,*.txt,*.model,*.tiktoken,*.jinja"


def log(msg: str) -> None:
    print(f"[model-downloader] {msg}", flush=True)


def model_dir_for(model_id: str, models_root: str) -> Path:
    return Path(models_root) / model_id.replace("/", "--")


def read_marker(model_dir: Path) -> dict | None:
    try:
        return json.loads((model_dir / MARKER).read_text())
    except (OSError, ValueError):
        return None


def marker_matches(marker: dict | None, model_id: str, revision: str) -> bool:
    return (
        marker is not None
        and marker.get("model_id") == model_id
        and revision in (marker.get("revision"), marker.get("commit_sha"))
    )


def verify(model_dir: Path) -> tuple[int, int]:
    """Fail loudly if the directory is not a complete safetensors checkpoint."""
    if not (model_dir / "config.json").is_file():
        raise RuntimeError("config.json is missing")
    if not any((model_dir / f).is_file() for f in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise RuntimeError("tokenizer files are missing")
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing = [s for s in shards if not (model_dir / s).is_file()]
        if missing:
            raise RuntimeError(f"weight shards are missing: {missing}")
    elif not list(model_dir.glob("*.safetensors")):
        raise RuntimeError("no *.safetensors weights found (only safetensors are downloaded)")
    files = [p for p in model_dir.rglob("*") if p.is_file() and ".cache" not in p.parts]
    return len(files), sum(p.stat().st_size for p in files)


def write_marker(model_dir: Path, data: dict) -> None:
    tmp = model_dir / f"{MARKER}.tmp"
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, model_dir / MARKER)


def download(model_id: str, revision: str, model_dir: Path, token: str | None) -> None:
    from huggingface_hub import HfApi, snapshot_download

    patterns = [p.strip() for p in os.environ.get("MODEL_ALLOW_PATTERNS", DEFAULT_PATTERNS).split(",") if p.strip()]
    # Pin the moving branch to one commit, so all files come from the same snapshot.
    sha = HfApi().model_info(model_id, revision=revision, token=token).sha
    log(f"downloading {model_id}@{revision} (commit {sha}) into {model_dir}")
    started = time.monotonic()
    snapshot_download(
        model_id,
        revision=sha,
        local_dir=model_dir,
        allow_patterns=patterns,
        token=token,
        max_workers=int(os.environ.get("DOWNLOAD_MAX_WORKERS", "8")),
    )
    n_files, n_bytes = verify(model_dir)
    write_marker(
        model_dir,
        {
            "model_id": model_id,
            "revision": revision,
            "commit_sha": sha,
            "files": n_files,
            "bytes": n_bytes,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    log(f"done: {n_files} files, {n_bytes / 2**30:.2f} GiB in {time.monotonic() - started:.0f}s")


def main() -> int:
    model_id = os.environ.get("MODEL_ID", "").strip()
    if not model_id:
        log("MODEL_ID is not set")
        return 1
    revision = os.environ.get("MODEL_REVISION", "").strip() or "main"
    models_root = os.environ.get("MODELS_ROOT", "/models")
    model_dir = Path(os.environ.get("MODEL_DIR") or model_dir_for(model_id, models_root))

    # Fast path: the volume already holds this model. No network, no lock.
    if marker_matches(read_marker(model_dir), model_id, revision):
        log(f"{model_id}@{revision} is already in {model_dir}, skipping download")
        return 0

    # An empty value from an optional Secret must not be sent as a token.
    token = os.environ.pop("HF_TOKEN", "").strip() or None

    from filelock import FileLock, Timeout

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(model_dir.parent / f".{model_dir.name}.lock"))
    try:
        with lock.acquire(timeout=float(os.environ.get("DOWNLOAD_LOCK_TIMEOUT", "3600"))):
            marker = read_marker(model_dir)
            if marker_matches(marker, model_id, revision):  # another pod finished it meanwhile
                log(f"{model_id}@{revision} was downloaded by another pod")
                return 0
            if marker is not None:
                # A different model/revision lives here: start clean, no stale shards.
                log(f"replacing {marker.get('model_id')}@{marker.get('revision')} in {model_dir}")
                shutil.rmtree(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            download(model_id, revision, model_dir, token)
    except Timeout:
        log("timed out waiting for another download into the same volume")
        return 1
    except Exception as exc:  # noqa: BLE001 - report and fail the init container
        log(f"download failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
