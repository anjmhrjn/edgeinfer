"""Download and verify checkpoints. Weights are cached outside the repo."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

CACHE_DIR = Path(os.environ.get("EDGEINFER_CACHE", "~/.cache/edgeinfer")).expanduser() / "weights"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch_url_weights(url: str, filename: str, sha256: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Return the cached checkpoint, downloading it if needed. Always verifies SHA-256."""
    path = cache_dir / filename
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    actual = sha256_file(path)
    if actual != sha256:
        raise ValueError(f"{path}: SHA-256 {actual} does not match config {sha256}")
    return path
