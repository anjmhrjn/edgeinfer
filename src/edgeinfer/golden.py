"""Golden outputs: FP32 reference input + logits per image, for parity checks.

One uncompressed .npz per image under artifacts/golden/<model>/ (gitignored:
derived from Cityscapes). Arrays:
    input   1x3xHxW float32, the preprocessed tensor fed to the model
    logits  1xCxhxw float32, raw model output at native stride (before upsampling)
Storing the input lets other runtimes (ORT, TensorRT, the C++ runtime) be checked
without depending on preprocessing. Their SHA-256 hashes are committed in
data/golden_hashes/<model>.sha256 (`shasum -a 256 -c` format).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from edgeinfer.backends.base import Backend
from edgeinfer.config import ModelConfig
from edgeinfer.data.cityscapes import Sample
from edgeinfer.preprocess import preprocess
from edgeinfer.reports import REPO_ROOT

GOLDEN_DIR = REPO_ROOT / "artifacts" / "golden"
HASH_DIR = REPO_ROOT / "data" / "golden_hashes"


def golden_bytes(x: np.ndarray, logits: np.ndarray) -> bytes:
    """Serialize one golden pair. Must be byte-identical for identical arrays."""
    buf = io.BytesIO()
    np.savez(buf, input=x, logits=logits)
    return buf.getvalue()


def write_golden(
    cfg: ModelConfig, backend: Backend, samples: Iterable[Sample], out_dir: Path
) -> dict[str, str]:
    """Write one .npz per sample; returns {path relative to repo root: sha256}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for sample in samples:
        x = preprocess(sample.image, cfg)
        logits = backend.infer(x)
        if logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError(f"{sample.entry}: logits must be finite float32")
        data = golden_bytes(x, logits)
        path = out_dir / f"{sample.entry.image_id}.npz"
        path.write_bytes(data)
        hashes[_rel(path)] = hashlib.sha256(data).hexdigest()
    return hashes


def format_hashes(hashes: dict[str, str]) -> str:
    return "".join(f"{h}  {p}\n" for p, h in sorted(hashes.items()))


def _rel(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def verify_golden(model: str, golden_dir: Path, image_ids: list[str]) -> bool:
    """Check golden files against the committed hashes in data/golden_hashes/<model>.sha256.

    Returns False if there is nothing to check against (no hash file, or golden files
    outside the repo, e.g. a test fixture). Raises if a file differs: parity against a
    changed reference would be meaningless."""
    hash_path = HASH_DIR / f"{model}.sha256"
    if not hash_path.is_file() or not golden_dir.resolve().is_relative_to(REPO_ROOT):
        return False
    committed = {}
    for line in hash_path.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        committed[rel] = digest
    for image_id in image_ids:
        path = golden_dir / f"{image_id}.npz"
        rel = _rel(path)
        if rel not in committed:
            raise ValueError(f"{rel} has no committed hash in {hash_path.name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != committed[rel]:
            raise ValueError(f"{rel} differs from its committed hash; golden outputs changed")
    return True
