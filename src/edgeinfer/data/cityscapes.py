"""Read Cityscapes images and gtFine labels from $CITYSCAPES_ROOT.

Expected layout (the two official zips unpacked into one root):
    <root>/leftImg8bit/<split>/<city>/<image_id>_leftImg8bit.png
    <root>/gtFine/<split>/<city>/<image_id>_gtFine_labelIds.png
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from edgeinfer.data.labels import to_train_ids
from edgeinfer.data.manifests import ManifestEntry

IMAGE_SUFFIX = "_leftImg8bit.png"
LABEL_SUFFIX = "_gtFine_labelIds.png"


def cityscapes_root(root: str | Path | None = None) -> Path:
    """Resolve the dataset root from the argument or $CITYSCAPES_ROOT."""
    if root is None:
        env = os.environ.get("CITYSCAPES_ROOT")
        if not env:
            raise RuntimeError("CITYSCAPES_ROOT is not set; point it at the unpacked dataset")
        root = env
    path = Path(root).expanduser()
    for sub in ("leftImg8bit", "gtFine"):
        if not (path / sub).is_dir():
            raise FileNotFoundError(f"{path / sub} not found; is CITYSCAPES_ROOT correct?")
    return path


def list_image_ids(root: Path, split: str) -> list[str]:
    """All image IDs in a split, sorted."""
    split_dir = root / "leftImg8bit" / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_dir}")
    return sorted(p.name.removesuffix(IMAGE_SUFFIX) for p in split_dir.glob(f"*/*{IMAGE_SUFFIX}"))


def image_path(root: Path, entry: ManifestEntry) -> Path:
    return root / "leftImg8bit" / entry.split / entry.city / f"{entry.image_id}{IMAGE_SUFFIX}"


def label_path(root: Path, entry: ManifestEntry) -> Path:
    return root / "gtFine" / entry.split / entry.city / f"{entry.image_id}{LABEL_SUFFIX}"


@dataclass(frozen=True)
class Sample:
    entry: ManifestEntry
    image: np.ndarray  # HxWx3 uint8, RGB, original resolution
    label: np.ndarray  # HxW uint8 trainIds, 255 = ignore


def load_sample(root: Path, entry: ManifestEntry) -> Sample:
    with Image.open(image_path(root, entry)) as im:
        if im.mode != "RGB":
            raise ValueError(f"{entry}: expected an RGB image, got mode {im.mode}")
        image = np.asarray(im, dtype=np.uint8)
    with Image.open(label_path(root, entry)) as lb:
        if lb.mode != "L":
            raise ValueError(f"{entry}: expected an 8-bit labelIds image, got mode {lb.mode}")
        label = to_train_ids(np.asarray(lb, dtype=np.uint8))
    if image.shape[:2] != label.shape:
        raise ValueError(f"{entry}: image {image.shape[:2]} and label {label.shape} differ")
    return Sample(entry, image, label)


def iter_samples(root: Path, entries: list[ManifestEntry]) -> Iterator[Sample]:
    """Stream samples one at a time (keeps memory flat on the 16 GB dev machine)."""
    for entry in entries:
        yield load_sample(root, entry)
