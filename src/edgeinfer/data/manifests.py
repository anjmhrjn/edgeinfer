"""Fixed image lists (manifests) for evaluation, calibration and golden outputs.

A manifest is a text file with one `<split>/<image_id>` per line, e.g.
`val/frankfurt_000000_000294`. Only manifests are committed, never images.
Sampling is seeded and done over sorted IDs, so regenerating from the same
dataset always gives the same files.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parents[3] / "data" / "manifests"


@dataclass(frozen=True)
class ManifestSpec:
    name: str
    split: str
    size: int | None  # None = every image in the split
    seed: int | None


MANIFEST_SPECS: dict[str, ManifestSpec] = {
    spec.name: spec
    for spec in (
        ManifestSpec("eval_val", split="val", size=None, seed=None),
        # Calibration comes from train: calibrating on val would leak eval data.
        ManifestSpec("calib_train200", split="train", size=200, seed=0),
        ManifestSpec("golden_val32", split="val", size=32, seed=1),
    )
}


@dataclass(frozen=True)
class ManifestEntry:
    split: str
    image_id: str  # e.g. "frankfurt_000000_000294"

    @property
    def city(self) -> str:
        return self.image_id.split("_", 1)[0]

    def __str__(self) -> str:
        return f"{self.split}/{self.image_id}"


def build_manifest(spec: ManifestSpec, image_ids: list[str]) -> list[ManifestEntry]:
    """Select entries for `spec` from all image IDs of its split. Output is sorted."""
    ids = sorted(set(image_ids))
    if len(ids) != len(image_ids):
        raise ValueError(f"{spec.name}: duplicate image IDs in {spec.split}")
    if spec.size is not None:
        if spec.size > len(ids):
            raise ValueError(f"{spec.name}: needs {spec.size} images, split has {len(ids)}")
        ids = sorted(random.Random(spec.seed).sample(ids, spec.size))
    return [ManifestEntry(spec.split, i) for i in ids]


def format_manifest(entries: list[ManifestEntry]) -> str:
    return "".join(f"{e}\n" for e in entries)


def parse_manifest(text: str) -> list[ManifestEntry]:
    entries = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        split, sep, image_id = line.partition("/")
        if not sep or not image_id or "/" in image_id:
            raise ValueError(f"line {lineno}: expected '<split>/<image_id>', got {line!r}")
        entries.append(ManifestEntry(split, image_id))
    return entries


def load_manifest(name_or_path: str | Path) -> list[ManifestEntry]:
    """Load a manifest by name (e.g. "eval_val") or by file path."""
    path = Path(name_or_path)
    if path.suffix != ".txt":
        path = MANIFEST_DIR / f"{name_or_path}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    return parse_manifest(path.read_text())
