import hashlib
from dataclasses import replace

import numpy as np

from edgeinfer.config import load_model_config
from edgeinfer.data.cityscapes import cityscapes_root, iter_samples, list_image_ids
from edgeinfer.data.manifests import ManifestSpec, build_manifest
from edgeinfer.golden import format_hashes, golden_bytes, write_golden


class FixedBackend:
    name = "fake"

    def infer(self, x):
        # Deterministic function of the input so hashes depend on the data.
        return np.tile(x.mean(axis=1, keepdims=True), (1, 19, 1, 1))[:, :, ::4, ::4].copy()


def test_golden_bytes_are_deterministic():
    x = np.random.default_rng(0).standard_normal((1, 3, 4, 8)).astype(np.float32)
    logits = np.ones((1, 19, 1, 2), dtype=np.float32)
    assert golden_bytes(x, logits) == golden_bytes(x.copy(), logits.copy())


def test_write_golden_round_trip_and_stable_hashes(fake_cityscapes, tmp_path):
    cfg = load_model_config("segformer_b0")
    cfg = replace(cfg, input=replace(cfg.input, height=4, width=8))
    root = cityscapes_root(fake_cityscapes)
    entries = build_manifest(ManifestSpec("t", "val", 2, 1), list_image_ids(root, "val"))

    first = write_golden(cfg, FixedBackend(), iter_samples(root, entries), tmp_path / "a")
    second = write_golden(cfg, FixedBackend(), iter_samples(root, entries), tmp_path / "b")
    assert list(first.values()) == list(second.values())
    assert len(format_hashes(first).splitlines()) == 2

    path = tmp_path / "a" / f"{entries[0].image_id}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() in first.values()
    with np.load(path) as z:
        assert z["input"].shape == (1, 3, 4, 8)
        assert z["logits"].dtype == np.float32
