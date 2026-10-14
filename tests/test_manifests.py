import pytest

from edgeinfer.cli import main
from edgeinfer.data.cityscapes import cityscapes_root, iter_samples, list_image_ids
from edgeinfer.data.labels import NUM_CLASSES
from edgeinfer.data.manifests import (
    MANIFEST_SPECS,
    ManifestSpec,
    build_manifest,
    format_manifest,
    load_manifest,
    parse_manifest,
)

TRAIN_IDS = [f"city_{i:06d}_000019" for i in range(300)]


def test_sampling_is_deterministic_and_sorted():
    spec = ManifestSpec("t", split="train", size=50, seed=0)
    a = build_manifest(spec, TRAIN_IDS)
    b = build_manifest(spec, list(reversed(TRAIN_IDS)))  # input order must not matter
    assert a == b
    assert [e.image_id for e in a] == sorted(e.image_id for e in a)
    assert len(a) == 50


def test_different_seed_gives_different_sample():
    a = build_manifest(ManifestSpec("t", "train", 50, seed=0), TRAIN_IDS)
    b = build_manifest(ManifestSpec("t", "train", 50, seed=1), TRAIN_IDS)
    assert a != b


def test_full_split_manifest_keeps_everything():
    spec = ManifestSpec("t", split="val", size=None, seed=None)
    assert len(build_manifest(spec, TRAIN_IDS)) == len(TRAIN_IDS)


def test_rejects_duplicates_and_oversized_samples():
    with pytest.raises(ValueError, match="duplicate"):
        build_manifest(ManifestSpec("t", "train", None, None), ["a_1_1", "a_1_1"])
    with pytest.raises(ValueError, match="needs"):
        build_manifest(ManifestSpec("t", "train", 5, 0), ["a_1_1"])


def test_format_parse_round_trip():
    entries = build_manifest(ManifestSpec("t", "train", 10, 0), TRAIN_IDS)
    assert parse_manifest(format_manifest(entries)) == entries


def test_parse_rejects_malformed_lines():
    with pytest.raises(ValueError, match="line 2"):
        parse_manifest("val/a_1_1\nnot-a-manifest-line\n")


def test_calibration_never_uses_val():
    assert MANIFEST_SPECS["calib_train200"].split == "train"
    assert all(s.split == "val" for n, s in MANIFEST_SPECS.items() if n != "calib_train200")


def test_cityscapes_root_errors_are_specific(monkeypatch, tmp_path):
    monkeypatch.delenv("CITYSCAPES_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="CITYSCAPES_ROOT is not set"):
        cityscapes_root()
    with pytest.raises(FileNotFoundError, match="leftImg8bit"):
        cityscapes_root(tmp_path)


def test_loader_on_fake_dataset(fake_cityscapes):
    root = cityscapes_root(fake_cityscapes)
    ids = list_image_ids(root, "val")
    assert len(ids) == 6
    entries = build_manifest(ManifestSpec("t", "val", None, None), ids)
    samples = list(iter_samples(root, entries))
    assert samples[0].image.shape == (4, 6, 3)
    assert samples[0].label.shape == (4, 6)
    labels = samples[0].label
    assert ((labels < NUM_CLASSES) | (labels == 255)).all()


def test_cli_writes_manifests_and_refuses_silent_changes(fake_cityscapes, tmp_path, monkeypatch):
    monkeypatch.setitem(
        MANIFEST_SPECS, "calib_train200", ManifestSpec("calib_train200", "train", 4, 0)
    )
    monkeypatch.setitem(MANIFEST_SPECS, "golden_val32", ManifestSpec("golden_val32", "val", 2, 1))
    out = tmp_path / "manifests"
    args = ["manifests", "--root", str(fake_cityscapes), "--out", str(out)]
    assert main(args) == 0
    assert len(load_manifest(out / "eval_val.txt")) == 6
    assert main(args) == 0  # identical regeneration is fine
    (out / "eval_val.txt").write_text("val/zurich_000000_000019\n")
    assert main(args) == 1
    assert main([*args, "--force"]) == 0
