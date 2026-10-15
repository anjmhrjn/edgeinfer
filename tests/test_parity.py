from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from edgeinfer.analyze import parity
from edgeinfer.analyze.parity import (
    compare_logits,
    diagnose_input,
    load_tolerances,
    run_layer_parity,
    run_parity,
)
from edgeinfer.backends import get_backend
from edgeinfer.config import load_model_config
from edgeinfer.data.cityscapes import cityscapes_root, iter_samples, list_image_ids
from edgeinfer.data.manifests import ManifestSpec, build_manifest
from edgeinfer.golden import verify_golden, write_golden
from edgeinfer.onnx_utils import layer_groups, layer_tensors
from tests.conftest import TINY_CONFIG

CFG = load_model_config(TINY_CONFIG)
TOL = load_tolerances("fp32")


@pytest.fixture
def golden_setup(fake_cityscapes, tmp_path):
    """Golden outputs for 3 fake val images from torch_ref on the tiny model."""
    root = cityscapes_root(fake_cityscapes)
    entries = build_manifest(ManifestSpec("g", "val", 3, 0), list_image_ids(root, "val"))
    ref = get_backend("torch_ref")
    ref.load(None, CFG)
    golden_dir = tmp_path / "golden"
    write_golden(CFG, ref, iter_samples(root, entries), golden_dir)
    return root, entries, golden_dir


def _ort(path, cfg=CFG):
    backend = get_backend("ort_cpu")
    backend.load(path, cfg)
    return backend


def _codes(findings):
    return {f["code"] for f in findings if f["severity"] == "error"}


def test_compare_identical_and_perturbed():
    ref = np.random.default_rng(0).standard_normal((1, 19, 4, 8)).astype(np.float32)
    m = compare_logits(ref, ref.copy())
    assert m["max_abs"] == 0 and m["cosine"] == pytest.approx(1.0) and m["argmax_agreement"] == 1
    cand = ref.copy()
    cand[0, :, 0, 0] = -ref[0, :, 0, 0]  # flips the class of one pixel
    m = compare_logits(ref, cand)
    assert m["argmax_agreement"] == pytest.approx(31 / 32) and m["max_abs"] > 0
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_logits(ref, ref[:, :, :2])


def test_diagnose_input_names_the_fault():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 255, (1, 3, 16, 32)).astype(np.float32)
    assert "channel_order" in diagnose_input(rgb[:, ::-1], rgb)
    assert "mean and std" in diagnose_input(rgb * 2 + 1, rgb)
    assert "resize" in diagnose_input(rgb[:, :, ::-1], rgb)


def test_ort_parity_passes_in_both_modes(golden_setup, tiny_onnx):
    root, entries, golden_dir = golden_setup
    for mode, samples in (("golden", None), ("images", iter_samples(root, entries))):
        metrics, findings, pairs = run_parity(
            CFG, entries, golden_dir, TOL, mode, backend=_ort(tiny_onnx), samples=samples
        )
        assert _codes(findings) == set(), findings
        assert metrics["n_images"] == 3 and len(pairs) == 3
    assert metrics["input_max_abs"] == 0.0
    assert abs(metrics["miou_delta_points"]) <= TOL.miou_delta_max


def test_channel_order_break_is_caught_and_diagnosed(golden_setup, tiny_onnx):
    root, entries, golden_dir = golden_setup
    bgr = replace(CFG, input=replace(CFG.input, channel_order="BGR"))
    _, findings, _ = run_parity(
        bgr, entries, golden_dir, TOL, "images",
        backend=_ort(tiny_onnx, bgr), samples=iter_samples(root, entries),
    )  # fmt: skip
    (f,) = [f for f in findings if f["code"] == "PARITY_INPUT_MISMATCH"]
    assert "channel_order" in f["message"] and "3/3 images" in f["message"]
    assert "PARITY_OUTPUT_DRIFT" in _codes(findings)


def test_npz_mode_drift_and_nonfinite(golden_setup, tmp_path):
    _, entries, golden_dir = golden_setup
    cand_dir = tmp_path / "cand"
    cand_dir.mkdir()
    for i, e in enumerate(entries):
        with np.load(golden_dir / f"{e.image_id}.npz") as z:
            logits = z["logits"] + 0.5
        if i == 0:
            logits[0, 0, 0, 0] = np.nan
        np.savez(cand_dir / f"{e.image_id}.npz", logits=logits)
    metrics, findings, _ = run_parity(CFG, entries, golden_dir, TOL, "npz", npz_dir=cand_dir)
    assert {"PARITY_OUTPUT_NONFINITE", "PARITY_OUTPUT_DRIFT"} <= _codes(findings)
    assert "PARITY_MIOU_SKIPPED" in {f["code"] for f in findings}
    assert metrics["logits_max_abs"] == pytest.approx(0.5, rel=1e-5)


def test_per_layer_finds_the_perturbed_layer(tiny_onnx, tmp_path):
    model = onnx.load(str(tiny_onnx))
    convs = [n for n in model.graph.node if n.op_type == "Conv"]
    target = convs[1]  # second conv of the stem
    init = next(t for t in model.graph.initializer if t.name == target.input[1])
    init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init) * 1.5, init.name))
    bad_path = tmp_path / "perturbed.onnx"
    onnx.save(model, str(bad_path))

    ref_model = onnx.load(str(tiny_onnx))
    ops = {n.output[0]: n.op_type for n in ref_model.graph.node}
    names = layer_tensors(ref_model)
    x = np.random.default_rng(0).standard_normal((1, 3, 32, 64)).astype(np.float32)
    metrics, findings = run_layer_parity(
        _ort(tiny_onnx), _ort(bad_path), [(n, ops[n]) for n in names],
        layer_groups(ref_model, names, 1 << 30), [(x, x)], TOL,
    )  # fmt: skip
    first = metrics["first_over_tolerance"]
    assert first["tensor"] == target.output[0] and first["op"] == "Conv"
    assert all(r["rel_max_abs"] == 0 for r in metrics["tensors"][: first["index"]])
    (f,) = findings
    assert f["code"] == "PARITY_LAYER_DRIFT" and f["where"] == target.output[0]


def test_layer_parity_chunked_matches_unchunked(tiny_onnx):
    model = onnx.load(str(tiny_onnx))
    ops = {n.output[0]: n.op_type for n in model.graph.node}
    names = layer_tensors(model)
    x = np.random.default_rng(0).standard_normal((1, 3, 32, 64)).astype(np.float32)
    results = []
    for max_bytes in (1 << 30, 4096):
        groups = layer_groups(model, names, max_bytes)
        m, _ = run_layer_parity(_ort(tiny_onnx), _ort(tiny_onnx), [(n, ops[n]) for n in names],
                                groups, [(x, x)], TOL)  # fmt: skip
        results.append(m["tensors"])
    assert results[0] == results[1]


def test_verify_golden(golden_setup, monkeypatch, tmp_path):
    from edgeinfer import golden

    _, entries, golden_dir = golden_setup
    ids = [e.image_id for e in entries]
    assert verify_golden("tiny_fixture", golden_dir, ids) is False  # outside the repo
    monkeypatch.setattr(golden, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(golden, "HASH_DIR", tmp_path / "hashes")
    hashes = {}
    import hashlib

    for i in ids:
        p = golden_dir / f"{i}.npz"
        hashes[str(p.resolve().relative_to(tmp_path))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "hashes").mkdir()
    (tmp_path / "hashes" / "tiny_fixture.sha256").write_text(golden.format_hashes(hashes))
    assert verify_golden("tiny_fixture", golden_dir, ids) is True
    p = golden_dir / f"{ids[0]}.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="golden outputs changed"):
        verify_golden("tiny_fixture", golden_dir, ids)


def test_tolerances_are_strict(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("fp32:\n  logits_max_abs: 1.0\n")
    with pytest.raises(ValueError, match="expected keys"):
        load_tolerances("fp32", path)
    with pytest.raises(KeyError, match="fp16"):
        load_tolerances("fp16", parity.TOLERANCES_PATH)


def test_int8_reports_drift_without_gating_it(golden_setup, tmp_path):
    _, entries, golden_dir = golden_setup
    cand_dir = tmp_path / "cand"
    cand_dir.mkdir()
    for e in entries:
        with np.load(golden_dir / f"{e.image_id}.npz") as z:
            np.savez(cand_dir / f"{e.image_id}.npz", logits=z["logits"] + 0.5)
    tol = load_tolerances("int8")
    assert tol.logits_max_abs is None and tol.miou_delta_max == 1.0
    metrics, findings, _ = run_parity(CFG, entries, golden_dir, tol, "npz", npz_dir=cand_dir)
    assert metrics["logits_max_abs"] == pytest.approx(0.5, rel=1e-5)  # still reported
    assert not any(f["severity"] == "error" for f in findings)
