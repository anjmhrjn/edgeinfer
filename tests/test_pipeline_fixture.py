"""End-to-end CLI pipeline on the tiny fixture model and a fake dataset (runs in CI):
export -> validate -> ORT eval -> golden -> parity, plus the deliberate breaks."""

import json

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from edgeinfer import reports
from edgeinfer.cli import main
from edgeinfer.data.cityscapes import list_image_ids
from edgeinfer.data.manifests import ManifestSpec, build_manifest, format_manifest
from tests.conftest import TINY_CONFIG


@pytest.fixture
def pipeline(fake_cityscapes, tmp_path, monkeypatch):
    # Reports from tests must never land in artifacts/reports/.
    monkeypatch.setattr(reports, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.delenv("CITYSCAPES_ROOT", raising=False)
    manifest = tmp_path / "golden_fake.txt"
    entries = build_manifest(ManifestSpec("g", "val", 4, 0), list_image_ids(fake_cityscapes, "val"))
    manifest.write_text(format_manifest(entries))

    assert main(["export", "--model", str(TINY_CONFIG), "--out", str(tmp_path / "onnx"),
                 "--report-dir", str(tmp_path / "reports")]) == 0  # fmt: skip
    (onnx_path,) = (tmp_path / "onnx" / "tiny_fixture").glob("*.onnx")
    assert main(["golden", "--model", str(TINY_CONFIG), "--manifest", str(manifest),
                 "--root", str(fake_cityscapes), "--out", str(tmp_path / "golden"),
                 "--hashes", str(tmp_path / "golden.sha256")]) == 0  # fmt: skip
    common = ["--manifest", str(manifest), "--root", str(fake_cityscapes),
              "--golden-dir", str(tmp_path / "golden")]  # fmt: skip
    return tmp_path, onnx_path, common


def _report(path):
    return json.loads(path.read_text())


def test_fixture_pipeline_passes(pipeline, fake_cityscapes):
    tmp, onnx_path, common = pipeline
    assert main(["validate", "--model", str(TINY_CONFIG), "--artifact", str(onnx_path),
                 "--out", str(tmp / "validate.json")]) == 0  # fmt: skip
    assert main(["eval", "--model", str(TINY_CONFIG), "--backend", "ort_cpu",
                 "--artifact", str(onnx_path), "--manifest", common[1],
                 "--root", str(fake_cityscapes), "--out", str(tmp / "eval.json")]) == 0  # fmt: skip
    out = tmp / "parity.json"
    assert main(["parity", "--model", str(TINY_CONFIG), "--artifact", str(onnx_path),
                 *common, "--per-layer", "--out", str(out)]) == 0  # fmt: skip
    r = _report(out)
    assert r["passed"] and r["metrics"]["input_max_abs"] == 0.0
    assert r["metrics"]["per_layer"]["n_over_tolerance"] == 0
    assert r["thresholds"]["logits_max_abs"] == 1e-3


def test_channel_order_break_fails_loudly(pipeline):
    tmp, onnx_path, common = pipeline
    bgr = tmp / "tiny_bgr.yaml"
    bgr.write_text(TINY_CONFIG.read_text().replace("channel_order: RGB", "channel_order: BGR"))
    assert main(["validate", "--model", str(bgr), "--artifact", str(onnx_path),
                 "--out", str(tmp / "validate_bgr.json")]) == 1  # fmt: skip
    v = _report(tmp / "validate_bgr.json")
    assert [f["code"] for f in v["findings"]] == ["VALIDATE_PREPROC_MISMATCH"]

    out = tmp / "parity_bgr.json"
    assert main(["parity", "--model", str(bgr), "--artifact", str(onnx_path), *common,
                 "--per-layer", "--out", str(out)]) == 1  # fmt: skip
    r = _report(out)
    codes = {f["code"] for f in r["findings"]}
    assert {"PARITY_INPUT_MISMATCH", "PARITY_OUTPUT_DRIFT", "PARITY_LAYER_DRIFT"} <= codes
    (inp,) = [f for f in r["findings"] if f["code"] == "PARITY_INPUT_MISMATCH"]
    assert "channel_order" in inp["message"]
    # The first drifting tensor is the first conv: the fault enters at the input.
    first_conv = next(n for n in onnx.load(str(onnx_path)).graph.node if n.op_type == "Conv")
    assert r["metrics"]["per_layer"]["first_over_tolerance"]["tensor"] == first_conv.output[0]


def test_per_layer_localizes_a_corrupted_weight(pipeline):
    tmp, onnx_path, common = pipeline
    model = onnx.load(str(onnx_path))
    target = next(n for n in model.graph.node if n.op_type == "MatMul")  # inside attention
    weight = next(t for t in model.graph.initializer if t.name in target.input)
    w = numpy_helper.to_array(weight).copy()
    w.flat[0] += 0.5
    weight.CopyFrom(numpy_helper.from_array(w, weight.name))
    bad = tmp / "corrupted.onnx"
    onnx.save(model, str(bad))

    out = tmp / "parity_corrupt.json"
    rc = main(["parity", "--model", str(TINY_CONFIG), "--artifact", str(bad), "--precision",
               "fp32", "--inputs", "golden", *common, "--per-layer", "--ref-artifact",
               str(onnx_path), "--out", str(out)])  # fmt: skip
    assert rc == 1
    first = _report(out)["metrics"]["per_layer"]["first_over_tolerance"]
    assert first["tensor"] == target.output[0] and first["op"] == "MatMul"
    assert np.isfinite(first["cosine_min"])
