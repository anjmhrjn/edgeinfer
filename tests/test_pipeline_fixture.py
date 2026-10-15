"""End-to-end CLI pipeline on the tiny fixture model and a fake dataset (runs in CI):
export -> validate -> ORT eval -> golden -> parity, plus the deliberate breaks; then
quantize -> validate -> eval -> parity -> sensitivity -> exclusions, plus config faults."""

import json

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from edgeinfer import reports
from edgeinfer.cli import main
from edgeinfer.data.cityscapes import list_image_ids
from edgeinfer.data.manifests import ManifestSpec, build_manifest, format_manifest
from edgeinfer.quant.config import QUANT_DIR
from tests.conftest import FIXTURES, TINY_CONFIG

TINY_GROUPS = FIXTURES / "tiny_fixture_groups.yaml"


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


@pytest.fixture
def int8_setup(pipeline, fake_cityscapes):
    """Calibration manifest (fake train split) and a quant config pointing at it."""
    tmp, onnx_path, common = pipeline
    calib = tmp / "calib_fake.txt"
    train = build_manifest(ManifestSpec("c", "train", None, None),
                           list_image_ids(fake_cityscapes, "train"))  # fmt: skip
    calib.write_text(format_manifest(train))
    qcfg = tmp / "int8_fake.yaml"
    qcfg.write_text((QUANT_DIR / "int8_entropy.yaml").read_text()
                    .replace("manifest: calib_train200", f"manifest: {calib}"))  # fmt: skip
    quant = ["--model", str(TINY_CONFIG), "--artifact", str(onnx_path), "--quant-config",
             str(qcfg), "--groups", str(TINY_GROUPS), "--root", str(fake_cityscapes),
             "--calib-dir", str(tmp / "calib"), "--out-root", str(tmp / "onnx")]  # fmt: skip
    return tmp, common, quant, qcfg


def test_int8_pipeline_end_to_end(int8_setup, fake_cityscapes):
    tmp, common, quant, _ = int8_setup
    assert main(["quantize", *quant, "--out", str(tmp / "q.json")]) == 0
    q = _report(tmp / "q.json")
    assert q["precision"] == "int8" and q["metrics"]["placement"]["excluded_nodes"] == {}
    (int8_path,) = (tmp / "onnx" / "tiny_fixture").glob("*__int8__*.onnx")
    assert main(["validate", "--model", str(TINY_CONFIG), "--artifact", str(int8_path),
                 "--out", str(tmp / "v.json")]) == 0  # fmt: skip
    assert main(["eval", "--model", str(TINY_CONFIG), "--backend", "ort_cpu", "--artifact",
                 str(int8_path), "--manifest", common[1], "--root", str(fake_cityscapes),
                 "--out", str(tmp / "e.json")]) == 0  # fmt: skip
    assert _report(tmp / "e.json")["precision"] == "int8"
    fp32 = next(p for p in (tmp / "onnx" / "tiny_fixture").glob("*__fp32__*.onnx"))
    assert main(
        [
            "parity",
            "--model",
            str(TINY_CONFIG),
            "--artifact",
            str(int8_path),
            *common,
            "--per-layer",
            "--ref-artifact",
            str(fp32),
            "--out",
            str(tmp / "p.json"),
        ]
    ) in (0, 1)
    p = _report(tmp / "p.json")
    assert p["thresholds"]["logits_max_abs"] is None  # int8 tolerances: drift is report-only
    assert p["metrics"]["per_layer"]["n_tensors"] > 0
    assert {f["code"] for f in p["findings"]} <= {"PARITY_MIOU_DELTA", "PARITY_GOLDEN_UNVERIFIED"}

    # Sensitivity, then the suggested exclusions through the quantizer.
    exc = tmp / "exc.yaml"
    sens = [a for a in quant if a not in ("--out-root", str(tmp / "onnx"))]
    rc = main(
        [
            "sensitivity",
            *sens,
            "--manifest",
            common[1],
            "--out-root",
            str(tmp / "onnx"),
            "--margin",
            "0",
            "--write-exclusions",
            str(exc),
            "--out",
            str(tmp / "s.json"),
        ]
    )
    s = _report(tmp / "s.json")
    assert rc == (0 if s["passed"] else 1)
    m = s["metrics"]
    assert [r["group"] for r in m["groups"]] == ["stem", "attn", "head"]
    assert sorted(m["ranking"]) == ["attn", "head", "stem"]
    assert m["noise_floor_points"] >= 0 and m["noise_run"]["threads"] == 1
    assert main(["quantize", *quant, "--exclusions", str(exc), "--out", str(tmp / "qx.json")]) == 0
    assert len(_report(tmp / "qx.json")["metrics"]["excluded_nodes"]) == sum(
        {"stem": 2, "attn": 4, "head": 2}[g] for g in m["suggested_exclusions"]
    )


def test_config_faults_are_caught_with_their_codes(int8_setup, fake_cityscapes):
    tmp, common, quant, qcfg = int8_setup
    # 1. Calibration manifest swapped to val.
    swapped = tmp / "int8_val.yaml"
    swapped.write_text(qcfg.read_text().replace(str(tmp / "calib_fake.txt"), common[1]))
    args = [a if a != str(qcfg) else str(swapped) for a in quant]
    assert main(["quantize", *args, "--out", str(tmp / "f1.json")]) == 1
    assert [f["code"] for f in _report(tmp / "f1.json")["findings"]] == [
        "QUANT_CALIB_ON_EVAL_SPLIT"
    ]
    # 2. Exclusion list with a typo.
    typo = tmp / "typo.yaml"
    typo.write_text("name: typo\nmodel: tiny_fixture\ngroups: [atention]\nnodes: []\n")
    assert main(["quantize", *quant, "--exclusions", str(typo), "--out", str(tmp / "f2.json")]) == 1
    assert [f["code"] for f in _report(tmp / "f2.json")["findings"]] == ["QUANT_UNKNOWN_EXCLUSION"]
    # 3. Unsigned activations: quantizes, but the validator refuses it for TensorRT.
    uint8 = tmp / "int8_u8.yaml"
    uint8.write_text(qcfg.read_text().replace("activation_type: int8", "activation_type: uint8"))
    args = [a if a != str(qcfg) else str(uint8) for a in quant]
    assert main(["quantize", *args, "--out", str(tmp / "f3.json")]) == 0
    art = tmp / "onnx" / "tiny_fixture" / _report(tmp / "f3.json")["metrics"]["file"]
    assert main(["validate", "--model", str(TINY_CONFIG), "--artifact", str(art),
                 "--out", str(tmp / "f3v.json")]) == 1  # fmt: skip
    codes = {f["code"] for f in _report(tmp / "f3v.json")["findings"]}
    assert "VALIDATE_QDQ_DTYPE" in codes
