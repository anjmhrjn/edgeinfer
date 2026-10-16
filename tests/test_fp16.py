"""FP32 -> FP16 ONNX conversion (D-024).

Most tests replace AutoCast with a fake runner, so they run in CI without the ModelOpt
environment. Tests marked `modelopt` run the real AutoCast in tools/modelopt/.
"""

import json
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from edgeinfer.artifacts import CONTRACT_KEY, META_PREFIX, load_sidecar
from edgeinfer.cli import main
from edgeinfer.config import ConfigError, load_model_config
from edgeinfer.data.cityscapes import list_image_ids
from edgeinfer.data.manifests import ManifestSpec, build_manifest, format_manifest
from edgeinfer.quant import QuantError
from edgeinfer.quant import fp16 as fp16_mod
from edgeinfer.quant.fp16 import (
    Fp16Calibration,
    Fp16Config,
    convert_fp16,
    load_fp16_config,
    precision_summary,
)
from tests.conftest import TINY_CONFIG

CFG = load_model_config(TINY_CONFIG)


def _manifest(root, tmp_path, split):
    path = tmp_path / f"{split}.txt"
    ids = list_image_ids(root, split)
    path.write_text(format_manifest(build_manifest(ManifestSpec("m", split, None, None), ids)))
    return path


def _config(manifest, limit=4):
    base = load_fp16_config("fp16_autocast")
    return replace(base, calibration=Fp16Calibration(manifest=str(manifest), limit=limit))


class FakeAutoCast:
    """Stands in for the subprocess: copies the model and drops its metadata, as AutoCast does."""

    def __init__(self):
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)
        model = onnx.load(args["onnx_path"])
        del model.metadata_props[:]
        onnx.save(model, args["output_path"])
        return {"versions": {"nvidia-modelopt": "fake"}, "seconds": 0.0}


@pytest.fixture
def setup(fake_cityscapes, tmp_path, tiny_onnx):
    return fake_cityscapes, tmp_path, tiny_onnx


def _convert(setup, fcfg, runner):
    root, tmp_path, src = setup
    return convert_fp16(src, CFG, fcfg, root, out_root=tmp_path / "onnx",
                        calib_root=tmp_path / "calib", runner=runner)  # fmt: skip


def test_shipped_config_loads():
    fcfg = load_fp16_config("fp16_autocast")
    assert fcfg.keep_io_types and fcfg.calibration.manifest == "calib_train200"


@pytest.mark.parametrize(
    "change",
    [{"low_precision_type": "bf16"}, {"data_max": 0}, {"keep_io_types": "yes"},
     {"nodes_to_exclude": "Conv_0"}, {"typo": 1}, {"calibration": {"manifest": "m", "limit": 0}}],
)  # fmt: skip
def test_config_is_strict(change):
    d = {**load_fp16_config("fp16_autocast").to_dict(), **change}
    with pytest.raises(ConfigError):
        Fp16Config.from_dict(d, where="test")


def test_conversion_restores_the_contract_and_writes_a_sidecar(setup):
    root, tmp_path, src = setup
    runner = FakeAutoCast()
    path, meta = _convert(setup, _config(_manifest(root, tmp_path, "train")), runner)
    assert "__fp16__" in path.name and meta["precision"] == "fp16"
    props = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    src_props = {p.key: p.value for p in onnx.load(str(src)).metadata_props}
    assert props[CONTRACT_KEY] == src_props[CONTRACT_KEY]  # dropped by AutoCast, restored
    assert props[META_PREFIX + "precision"] == "fp16"
    assert props[META_PREFIX + "parent_artifact_id"] == load_sidecar(src)["artifact_id"]
    assert load_sidecar(path)["artifact_id"] == meta["artifact_id"]
    # AutoCast got one NPZ per calibration image, preprocessed for the model.
    npz = sorted((tmp_path / "calib").rglob("*.npz"))
    assert len(npz) == 4
    assert np.load(npz[0])["input"].shape == (1, 3, CFG.input.height, CFG.input.width)
    assert runner.calls[0]["keep_io_types"] is True


def test_existing_artifact_is_reused(setup):
    root, tmp_path, _ = setup
    fcfg = _config(_manifest(root, tmp_path, "train"))
    runner = FakeAutoCast()
    first, _ = _convert(setup, fcfg, runner)
    second, _ = _convert(setup, fcfg, runner)
    assert first == second and len(runner.calls) == 1
    third, _ = _convert(setup, replace(fcfg, data_max=256.0), runner)
    assert third != first and len(runner.calls) == 2


def test_calibrating_on_val_is_refused(setup):
    root, tmp_path, _ = setup
    runner = FakeAutoCast()
    with pytest.raises(QuantError) as e:
        _convert(setup, _config(_manifest(root, tmp_path, "val")), runner)
    assert e.value.code == "QUANT_CALIB_ON_EVAL_SPLIT" and not runner.calls


def test_only_fp32_exports_are_converted(setup):
    root, tmp_path, _ = setup
    fcfg = _config(_manifest(root, tmp_path, "train"))
    fp16_path, _ = _convert(setup, fcfg, FakeAutoCast())
    with pytest.raises(QuantError) as e:
        _convert((root, tmp_path, fp16_path), fcfg, FakeAutoCast())
    assert e.value.code == "QUANT_INPUT_NOT_FP32"


def test_autocast_failure_is_a_finding(monkeypatch):
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="Traceback ...\nBoom")

    monkeypatch.setattr(fp16_mod.subprocess, "run", failed)
    with pytest.raises(QuantError) as e:
        fp16_mod.run_autocast_subprocess({})
    assert e.value.code == "FP16_AUTOCAST_FAILED" and "Boom" in str(e.value)

    def killed(*args, **kwargs):
        return subprocess.CompletedProcess(args, 137, stdout="INFO - Running ...", stderr="")

    monkeypatch.setattr(fp16_mod.subprocess, "run", killed)
    with pytest.raises(QuantError, match="out of memory"):
        fp16_mod.run_autocast_subprocess({})


def test_precision_summary():
    # float input -> Cast(fp16) -> Relu (fp16) -> Cast(float) -> Relu (fp32) -> float output
    nodes = [
        helper.make_node("Cast", ["x"], ["x16"], to=TensorProto.FLOAT16, name="in_cast"),
        helper.make_node("Relu", ["x16"], ["r16"], name="relu16"),
        helper.make_node("Cast", ["r16"], ["r32"], to=TensorProto.FLOAT, name="out_cast"),
        helper.make_node("Relu", ["r32"], ["y"], name="relu32"),
    ]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )  # fmt: skip
    summary = precision_summary(helper.make_model(graph))
    assert summary == {"fp16_nodes": 1, "fp16_by_op": {"Relu": 1}, "fp32_nodes": ["relu32"],
                       "cast_nodes": 2}  # fmt: skip


@pytest.mark.modelopt
def test_real_autocast_on_tiny_model(setup):
    """The real AutoCast in tools/modelopt/, then the validator and ORT on its output."""
    if shutil.which("uv") is None:
        pytest.fail("uv is needed to run the ModelOpt environment")
    root, tmp_path, src = setup
    manifest = _manifest(root, tmp_path, "train")
    fcfg_path = tmp_path / "fp16.yaml"
    fcfg_path.write_text(json.dumps(_config(manifest).to_dict()))  # JSON is valid YAML
    out = tmp_path / "report.json"
    argv = ["fp16", "--model", str(TINY_CONFIG), "--artifact", str(src),
            "--fp16-config", str(fcfg_path), "--root", str(root),
            "--out-root", str(tmp_path / "onnx"), "--calib-dir", str(tmp_path / "calib"),
            "--out", str(out)]  # fmt: skip
    assert main(argv) == 0
    report = json.loads(out.read_text())
    metrics, versions = report["metrics"], report["env"]["versions"]
    assert metrics["placement"]["fp16_nodes"] > 0
    # The ModelOpt environment's versions never overwrite the main environment's.
    assert versions["onnx"] == onnx.__version__
    assert versions["modelopt_env.onnx"] != onnx.__version__
    assert versions["modelopt_env.nvidia-modelopt"]
    fp16_path = tmp_path / "onnx" / CFG.name / metrics["file"]
    assert main(["validate", "--model", str(TINY_CONFIG), "--artifact", str(fp16_path),
                 "--out", str(tmp_path / "validate.json")]) == 0  # fmt: skip
    x = np.random.default_rng(0).standard_normal((1, 3, 32, 64)).astype(np.float32)
    ref = ort.InferenceSession(str(src)).run(None, {"input": x})[0]
    got = ort.InferenceSession(str(fp16_path)).run(None, {"input": x})[0]
    assert got.dtype == np.float32  # keep_io_types
    assert np.abs(got - ref).max() < 1e-2 * max(1.0, float(np.abs(ref).max()))
