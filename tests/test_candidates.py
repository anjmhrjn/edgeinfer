"""Sensitivity configurations generated ahead of a TensorRT run (D-032)."""

import hashlib
import json
import shutil
from dataclasses import replace
from functools import partial

import pytest

from edgeinfer.analyze.candidates import (
    CandidateError,
    CandidateSet,
    generate,
    subsets,
    top_groups,
)
from edgeinfer.config import load_model_config
from edgeinfer.quant.calib import calibrate
from edgeinfer.quant.config import load_group_rules, load_quant_config
from edgeinfer.quant.mixed import convert_int8_fp16
from tests.conftest import TINY_CONFIG
from tests.test_mixed import FakeConvert
from tests.test_quant import TINY_GROUPS, _manifest


@pytest.fixture
def setup(fake_cityscapes, tmp_path, tiny_onnx):
    cfg = load_model_config(TINY_CONFIG)
    q = load_quant_config("int8_entropy")
    calib = _manifest(fake_cityscapes, tmp_path, "train")
    q = replace(q, calibration=replace(q.calibration, manifest=str(calib)))
    cal = calibrate(tiny_onnx, cfg, q, fake_cityscapes, tmp_path / "calib")
    return cfg, q, cal, load_group_rules("tiny_fixture", TINY_GROUPS), tmp_path, tiny_onnx


def test_subsets_smallest_first_in_ranking_order():
    assert subsets(["b", "a"]) == [(), ("b",), ("a",), ("b", "a")]
    assert len(subsets(list("abcde"))) == 32


def _ranking_report(model="m", ranking=("x", "y", "z")):
    return {"tool": "sensitivity", "model": model, "metrics": {"ranking": list(ranking)}}


def test_top_groups():
    assert top_groups(_ranking_report(), "m", 2) == ["x", "y"]
    for report, n in ((_ranking_report(model="other"), 2), (_ranking_report(), 4),
                      ({"tool": "parity", "model": "m"}, 1)):  # fmt: skip
        with pytest.raises(CandidateError) as e:
            top_groups(report, "m", n)
        assert e.value.code == "SENS_RANKING_MISMATCH"


def test_an_unconverted_candidate_fails_validation(setup):
    """A conversion that leaves FP32 compute in an int8_fp16 model must stop generation."""
    cfg, q, cal, rules, tmp, fp32 = setup
    fake = partial(convert_int8_fp16, runner=FakeConvert())
    with pytest.raises(CandidateError) as e:
        generate(fp32, cfg, q, cal, rules, ["attn"], tmp / "onnx", convert=fake)
    assert e.value.code == "SENS_CANDIDATE_INVALID" and "VALIDATE_FLOAT_PRECISION" in str(e.value)


def _report(tmp, configs, groups):
    return {"tool": "candidates", "model": "tiny_fixture",
            "metrics": {"groups": groups, "configs": configs}}  # fmt: skip


def test_candidate_lookup(tmp_path):
    d = tmp_path / "onnx" / "tiny_fixture"
    d.mkdir(parents=True)
    configs = []
    for excluded, aid in (([], "aaa"), (["x"], "bbb")):
        name = f"tiny_fixture__int8_fp16__{aid}.onnx"
        (d / name).write_bytes(b"onnx")
        meta = {"artifact_id": aid, "sha256": hashlib.sha256(b"onnx").hexdigest()}
        (d / name.replace(".onnx", ".meta.json")).write_text(json.dumps(meta))
        configs.append({"excluded": excluded, "int8_fp16": {"file": name, "artifact_id": aid},
                        "int8": {"file": "absent.onnx", "artifact_id": "ccc"}})  # fmt: skip
    cands = CandidateSet(_report(tmp_path, configs, ["x", "y"]), tmp_path / "onnx")
    assert cands.path(()).name.endswith("aaa.onnx")
    assert cands.path(("x",)).name.endswith("bbb.onnx")
    cases = [(("y",), "int8_fp16", "SENS_CANDIDATE_MISSING"),  # not generated
             ((), "int8", "SENS_CANDIDATE_MISSING")]  # listed, not on this machine  # fmt: skip
    for excluded, precision, code in cases:
        with pytest.raises(CandidateError) as e:
            cands.path(excluded, precision)
        assert e.value.code == code
    meta = {"artifact_id": "zzz", "sha256": hashlib.sha256(b"onnx").hexdigest()}
    (d / "tiny_fixture__int8_fp16__bbb.meta.json").write_text(json.dumps(meta))
    with pytest.raises(CandidateError) as e:
        cands.path(("x",))
    assert e.value.code == "SENS_CANDIDATE_MISMATCH"


@pytest.mark.modelopt
def test_generate_every_subset_with_the_real_conversion(setup):
    if shutil.which("uv") is None:
        pytest.fail("uv is needed to run the ModelOpt environment")
    cfg, q, cal, rules, tmp, fp32 = setup
    configs = generate(fp32, cfg, q, cal, rules, ["head", "stem"], tmp / "onnx")
    assert [c["excluded"] for c in configs] == [[], ["head"], ["stem"], ["head", "stem"]]
    assert len({c["int8_fp16"]["artifact_id"] for c in configs}) == 4
    cands = CandidateSet(_report(tmp, configs, ["head", "stem"]), tmp / "onnx")
    assert cands.path(("stem", "head")).name == configs[3]["int8_fp16"]["file"]  # order-free
    assert cands.path((), "int8").name == configs[0]["int8"]["file"]
