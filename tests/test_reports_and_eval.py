import json

import jsonschema
import numpy as np
import pytest

from edgeinfer.config import load_model_config
from edgeinfer.data.cityscapes import cityscapes_root, iter_samples, list_image_ids
from edgeinfer.data.manifests import ManifestSpec, build_manifest
from edgeinfer.eval.harness import evaluate
from edgeinfer.reports import make_report, validate_report, write_report

META = {
    "backend": "fake",
    "precision": "fp32",
    "device": "cpu",
    "artifact_id": "x",
    "versions": {},
}


class OracleBackend:
    """Returns logits whose argmax is the given class everywhere."""

    name = "fake"

    def __init__(self, cls: int) -> None:
        self.cls = cls

    def infer(self, x):
        logits = np.zeros((1, 19, x.shape[2] // 4, x.shape[3] // 4), dtype=np.float32)
        logits[0, self.cls] = 1.0
        return logits


def test_report_round_trip(tmp_path):
    report = make_report(
        tool="eval", model="m", backend_meta=META, inputs={}, metrics={"miou": 0.5}
    )
    path = write_report(report, tmp_path / "r.json")
    loaded = json.loads(path.read_text())
    validate_report(loaded)
    assert len(loaded["git_sha"]) == 40
    assert loaded["env"]["versions"]["numpy"]


def test_report_rejects_bad_finding():
    report = make_report(tool="eval", model="m", backend_meta=META, inputs={}, metrics={})
    report["findings"] = [{"severity": "fatal", "code": "X", "message": "m"}]
    with pytest.raises(jsonschema.ValidationError):
        validate_report(report)


def test_report_refuses_nan(tmp_path):
    report = make_report(
        tool="eval", model="m", backend_meta=META, inputs={}, metrics={"miou": float("nan")}
    )
    with pytest.raises(ValueError):
        write_report(report, tmp_path / "r.json")


def test_harness_end_to_end_on_fake_data(fake_cityscapes):
    from dataclasses import replace

    cfg = load_model_config("segformer_b0")
    cfg = replace(cfg, input=replace(cfg.input, height=4, width=8))
    root = cityscapes_root(fake_cityscapes)
    entries = build_manifest(ManifestSpec("t", "val", None, None), list_image_ids(root, "val"))
    metrics, findings = evaluate(cfg, OracleBackend(0), iter_samples(root, entries))
    assert metrics["n_images"] == 6
    assert 0.0 <= metrics["miou"] <= 1.0
    assert metrics["per_class_iou"]["road"] is not None
    # Only "road" is ever predicted; classes missing from the random labels are flagged.
    assert all(f["code"] == "EVAL_CLASS_ABSENT" for f in findings)
