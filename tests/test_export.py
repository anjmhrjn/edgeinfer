import json

import numpy as np
import onnx
import pytest

from edgeinfer.artifacts import CONTRACT_KEY, io_contract, load_sidecar
from edgeinfer.cli import main
from edgeinfer.config import load_model_config
from edgeinfer.export.onnx_export import artifact_id, export_model
from tests.conftest import TINY_CONFIG


@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory):
    cfg = load_model_config(TINY_CONFIG)
    path, meta = export_model(cfg, tmp_path_factory.mktemp("onnx"))
    return cfg, path, meta


def test_export_writes_static_model_with_contract(tiny_export):
    cfg, path, meta = tiny_export
    model = onnx.load(str(path))
    assert [o.version for o in model.opset_import if o.domain == ""] == [20]
    (inp,) = model.graph.input
    (out,) = model.graph.output
    assert inp.name == "input" and out.name == "logits"
    assert [d.dim_value for d in inp.type.tensor_type.shape.dim] == [1, 3, 32, 64]
    assert [d.dim_value for d in out.type.tensor_type.shape.dim] == [1, 19, 8, 16]
    props = {p.key: p.value for p in model.metadata_props}
    assert json.loads(props[CONTRACT_KEY]) == io_contract(cfg)
    assert props["edgeinfer.artifact_id"] == meta["artifact_id"] == artifact_id(cfg)
    assert meta["sanity_max_abs"] < 1e-3 * meta["sanity_max_abs_ref"]


def test_export_strips_absolute_source_paths(tiny_export):
    _, path, _ = tiny_export
    assert b"stack_trace" not in path.read_bytes()


def test_existing_artifact_is_reused_and_checked(tiny_export):
    cfg, path, meta = tiny_export
    again, meta2 = export_model(cfg, path.parents[1])
    assert again == path and meta2["sha256"] == meta["sha256"]
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="modified"):
        load_sidecar(path)


def test_artifact_id_depends_on_config():
    cfg = load_model_config(TINY_CONFIG)
    from dataclasses import replace

    bgr = replace(cfg, input=replace(cfg.input, channel_order="BGR"))
    assert artifact_id(cfg) != artifact_id(bgr)


def test_export_cli_writes_report(tmp_path):
    rc = main(
        ["export", "--model", str(TINY_CONFIG), "--out", str(tmp_path / "onnx"),
         "--report-dir", str(tmp_path / "reports")]
    )  # fmt: skip
    assert rc == 0
    (report_path,) = (tmp_path / "reports").glob("export__tiny_fixture__*.json")
    report = json.loads(report_path.read_text())
    assert report["tool"] == "export" and report["metrics"]["op_counts"]["Conv"] == 4
    assert np.isfinite(report["metrics"]["sanity_max_abs"])
