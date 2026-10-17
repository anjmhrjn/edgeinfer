"""The deploy pipeline end to end on the tiny fixture model and a fake dataset, on ort_cpu
(runs in CI): a passing run is published to `dev`, a rerun skips or reuses every stage,
a preprocessing fault is blocked at the parity gate, and rollback moves the alias back."""

import json
import subprocess

import pytest
import yaml

from edgeinfer import reports
from edgeinfer.cli import main
from edgeinfer.data.cityscapes import list_image_ids
from edgeinfer.data.manifests import ManifestSpec, build_manifest, format_manifest
from edgeinfer.pipeline import registry, reuse, runner
from edgeinfer.pipeline.config import load_recipe
from edgeinfer.pipeline.runner import STAGES, Paths, Run
from edgeinfer.quant.config import QUANT_DIR
from tests.conftest import FIXTURES, TINY_CONFIG

HEAD = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                      check=True).stdout.strip()  # fmt: skip


@pytest.fixture
def env(fake_cityscapes, tmp_path, monkeypatch):
    """Golden outputs, manifests, a quant config and a gates file for the fixture model.
    Reports are treated as coming from a clean tree at HEAD, so reuse and publish behave
    as they do in CI whatever the state of the working tree."""
    monkeypatch.setattr(reports, "git_state", lambda: (HEAD, False))
    monkeypatch.setattr(runner, "git_state", lambda: (HEAD, False))
    monkeypatch.setattr(reuse, "code_unchanged_since", lambda sha: sha == HEAD)
    monkeypatch.delenv("CITYSCAPES_ROOT", raising=False)

    val = build_manifest(ManifestSpec("g", "val", 4, 0), list_image_ids(fake_cityscapes, "val"))
    golden_manifest = tmp_path / "golden_fake.txt"
    golden_manifest.write_text(format_manifest(val))
    train = build_manifest(ManifestSpec("c", "train", None, None),
                           list_image_ids(fake_cityscapes, "train"))  # fmt: skip
    calib = tmp_path / "calib_fake.txt"
    calib.write_text(format_manifest(train))
    qcfg = tmp_path / "int8_fake.yaml"
    qcfg.write_text((QUANT_DIR / "int8_entropy.yaml").read_text()
                    .replace("manifest: calib_train200", f"manifest: {calib}"))  # fmt: skip
    assert main(["golden", "--model", str(TINY_CONFIG), "--manifest", str(golden_manifest),
                 "--root", str(fake_cityscapes), "--out", str(tmp_path / "golden"),
                 "--hashes", str(tmp_path / "golden.sha256")]) == 0  # fmt: skip

    perf = tmp_path / "perf_fake.yaml"
    perf.write_text(yaml.safe_dump({
        "warmup_iterations": 1, "iterations": 4, "repeats": 1, "manifest": str(golden_manifest),
        "max_images": 2, "postprocess_size": "label", "max_repeat_p99_spread": 0.10,
        "gpu_clock_mhz": None,
    }))  # fmt: skip
    gates = tmp_path / "gates_fake.yaml"
    gates.write_text(yaml.safe_dump({
        "parity": {"manifest": str(golden_manifest), "inputs": "images"},
        "accuracy": {"manifest": str(golden_manifest), "precisions": ["int8", "int8_fp16"]},
        "perf": {"perf_config": str(perf)},
    }))  # fmt: skip
    paths = Paths(reports_dir=tmp_path / "reports", onnx_root=tmp_path / "onnx",
                  registry_dir=tmp_path / "registry", golden_dir=tmp_path / "golden",
                  calib_dir=tmp_path / "calib", root=str(fake_cityscapes))  # fmt: skip

    def recipe(name, precision="int8", model=TINY_CONFIG):
        quant = ({"config": str(qcfg), "exclusions": None,
                  "groups": str(FIXTURES / "tiny_fixture_groups.yaml")}
                 if precision == "int8" else None)  # fmt: skip
        path = tmp_path / "recipes" / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump({
            "model": str(model), "precision": precision, "quant": quant, "fp16": None,
            "build": None, "target": "ort_cpu", "gates": str(gates),
        }))  # fmt: skip
        return load_recipe(path)

    return paths, recipe


def _statuses(run):
    return {name: run.stages[name]["status"] for name in STAGES}


def test_passing_run_is_published_to_dev(env):
    paths, recipe = env
    run = Run(recipe("tiny_int8"), "20261017T000001Z", paths)
    assert run.run_all() == 0
    assert _statuses(run) == {
        "export": "ran", "convert": "ran", "validate": "ran", "push": "skipped",
        "build": "skipped", "parity": "ran", "accuracy": "ran", "perf": "ran",
        "publish": "ran",
    }  # fmt: skip
    # ORT CPU latency isn't a deployment latency: perf passes but isn't enforced, so the
    # deployment reaches `dev`, never `production`.
    assert run.stages["perf"]["enforced"] is False
    report = json.loads(run.state_path.read_text())
    assert report["passed"] and report["tool"] == "deploy"
    codes = [f["code"] for f in report["findings"]]
    assert codes == ["DEPLOY_GATE_NOT_ENFORCED", "REGISTRY_PUBLISHED"]
    verdict = run.stages["accuracy"]["verdict"]
    assert verdict["drop_points"] <= verdict["budget_points"]
    index = registry.load("tiny_fixture", paths.registry_dir)
    assert index["aliases"] == {"dev": 1}
    v1 = index["versions"][0]
    assert v1["artifact_id"] == run.artifacts["onnx"]["id"] and v1["precision"] == "int8"
    assert v1["gate_reports"]["parity"] == [f"deploy/tiny_int8/{run.run_id}/parity.json"]


def test_rerun_skips_or_reuses_every_stage(env):
    paths, recipe = env
    r = recipe("tiny_int8")
    assert Run(r, "20261017T000001Z", paths).run_all() == 0
    again = Run(r, "20261017T000002Z", paths)
    assert again.run_all() == 0
    statuses = _statuses(again)
    assert statuses.pop("publish") == "ran"
    assert set(statuses.values()) <= {"skipped", "reused"}
    assert again.stages["parity"]["checks"][0]["reused_from"] == "20261017T000001Z"
    assert again.metrics["published"] == {"version": 1, "alias": "dev", "new_version": False}


def test_changed_code_reruns_the_gates(env, monkeypatch):
    paths, recipe = env
    r = recipe("tiny_int8")
    assert Run(r, "20261017T000001Z", paths).run_all() == 0
    monkeypatch.setattr(reuse, "code_unchanged_since", lambda sha: False)
    again = Run(r, "20261017T000002Z", paths)
    assert again.run_all() == 0
    check = again.stages["parity"]["checks"][0]
    assert check["status"] == "ran" and check["reason"] == f"code changed since {HEAD[:7]}"


def test_preprocessing_fault_is_blocked_at_parity(env, tmp_path):
    paths, recipe = env
    bgr = tmp_path / "tiny_bgr.yaml"
    bgr.write_text(TINY_CONFIG.read_text().replace("channel_order: RGB", "channel_order: BGR"))
    run = Run(recipe("tiny_bgr", model=bgr), "20261017T000003Z", paths)
    assert run.run_all() == 1
    statuses = _statuses(run)
    assert statuses["validate"] == "ran"  # the export's contract is BGR too: consistent
    assert statuses["parity"] == "failed"
    assert {statuses[s] for s in STAGES[STAGES.index("parity") + 1 :]} == {"not_run"}
    (gate,) = [f for f in run.findings if f["code"] == "DEPLOY_GATE_FAILED"]
    assert gate["where"] == "parity" and "PARITY_INPUT_MISMATCH" in gate["message"]
    assert not json.loads(run.state_path.read_text())["passed"]
    assert not (paths.registry_dir / "tiny_fixture.json").exists()  # nothing published


def test_failed_check_is_never_reused(env, tmp_path):
    paths, recipe = env
    bgr = tmp_path / "tiny_bgr.yaml"
    bgr.write_text(TINY_CONFIG.read_text().replace("channel_order: RGB", "channel_order: BGR"))
    r = recipe("tiny_bgr", model=bgr)
    assert Run(r, "20261017T000003Z", paths).run_all() == 1
    again = Run(r, "20261017T000004Z", paths)
    assert again.run_all() == 1
    assert again.stages["parity"]["checks"][0]["status"] == "ran"


def test_stages_one_at_a_time_share_the_run_state(env):
    """How the Airflow DAG drives a run: one process per stage, same run id."""
    paths, recipe = env
    r = recipe("tiny_fp32", precision="fp32")
    assert Run(r, "20261017T000005Z", paths).run_stage("parity") == 2  # export not done yet
    for name in STAGES:
        assert Run(r, "20261017T000005Z", paths).run_stage(name) == 0, name
    run = Run(r, "20261017T000005Z", paths)
    assert run.stages["accuracy"]["status"] == "skipped"  # FP32 is the reference
    assert run.metrics["published"]["alias"] == "dev"
    assert run.run_stage("publish") == 0  # already passed: nothing to do


def test_dirty_tree_blocks_publish(env, monkeypatch):
    paths, recipe = env
    monkeypatch.setattr(runner, "git_state", lambda: (HEAD, True))
    run = Run(recipe("tiny_fp32", precision="fp32"), "20261017T000006Z", paths)
    assert run.run_all() == 1
    assert run.stages["publish"]["status"] == "failed"
    assert [f["code"] for f in run.findings if f["severity"] == "error"] == ["DEPLOY_DIRTY_TREE"]
