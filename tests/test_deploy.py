"""The deploy pipeline end to end on the tiny fixture model and a fake dataset, on ort_cpu
(runs in CI): a passing run is published to `dev`, a rerun skips or reuses every stage,
a preprocessing fault is blocked at the parity gate, and rollback moves the alias back."""

import dataclasses
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
    monkeypatch.setattr(reports, "REPORTS_DIR", tmp_path / "reports")  # never artifacts/reports/
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


@pytest.fixture
def cli(env, monkeypatch):
    """`edgeinfer deploy ...` through the CLI, writing under the test's directories."""
    paths, recipe = env
    monkeypatch.setattr(runner, "Paths", lambda root=None: paths)
    return paths, recipe


def test_cli_run_dry_run_rollback_and_status(cli, capsys):
    paths, recipe = cli
    recipe("tiny_fp32", precision="fp32")  # writes the recipe files
    recipe("tiny_int8")
    recipes = paths.reports_dir.parent / "recipes"
    reg = ["--registry-dir", str(paths.registry_dir)]

    assert main(["deploy", "run", "--recipe", str(recipes / "tiny_fp32.yaml")]) == 0
    assert main(["deploy", "run", "--recipe", str(recipes / "tiny_int8.yaml")]) == 0
    assert registry.load("tiny_fixture", paths.registry_dir)["aliases"] == {"dev": 2}
    capsys.readouterr()

    # A dry run after a run: every gate check would be reused, and nothing runs.
    n_runs = len(list((paths.reports_dir / "deploy").glob("*/*/deploy.json")))
    assert main(["deploy", "run", "--recipe", str(recipes / "tiny_int8.yaml"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    rows = [line.split() for line in out.splitlines()[1:]]
    gates = {r[0]: r[1] for r in rows if "/" in r[0]}
    assert gates == {"validate/validate": "reuse", "parity/parity": "reuse",
                     "accuracy/eval_fp32": "reuse", "accuracy/eval": "reuse",
                     "perf/perf": "reuse"}  # fmt: skip
    assert len(list((paths.reports_dir / "deploy").glob("*/*/deploy.json"))) == n_runs

    out_report = paths.reports_dir / "rollback.json"
    assert main(["deploy", "rollback", "--model", "tiny_fixture", "--alias", "dev",
                 "--reason", "int8 regression", "--out", str(out_report), *reg]) == 0  # fmt: skip
    r = json.loads(out_report.read_text())
    assert r["metrics"]["from_version"] == 2 and r["metrics"]["to_version"] == 1
    assert (
        r["artifact_id"]
        == registry.load("tiny_fixture", paths.registry_dir)["versions"][0]["artifact_id"]
    )
    assert [f["code"] for f in r["findings"]] == ["REGISTRY_ROLLBACK"]
    assert main(["deploy", "rollback", "--model", "tiny_fixture", "--alias", "dev",
                 "--reason", "again", "--out", str(out_report), *reg]) == 1  # fmt: skip
    assert [f["code"] for f in json.loads(out_report.read_text())["findings"]] == [
        "REGISTRY_NO_PREVIOUS"
    ]
    capsys.readouterr()
    assert main(["deploy", "status", "--model", "tiny_fixture", *reg]) == 0
    status = capsys.readouterr().out
    assert "v1  fp32" in status and "<- dev" in status and "rolled back from dev" in status


def test_cli_stage_needs_a_run_id(cli):
    paths, recipe = cli
    recipe("tiny_fp32", precision="fp32")
    path = paths.reports_dir.parent / "recipes" / "tiny_fp32.yaml"
    assert main(["deploy", "run", "--recipe", str(path), "--stage", "parity"]) == 2
    assert main(["deploy", "run", "--recipe", str(path), "--run-id", "../x"]) == 2


class StubGpu:
    """Stands in for the instance: answers build, parity, eval and perf with valid reports,
    so the runner's TensorRT path is tested without a GPU. Engines are named after the
    ONNX file they come from; the FP32 reference scores 0.70 mIoU, the candidate 0.695."""

    remote = True

    def __init__(self):
        self.pushed, self.host_commands, self.commands = [], [], []
        self.precision = {}  # engine -> precision, from its build

    def dataset_args(self):
        return []

    def push(self, paths):
        self.pushed.append(paths)
        return 0

    def host(self, command):
        self.host_commands.append(command)
        return 0

    def run(self, argv):
        from pathlib import Path

        from edgeinfer.artifacts import load_sidecar
        from edgeinfer.reports import make_report, write_report

        self.commands.append(argv[0])
        args = dict(zip(argv[1::2], argv[2::2], strict=False))
        out = Path(args["--out"])
        if argv[0] == "build":
            onnx = load_sidecar(Path(args["--artifact"]))
            engine = f"e_{onnx['artifact_id'][:8]}"
            metrics = {"file": f"{engine}.engine", "sha256": f"sha_{engine}",
                       "reused": engine in self.precision}  # fmt: skip
            self.precision[engine] = onnx["precision"]
            meta = {"artifact_id": engine, "precision": onnx["precision"]}
            inputs = {}
        else:
            engine = Path(args["--artifact"]).stem
            meta = {"artifact_id": engine, "precision": "x"}
            inputs = {"artifact_sha256": f"sha_{engine}"}
            metrics = {"miou": 0.70 if self.precision[engine] == "fp32" else 0.695,
                       "n_images": 500}  # fmt: skip
        report = make_report(tool=argv[0], model="tiny_fixture",
                             backend_meta={"backend": "trt", "device": "stub", **meta},
                             inputs=inputs, metrics=metrics)  # fmt: skip
        write_report(report, out)
        return 0


def test_trt_target_with_a_stub_gpu(env):
    paths, recipe = env
    r = dataclasses.replace(recipe("tiny_int8"), target="trt", build="build")
    gpu = StubGpu()
    run = Run(r, "20261017T000007Z", paths, gpu=gpu)
    assert run.run_all() == 0
    # The candidate and the FP32 reference (for the accuracy gate) go to the instance...
    onnx = [run.artifacts[a]["path"] for a in ("onnx", "fp32")]
    assert gpu.pushed == [sorted(onnx + [p.replace(".onnx", ".meta.json") for p in onnx])]
    # ...both are built there, and every gate measures the engine, not the ONNX model.
    assert gpu.commands == ["build", "build", "parity", "eval", "eval", "perf"]
    engine = run.artifacts["engine"]
    assert engine["path"].startswith("artifacts/engines/tiny_fixture/e_")
    assert run.stages["parity"]["checks"][0]["key"]["artifact"]["id"] == engine["id"]
    assert run.stages["accuracy"]["verdict"]["drop_points"] == pytest.approx(0.5)
    assert gpu.host_commands == []  # the gates file's perf config locks no clock
    # Every gate was enforced, so the deployment reaches production, at its instance path.
    assert run.metrics["published"]["alias"] == "production"
    (v1,) = registry.load("tiny_fixture", paths.registry_dir)["versions"]
    assert v1["location"] == f"instance:{engine['path']}" and v1["sha256"] == engine["sha256"]

    again = Run(r, "20261017T000008Z", paths, gpu=gpu)
    gpu.commands.clear()
    assert again.run_all() == 0
    assert gpu.commands == ["build", "build"]  # builds reuse their engines; gates reuse reports
    assert again.stages["build"]["status"] == "skipped"
    assert {again.stages[g]["status"] for g in ("validate", "parity", "accuracy", "perf")} == {
        "reused"
    }
