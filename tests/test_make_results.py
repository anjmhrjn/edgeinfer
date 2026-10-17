"""scripts/make_results.py: every published number comes from one clean report (D-041)."""

import json

import pytest
import yaml

from edgeinfer.reports import REPO_ROOT


@pytest.fixture
def make_results(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    import make_results

    return make_results


def _report(tool, artifact, sha="aa" * 32, dirty=False, passed=True, **metrics):
    return {"tool": tool, "artifact_id": artifact, "git_dirty": dirty, "passed": passed,
            "inputs": {"artifact_sha256": sha, "manifest": "eval_val", "input_hw": [512, 1024],
                       "perf_config": {"gpu_clock_mhz": 990}},
            "env": {"device": "cuda:0 (Tesla T4, sm_75)", "versions": {"tensorrt": "11.3.0.99"}},
            "metrics": metrics}  # fmt: skip


def _eval(artifact, miou, held_out, **kw):
    return _report("eval", artifact, miou=miou, n_images=500,
                   subsets={"val_minus_sens100": {"miou": held_out}}, **kw)  # fmt: skip


def _perf(artifact, p99, **kw):
    spans = {"infer": {"p50": p99 - 0.1, "p99": p99}, "e2e": {"p99": 700.0}}
    return _report("perf", artifact, spans_ms=spans, device_used_mib_max=200.0,
                   artifact_size_bytes=6 * 2**20, **kw)  # fmt: skip


@pytest.fixture
def env(tmp_path):
    """Two engines of one model, a manifest naming their reports, and a README."""
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "eval_fp32.json": _eval("fp32id", 0.70, 0.69),
        "perf_fp32.json": _perf("fp32id", 20.0),
        "eval_int8.json": _eval("int8id", 0.695, 0.684),
        "perf_int8.json": _perf("int8id", 6.5),
        "parity_int8.json": _report("parity", "int8id"),
    }
    for name, report in files.items():
        (reports / name).write_text(json.dumps(report))
    manifest = {
        "models": {"m": "Model M"},
        "held_out": "val_minus_sens100",
        "main": {"m": [
            {"label": "FP32", "eval": "eval_fp32.json", "perf": "perf_fp32.json"},
            {"label": "INT8", "final": True, "eval": "eval_int8.json",
             "perf": "perf_int8.json", "parity": "parity_int8.json"},
        ]},
        "accuracy_chain": {}, "ort_int8": {}, "pareto": {}, "profile": {}, "deploy": [],
        "rerun": [],
        "readme": [{"model": "m", "label": "FP32"}, {"model": "m", "label": "INT8"}],
    }  # fmt: skip
    (tmp_path / "results.yaml").write_text(yaml.safe_dump(manifest))
    (tmp_path / "README.md").write_text("# x\n\n<!-- results:start -->\n<!-- results:end -->\n")
    return tmp_path


def _args(env, *extra):
    paths = {"--manifest": "results.yaml", "--reports-dir": "reports",
             "--results": "results.md", "--readme": "README.md"}  # fmt: skip
    return [a for flag, name in paths.items() for a in (flag, str(env / name))] + list(extra)


def _edit(env, name, change):
    path = env / "reports" / name
    report = json.loads(path.read_text())
    change(report)
    path.write_text(json.dumps(report))


def test_renders_and_check_passes_after_writing(make_results, env):
    assert make_results.main(_args(env)) == 0
    results = (env / "results.md").read_text()
    # INT8 vs FP32: 69.5 - 70.0 = -0.5 points, held out 68.4 - 69.0 = -0.6.
    assert "| **INT8 (final)** | 69.500 | −0.500 | −0.600 | 6.400 | 6.500 |" in results
    assert "| FP32 | 70.000 | — | — |" in results and "reference" in results
    assert "[eval_int8.json](../artifacts/reports/eval_int8.json)" in results
    readme = (env / "README.md").read_text()
    assert readme.startswith("# x\n\n<!-- results:start -->") and "| Model M | FP32 |" in readme
    assert make_results.main(_args(env, "--check")) == 0


def test_check_fails_when_a_generated_number_is_edited(make_results, env):
    make_results.main(_args(env))
    results = env / "results.md"
    results.write_text(results.read_text().replace("69.500", "69.900"))
    assert make_results.main(_args(env, "--check")) == 1


def test_check_fails_when_a_report_changes(make_results, env):
    make_results.main(_args(env))
    _edit(env, "eval_int8.json", lambda r: r["metrics"].update(miou=0.69))
    assert make_results.main(_args(env, "--check")) == 1


def test_missing_report_is_refused(make_results, env, capsys):
    (env / "reports" / "parity_int8.json").unlink()
    assert make_results.main(_args(env)) == 2
    assert "parity_int8.json: no such report" in capsys.readouterr().out


def test_dirty_report_is_refused(make_results, env, capsys):
    _edit(env, "perf_int8.json", lambda r: r.update(git_dirty=True))
    assert make_results.main(_args(env)) == 2
    assert "perf_int8.json: written from a dirty tree" in capsys.readouterr().out


def test_row_measuring_two_artifacts_is_refused_unless_explained(make_results, env, capsys):
    _edit(env, "perf_int8.json", lambda r: r.update(artifact_id="otherid"))
    assert make_results.main(_args(env)) == 2
    assert "measured different artifacts" in capsys.readouterr().out
    manifest = yaml.safe_load((env / "results.yaml").read_text())
    manifest["main"]["m"][1]["different_artifacts"] = "perf from a rebuild (reason)"
    (env / "results.yaml").write_text(yaml.safe_dump(manifest))
    assert make_results.main(_args(env)) == 0
    assert "- INT8: perf from a rebuild (reason)" in (env / "results.md").read_text()


def test_same_id_but_different_bytes_is_refused(make_results, env, capsys):
    # A rebuild keeps the engine id but can change the bytes (D-029, D-040).
    _edit(env, "perf_int8.json", lambda r: r["inputs"].update(artifact_sha256="cc" * 32))
    assert make_results.main(_args(env)) == 2
    assert "same artifact id but different bytes" in capsys.readouterr().out


def test_failed_gate_shows_in_the_row(make_results, env):
    _edit(env, "parity_int8.json", lambda r: r.update(passed=False))
    assert make_results.main(_args(env)) == 0
    assert "| fail: parity |" in (env / "results.md").read_text()


def test_wrong_image_set_is_refused(make_results, env, capsys):
    _edit(env, "eval_int8.json", lambda r: r["inputs"].update(manifest="sens_val100"))
    assert make_results.main(_args(env)) == 2
    assert "manifest is 'sens_val100', expected 'eval_val'" in capsys.readouterr().out


def test_committed_results_are_current(make_results):
    # The repo's own docs/results.md and README table match the committed reports.
    assert make_results.main(["--check"]) == 0
