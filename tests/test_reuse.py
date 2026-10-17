import json

from edgeinfer.pipeline import reuse
from edgeinfer.pipeline.reuse import _report_matches, check_key, find_reusable

KEY = {"check": "parity", "artifact": {"id": "a", "sha256": "s"}, "settings": {}, "files": {}}


def _report(**over):
    return {"passed": True, "git_dirty": False, "git_sha": "abc", "artifact_id": "a",
            "inputs": {"artifact_sha256": "s"}} | over  # fmt: skip


def test_unknown_commit_never_counts_as_unchanged_code():
    assert reuse.code_unchanged_since("0" * 40) is False


def test_report_checks(monkeypatch):
    monkeypatch.setattr(reuse, "code_unchanged_since", lambda sha: sha == "abc")
    assert _report_matches(_report(), KEY) is None
    assert _report_matches(_report(passed=False), KEY) == "the check failed"
    assert "dirty" in _report_matches(_report(git_dirty=True), KEY)
    assert "another artifact" in _report_matches(_report(artifact_id="b"), KEY)
    assert "same id" in _report_matches(_report(inputs={"artifact_sha256": "t"}), KEY)
    assert _report_matches(_report(git_sha="def"), KEY) == "code changed since def"


def test_key_hashes_file_contents(tmp_path):
    f = tmp_path / "tolerances.yaml"
    f.write_text("a: 1\n")
    k1 = check_key("parity", {"id": "a", "sha256": "s"}, {}, [f])
    f.write_text("a: 2\n")
    assert check_key("parity", {"id": "a", "sha256": "s"}, {}, [f]) != k1


def test_lookup_skips_missing_reports_and_failed_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse, "code_unchanged_since", lambda sha: True)
    run = tmp_path / "deploy" / "r" / "20261017T000001Z"
    run.mkdir(parents=True)
    checks = [{"key": KEY, "passed": False, "report": "failed.json"},
              {"key": KEY, "passed": True, "report": "gone.json"}]  # fmt: skip
    state = {"metrics": {"run_id": "20261017T000001Z", "stages": {"parity": {"checks": checks}}}}
    (run / "deploy.json").write_text(json.dumps(state))
    found = find_reusable(KEY, tmp_path)
    assert found.report is None and "missing" in found.reason
    (tmp_path / "gone.json").write_text(json.dumps(_report()))
    found = find_reusable(KEY, tmp_path)
    assert found.report == "gone.json" and found.run_id == "20261017T000001Z"
