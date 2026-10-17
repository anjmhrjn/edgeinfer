"""Gate reuse (D-040): when an earlier passing check can stand in for running it again.

A check (validate, parity, an eval, perf) is identified by a key: the check's name, the
artifact it measured (id and file sha256: an engine rebuilt with the same id can pick
different kernels), its settings from the gates file, and the content hash of every
config, manifest and hash file it reads. Every deploy run records each check's key next to
its report. A later check with the same key reuses that report if it passed, comes from a
clean tree, and the code that produced it hasn't changed since (`CODE_PATHS`, compared
with the working tree, so uncommitted changes count). Failed checks are never reused.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edgeinfer.artifacts import sha256_file
from edgeinfer.pipeline.executor import rel
from edgeinfer.reports import REPO_ROOT

# The code a check's result depends on. The pipeline package only orchestrates checks and
# can't change what they measure, so editing it doesn't invalidate earlier results.
CODE_PATHS = ("src/edgeinfer", ":(exclude)src/edgeinfer/pipeline", "uv.lock")


def check_key(
    check: str, artifact: dict[str, str], settings: dict[str, Any], files: list[Path]
) -> dict[str, Any]:
    return {
        "check": check,
        "artifact": {"id": artifact["id"], "sha256": artifact["sha256"]},
        "settings": settings,
        "files": {rel(p): sha256_file(p) for p in sorted(set(files))},
    }


def code_unchanged_since(sha: str) -> bool:
    """True when `CODE_PATHS` are the same in the working tree as at commit `sha`."""
    done = subprocess.run(
        ["git", "diff", "--quiet", sha, "--", *CODE_PATHS],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    # 0: no difference, 1: differences; anything else (e.g. an unknown commit) can't show
    # the code is the same, so the result isn't reused.
    return done.returncode == 0


@dataclass(frozen=True)
class Reuse:
    report: str | None  # path relative to the reports directory, when reusable
    run_id: str | None  # the run whose check is reused
    reason: str  # why it is or isn't reusable


def _runs(reports_dir: Path) -> list[dict[str, Any]]:
    """Every deploy run's state, newest first."""
    runs = []
    for path in (reports_dir / "deploy").glob("*/*/deploy.json"):
        runs.append(json.loads(path.read_text()))
    return sorted(runs, key=lambda r: r["metrics"]["run_id"], reverse=True)


def _report_matches(report: dict[str, Any], key: dict[str, Any]) -> str | None:
    """Why `report` can't stand for a check with `key`, or None if it can."""
    if not report["passed"]:
        return "the check failed"
    if report["git_dirty"]:
        return "its report comes from a dirty tree"
    if report["artifact_id"] != key["artifact"]["id"]:
        return "its report measured another artifact"
    sha = report["inputs"].get("artifact_sha256")
    if sha is not None and sha != key["artifact"]["sha256"]:
        return "its report measured another file with the same id"
    if not code_unchanged_since(report["git_sha"]):
        return f"code changed since {report['git_sha'][:7]}"
    return None


def find_reusable(key: dict[str, Any], reports_dir: Path) -> Reuse:
    reason = "no earlier check with this key"
    for run in _runs(reports_dir):
        for stage in run["metrics"]["stages"].values():
            for check in stage.get("checks", []):
                if check["key"] != key or not check["passed"]:
                    continue
                path = reports_dir / check["report"]
                if not path.is_file():
                    reason = f"its report {check['report']} is missing"
                    continue
                why_not = _report_matches(json.loads(path.read_text()), key)
                if why_not is None:
                    return Reuse(
                        check["report"],
                        run["metrics"]["run_id"],
                        f"same key as run {run['metrics']['run_id']}",
                    )
                reason = why_not
    return Reuse(None, None, reason)
