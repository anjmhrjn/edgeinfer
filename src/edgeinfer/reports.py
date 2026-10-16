"""The JSON report envelope every tool writes, plus schema validation.

Every reported number must trace back to one of these files, so each report
records the git SHA (and whether the tree was dirty) and the environment.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_VERSION = "1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"

_FINDING = {
    "type": "object",
    "required": ["severity", "code", "message"],
    "properties": {
        "severity": {"enum": ["error", "warn", "info"]},
        "code": {"type": "string", "pattern": "^[A-Z][A-Z0-9_]+$"},
        "message": {"type": "string"},
        "where": {"type": "string"},
    },
    "additionalProperties": False,
}

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version", "tool", "created_at", "git_sha", "git_dirty", "env", "model",
        "backend", "precision", "artifact_id", "inputs", "thresholds", "metrics", "passed",
        "findings",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "tool": {"type": "string"},
        "created_at": {"type": "string"},
        "git_sha": {"type": "string"},
        "git_dirty": {"type": "boolean"},
        "env": {
            "type": "object",
            "required": ["host", "device", "versions"],
            "properties": {"versions": {"type": "object"}},
        },
        "model": {"type": "string"},
        "backend": {"type": "string"},
        "precision": {"type": "string"},
        "artifact_id": {"type": "string"},
        "inputs": {"type": "object"},
        "thresholds": {"type": "object"},
        "metrics": {"type": "object"},
        "passed": {"type": "boolean"},
        "findings": {"type": "array", "items": _FINDING},
    },
    "additionalProperties": False,
}  # fmt: skip


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_state() -> tuple[str, bool]:
    """(HEAD SHA, whether tracked files have uncommitted changes)."""
    return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain", "--untracked-files=no"))


def package_versions(*names: str) -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("edgeinfer", "numpy", *names):
        versions[name] = metadata.version(name)
    return versions


def make_report(
    *,
    tool: str,
    model: str,
    backend_meta: dict[str, Any],
    inputs: dict[str, Any],
    metrics: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
    findings: list[dict[str, str]] | None = None,
    passed: bool = True,
) -> dict[str, Any]:
    sha, dirty = git_state()
    versions = {**package_versions(), **backend_meta.get("versions", {})}
    env = {
        "host": socket.gethostname(),
        "device": backend_meta["device"],
        "platform": platform.platform(),
        "threads": backend_meta.get("threads"),
        "versions": versions,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "env": env,
        "model": model,
        "backend": backend_meta["backend"],
        "precision": backend_meta["precision"],
        "artifact_id": backend_meta["artifact_id"],
        "inputs": (
            {**inputs, "artifact_sha256": backend_meta["artifact_sha256"]}
            if "artifact_sha256" in backend_meta
            else inputs
        ),
        "thresholds": thresholds or {},
        "metrics": metrics,
        "passed": passed,
        "findings": findings or [],
    }
    validate_report(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    jsonschema.validate(report, REPORT_SCHEMA)


def write_report(report: dict[str, Any], path: Path) -> Path:
    validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False, allow_nan=False) + "\n")
    return path
