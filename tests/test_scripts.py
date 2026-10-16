"""Static checks on the GPU shell scripts and the container build context (D-023)."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted([*ROOT.glob("scripts/*.sh"), ROOT / "docker" / "entrypoint.sh"])


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_rsync_never_sends_private_files_or_data():
    patterns = {
        line.strip()
        for line in (ROOT / "scripts" / "rsync-excludes.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    # Private notes stay on the Mac; Cityscapes data and derived tensors move only through
    # the explicit push commands (license: no redistribution).
    assert {"/CLAUDE.local.md", "/private/", "/data/raw/", "/artifacts/", "*.npz"} <= patterns


def test_docker_build_context_is_an_allowlist():
    lines = [line for line in (ROOT / ".dockerignore").read_text().splitlines() if line.strip()]
    rules = [line for line in lines if not line.startswith("#")]
    assert rules[0] == "*", "the build context must start from nothing"
    assert all(rule.startswith("!") for rule in rules[1:])
