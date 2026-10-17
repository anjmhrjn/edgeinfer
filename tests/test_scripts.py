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


def test_c5_gpu_labels_and_phases(monkeypatch):
    # Report file names carry these labels, and the pool of final-engine candidates is
    # chosen by kind: the placement pair (D-015) and the baselines must never be in it.
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import c5_gpu

    kinds = {label: c5_gpu.kind_of(label) for label in
             ("fp16", "int8_c3_sens", "int8fp16_unfused", "int8fp16_minmax_ort",
              "int8fp16_minmax_modelopt", "sens_full")}  # fmt: skip
    assert {k for k, v in kinds.items() if v in c5_gpu.POOL_KINDS} == {
        "int8_c3_sens", "int8fp16_unfused", "sens_full"}  # fmt: skip
    assert c5_gpu.sens_label([]) == "sens_full"
    assert c5_gpu.sens_label(["s2.mlp", "s0.patch_embed"]) == "sens_s0.patch_embed+s2.mlp"
    doc = c5_gpu.__doc__
    assert all(f"\n  {phase} " in doc for phase in c5_gpu.PHASES)
