"""Where a stage's command runs: in this process, or on the GPU instance (D-038).

Both run an `edgeinfer` command line and return its exit code; the command's report is
the stage's output. Paths in the command line are relative to the repo root, so the same
command line means the same files on the Mac and in the container on the instance.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from edgeinfer.reports import REPO_ROOT

REMOTE_SH = REPO_ROOT / "scripts" / "remote.sh"


def rel(path: Path) -> str:
    """`path` relative to the repo root when it's inside it, else absolute (tests)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class LocalExecutor:
    """Runs commands in this process through the CLI's own entry point."""

    remote = False

    def __init__(self, root: str | None = None) -> None:
        self._root = root

    def dataset_args(self) -> list[str]:
        return ["--root", self._root] if self._root else []

    def run(self, argv: list[str]) -> int:
        from edgeinfer.cli import main

        print(f"$ edgeinfer {shlex.join(argv)}", flush=True)
        return main(argv)


class RemoteExecutor:
    """Runs commands in the TensorRT container on the instance through scripts/remote.sh,
    which syncs the code first and pulls the reports back after (D-023)."""

    remote = True

    def dataset_args(self) -> list[str]:
        return []  # remote.sh sets CITYSCAPES_ROOT in the container

    def _remote_sh(self, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
        print(f"$ scripts/remote.sh {shlex.join(args)}", flush=True)
        return subprocess.run(
            [str(REMOTE_SH), *args], cwd=REPO_ROOT, text=True, capture_output=capture, check=False
        )

    def _check_paths(self, paths: list[str]) -> None:
        bad = [p for p in paths if Path(p).is_absolute()]
        if bad:
            raise ValueError(f"remote stages need paths inside the repo, got {bad}")

    def run(self, argv: list[str]) -> int:
        self._check_paths([a for a in argv if a.startswith("/")])
        return self._remote_sh("run", f"edgeinfer {shlex.join(argv)}").returncode

    def push(self, paths: list[str]) -> int:
        self._check_paths(paths)
        return self._remote_sh("push-artifacts", *paths).returncode

    def host(self, command: str) -> int:
        return self._remote_sh("host", command).returncode

    def sha256(self, path: str) -> str | None:
        """sha256 of a file on the instance, or None if it isn't there."""
        done = self._remote_sh("sha256", path, capture=True)
        return done.stdout.strip() or None if done.returncode == 0 else None
