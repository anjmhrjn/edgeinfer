"""GPU telemetry from nvidia-smi: clocks, temperature, power, memory (D-022).

The T4 is a 70 W card that can lower its clocks when hot or power-limited, so every
GPU measurement records the clocks and temperature around it. nvidia-smi instead of an
NVML binding: no extra dependency, and the queried fields are stable across drivers.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

FIELDS = (
    "name",
    "driver_version",
    "pstate",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "memory.used",
    "memory.total",
)


class TelemetryError(RuntimeError):
    pass


def _value(raw: str) -> Any:
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        return raw  # name, driver version, pstate; "[N/A]" stays visible as text


def gpu_snapshot(index: int = 0) -> dict[str, Any]:
    """One reading of FIELDS for GPU `index` (clocks MHz, temperature C, power W, memory MiB)."""
    if shutil.which("nvidia-smi") is None:
        raise TelemetryError("nvidia-smi not found")
    proc = subprocess.run(
        ["nvidia-smi", f"--id={index}", f"--query-gpu={','.join(FIELDS)}",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )  # fmt: skip
    if proc.returncode != 0:
        raise TelemetryError(f"nvidia-smi failed: {proc.stderr.strip() or proc.stdout.strip()}")
    values = proc.stdout.strip().split(",")
    if len(values) != len(FIELDS):
        raise TelemetryError(f"nvidia-smi returned {len(values)} fields, expected {len(FIELDS)}")
    return {field: _value(v) for field, v in zip(FIELDS, values, strict=True)}
