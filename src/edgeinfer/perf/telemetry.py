"""GPU telemetry from nvidia-smi: clocks, temperature, power, memory (D-022).

The T4 is a 70 W card that can lower its clocks when hot or power-limited, so every
GPU measurement records the clocks and temperature around it. nvidia-smi instead of an
NVML binding: no extra dependency, and the queried fields are stable across drivers.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
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


# Sampled during a measured run. clocks_event_reasons is the driver's bitmask of why the
# SM clock is below its maximum (nvidia-smi --help-query-gpu, driver 595).
SAMPLE_FIELDS = ("clocks.sm", "power.draw", "temperature.gpu", "clocks_event_reasons.active")
# Bits that mean the GPU ran slower than it was asked to. Idle (0x1) is expected between
# frames, and the application/locked-clock setting (0x2) is the requested state.
THROTTLE_REASONS = {
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
}


def summarize_samples(lines: list[str], interval_ms: int) -> dict[str, Any]:
    """Min/max over `nvidia-smi --query-gpu=SAMPLE_FIELDS --format=csv,noheader,nounits` lines."""
    rows = [[v.strip() for v in line.split(",")] for line in lines if line.strip()]
    if not rows:
        raise TelemetryError("nvidia-smi produced no samples")
    try:
        sm = [float(r[0]) for r in rows]
        power = [float(r[1]) for r in rows]
        temp = [float(r[2]) for r in rows]
        reasons = 0
        for r in rows:
            reasons |= int(r[3], 16)
    except (ValueError, IndexError) as e:
        bad = next(line for line in lines if line.strip())
        raise TelemetryError(f"unparseable nvidia-smi sample ({e}): {bad!r}") from e
    return {
        "samples": len(rows),
        "interval_ms": interval_ms,
        "clocks.sm_min": min(sm),
        "clocks.sm_max": max(sm),
        "power.draw_max": max(power),
        "temperature.gpu_max": max(temp),
        "clocks_event_reasons": f"{reasons:#x}",
        "throttle_reasons": [name for bit, name in THROTTLE_REASONS.items() if reasons & bit],
    }


class GpuSampler:
    """Samples SAMPLE_FIELDS every `interval_ms` in an nvidia-smi child process for the
    duration of a `with` block. Afterwards `summary` holds the result, or `error` says why
    there is none; sampling problems never interrupt the measurement itself.

    Snapshots before and after a run miss throttling: between frames the GPU idles and is
    back at full clock. The child process writes to a temporary file, not a pipe, so a
    long run can't fill a pipe buffer and stall it.
    """

    def __init__(self, index: int = 0, interval_ms: int = 100) -> None:
        self.index, self.interval_ms = index, interval_ms
        self.summary: dict[str, Any] | None = None
        self.error: str | None = None
        self._proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> GpuSampler:
        if shutil.which("nvidia-smi") is None:
            self.error = "nvidia-smi not found"
            return self
        self._out = tempfile.TemporaryFile(mode="w+")
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--id={self.index}", f"--query-gpu={','.join(SAMPLE_FIELDS)}",
             "--format=csv,noheader,nounits", f"--loop-ms={self.interval_ms}"],
            stdout=self._out, stderr=subprocess.STDOUT, text=True,
        )  # fmt: skip
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is None:
            return
        died = self._proc.poll()
        if died is None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        self._out.seek(0)
        lines = self._out.read().splitlines()
        self._out.close()
        if died is not None:
            self.error = f"nvidia-smi exited early ({died}): {' '.join(lines)[:200]}"
            return
        try:
            self.summary = summarize_samples(lines, self.interval_ms)
        except TelemetryError as e:
            self.error = str(e)


def clock_findings(during: list[dict[str, Any]], lock_mhz: int | None) -> list[dict[str, str]]:
    """Findings on clocks sampled during measured runs, against the configured lock (D-027)."""
    findings = []
    throttled = sorted({r for d in during for r in d["throttle_reasons"]})
    if throttled:
        low = min(d["clocks.sm_min"] for d in during)
        findings.append({
            "severity": "warn", "code": "PERF_GPU_THROTTLED",
            "message": f"the GPU slowed itself during measurement ({', '.join(throttled)}); "
                       f"SM clock went down to {low:.0f} MHz",
        })  # fmt: skip
    if lock_mhz is None:
        findings.append({
            "severity": "warn", "code": "PERF_CLOCK_UNLOCKED",
            "message": "GPU clocks were not locked (configs/perf.yaml gpu_clock_mhz is null); "
                       "latency depends on how busy the GPU was",
        })  # fmt: skip
    elif not during:
        findings.append({
            "severity": "error", "code": "PERF_CLOCK_NOT_LOCKED",
            "message": f"no clock samples, so the configured lock of {lock_mhz} MHz "
                       "can't be confirmed",
        })  # fmt: skip
    else:
        high = max(d["clocks.sm_max"] for d in during)
        if high > lock_mhz:
            findings.append({
                "severity": "error", "code": "PERF_CLOCK_NOT_LOCKED",
                "message": f"SM clock reached {high:.0f} MHz, above the configured lock of "
                           f"{lock_mhz} MHz; lock it on the host (scripts/c4_runs.sh "
                           "clocks-lock)",
            })  # fmt: skip
    return findings
