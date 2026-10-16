"""Performance probe: latency distribution per stage, throughput, run-to-run variation (D-022).

One iteration is one frame through the whole Python pipeline:

    decoded uint8 frame --preprocess--> NCHW float32 --infer--> logits --postprocess--> class map
    |<------------------------------------ e2e ------------------------------------------->|

Host spans (preprocess, infer, postprocess, e2e) come from a host clock read between the
stages, so e2e is exactly their sum. A TimedBackend also reports its device stages
(h2d, compute, d2h) measured on the device; they must fit inside the host-measured infer
span, and an iteration where they don't means a timer was read before the device
finished (missing synchronization), which the probe reports as an error.

Image decoding isn't timed: frames are decoded into memory before the first iteration,
as a camera would deliver them.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from edgeinfer.backends.base import DEVICE_STAGES, Backend, TimedBackend
from edgeinfer.config import CONFIG_DIR, ConfigError, ModelConfig, _choice, _take
from edgeinfer.perf.stats import summarize
from edgeinfer.perf.telemetry import TelemetryError, gpu_snapshot
from edgeinfer.postprocess import postprocess
from edgeinfer.preprocess import preprocess

HOST_SPANS = ("e2e", "preprocess", "infer", "postprocess")
SPANS = (*HOST_SPANS, *DEVICE_STAGES)
POSTPROCESS_SIZES = ("label", "input", "native")
# Device stages may exceed the host infer span by this much before it counts as an
# inconsistency: CUDA event resolution is about 0.5 us, host clock reads cost about 1 us.
SPAN_SLACK_MS = 0.05
# A p99 needs at least 10 samples above it to be more than a couple of outliers.
MIN_SAMPLES_FOR_P99 = 1000


@dataclass(frozen=True)
class PerfConfig:
    warmup_iterations: int
    iterations: int
    repeats: int
    manifest: str
    max_images: int
    postprocess_size: str
    max_repeat_p99_spread: float

    @classmethod
    def from_dict(cls, d: Any, where: str) -> PerfConfig:
        d = _take(cls, d, where)
        minimum = {"warmup_iterations": 0, "iterations": 1, "repeats": 1, "max_images": 1}
        for key, low in minimum.items():
            if not isinstance(d[key], int) or d[key] < low:
                raise ConfigError(f"{where}.{key}: expected an int >= {low}, got {d[key]!r}")
        _choice(d["postprocess_size"], POSTPROCESS_SIZES, f"{where}.postprocess_size")
        spread = d["max_repeat_p99_spread"]
        if not isinstance(spread, int | float) or spread <= 0:
            raise ConfigError(f"{where}.max_repeat_p99_spread: expected > 0, got {spread!r}")
        return cls(**{**d, "max_repeat_p99_spread": float(spread)})


def load_perf_config(path: str | Path | None = None) -> PerfConfig:
    path = Path(path) if path else CONFIG_DIR / "perf.yaml"
    return PerfConfig.from_dict(yaml.safe_load(path.read_text()), where=path.name)


def latency_budget(budgets: dict[str, Any]) -> dict[str, Any] | None:
    """The optional `latency` section of budgets.yaml: {span, p99_ms_max}."""
    latency = budgets.get("latency")
    if latency is None:
        return None
    _take(_LatencyBudget, latency, "budgets.yaml: latency")
    if latency["span"] not in SPANS:
        raise ConfigError(f"budgets.yaml: latency.span {latency['span']!r} not in {list(SPANS)}")
    if not isinstance(latency["p99_ms_max"], int | float) or latency["p99_ms_max"] <= 0:
        raise ConfigError("budgets.yaml: latency.p99_ms_max must be a positive number")
    return latency


@dataclass(frozen=True)
class _LatencyBudget:
    span: str
    p99_ms_max: float


def is_dev_host(meta: dict[str, Any]) -> bool:
    return str(meta.get("device", "")).startswith("cpu")


def _host_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # macOS: bytes; Linux: KiB


def run_probe(
    cfg: ModelConfig,
    backend: Backend,
    frames: Sequence[np.ndarray],
    pcfg: PerfConfig,
    budgets: dict[str, Any],
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
    preprocess_fn: Callable[[np.ndarray, ModelConfig], np.ndarray] = preprocess,
    postprocess_fn: Callable[[np.ndarray, int, int], np.ndarray] = postprocess,
    telemetry_fn: Callable[[], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]], bool]:
    """Returns (metrics, findings, passed). `frames` are decoded uint8 HxWx3 images."""
    if not frames:
        raise ValueError("no frames to measure")
    meta = backend.metadata()
    findings: list[dict[str, str]] = []
    if meta.get("debug_outputs"):
        # Marked debug outputs disable layer fusion, so its latency says nothing about
        # the deployed engine.
        findings.append({
            "severity": "error", "code": "PERF_DEBUG_ENGINE",
            "message": "refusing to benchmark a debug engine (marked layer outputs disable "
                       "fusion); benchmark the deployed engine instead",
            "where": str(meta.get("artifact", "")),
        })  # fmt: skip
        return {}, findings, False

    budget = latency_budget(budgets)
    timed = isinstance(backend, TimedBackend)
    if budget is not None and budget["span"] in DEVICE_STAGES and not timed:
        raise ConfigError(
            f"latency budget is on span {budget['span']!r}, which {backend.name} doesn't report"
        )
    dev_host = is_dev_host(meta)
    if telemetry_fn is None and not dev_host:
        telemetry_fn = gpu_snapshot

    def target_hw(frame: np.ndarray, logits_hw: tuple[int, int]) -> tuple[int, int]:
        if pcfg.postprocess_size == "label":
            return frame.shape[0], frame.shape[1]
        if pcfg.postprocess_size == "input":
            return cfg.input.height, cfg.input.width
        return logits_hw

    def iteration(i: int) -> tuple[dict[str, float], bool]:
        frame = frames[i % len(frames)]
        t0 = clock()
        x = preprocess_fn(frame, cfg)
        t1 = clock()
        if timed:
            logits, device = backend.infer_timed(x)  # type: ignore[attr-defined]
        else:
            logits, device = backend.infer(x), None
        t2 = clock()
        postprocess_fn(logits, *target_hw(frame, logits.shape[2:]))
        t3 = clock()
        ms = {"preprocess": (t1 - t0) / 1e6, "infer": (t2 - t1) / 1e6,
              "postprocess": (t3 - t2) / 1e6, "e2e": (t3 - t0) / 1e6}  # fmt: skip
        consistent = True
        if device is not None:
            if set(device) != set(DEVICE_STAGES):
                raise ValueError(f"infer_timed returned stages {sorted(device)}, "
                                 f"expected {list(DEVICE_STAGES)}")  # fmt: skip
            ms.update({k: float(device[k]) for k in DEVICE_STAGES})
            consistent = sum(device.values()) <= ms["infer"] + SPAN_SLACK_MS
        return ms, consistent

    def telemetry() -> dict[str, Any] | None:
        if telemetry_fn is None:
            return None
        try:
            return telemetry_fn()
        except TelemetryError as e:
            telemetry_errors.append(str(e))
            return None

    telemetry_errors: list[str] = []
    for i in range(pcfg.warmup_iterations):
        iteration(i)

    pooled: dict[str, list[float]] = {}
    repeats = []
    inconsistent = 0
    total_wall_ns = 0
    for r in range(pcfg.repeats):
        gc.collect()  # don't carry earlier garbage into the measured loop
        before = telemetry()
        e2e: list[float] = []
        start = clock()
        for i in range(pcfg.iterations):
            ms, consistent = iteration(pcfg.warmup_iterations + r * pcfg.iterations + i)
            inconsistent += not consistent
            e2e.append(ms["e2e"])
            for span, value in ms.items():
                pooled.setdefault(span, []).append(value)
        wall_ns = clock() - start
        total_wall_ns += wall_ns
        stats = summarize(e2e)
        repeats.append({
            "e2e_p50_ms": stats["p50"], "e2e_p99_ms": stats["p99"],
            "throughput_fps": pcfg.iterations / (wall_ns / 1e9),
            "telemetry_before": before, "telemetry_after": telemetry(),
        })  # fmt: skip

    spans = {span: summarize(pooled[span]) for span in SPANS if span in pooled}
    p99s = [rep["e2e_p99_ms"] for rep in repeats]
    spread = (max(p99s) - min(p99s)) / float(np.median(p99s))
    n_total = pcfg.repeats * pcfg.iterations
    metrics = {
        "spans_ms": spans,
        "throughput_fps": n_total / (total_wall_ns / 1e9),
        "repeats": repeats,
        "repeat_p99_spread": spread,
        "span_inconsistencies": inconsistent,
        "device_stages_timed": timed,
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
    }

    if dev_host:
        findings.append({
            "severity": "info", "code": "PERF_DEV_HOST",
            "message": f"measured on a development host ({meta.get('device')}); "
                       "not a deployment latency",
        })  # fmt: skip
    if pcfg.warmup_iterations == 0:
        findings.append({
            "severity": "warn", "code": "PERF_NO_WARMUP",
            "message": "no warmup iterations: first-call initialization is in the measurement",
        })  # fmt: skip
    if n_total < MIN_SAMPLES_FOR_P99:
        findings.append({
            "severity": "warn", "code": "PERF_FEW_SAMPLES",
            "message": f"{n_total} samples; p99 needs >= {MIN_SAMPLES_FOR_P99} "
                       "to rest on more than a few outliers",
        })  # fmt: skip
    if spread > pcfg.max_repeat_p99_spread:
        findings.append({
            "severity": "warn", "code": "PERF_UNSTABLE",
            "message": f"p99 e2e varies by {spread:.1%} across repeats "
                       f"(limit {pcfg.max_repeat_p99_spread:.0%}): "
                       f"{', '.join(f'{p:.2f}' for p in p99s)} ms",
        })  # fmt: skip
    if inconsistent:
        findings.append({
            "severity": "error", "code": "PERF_SPAN_INCONSISTENT",
            "message": f"in {inconsistent} of {n_total} iterations the device stages took "
                       "longer than the host-measured infer call; a timer was read before "
                       "the device finished (missing synchronization)",
            "where": "infer_timed",
        })  # fmt: skip
    if telemetry_errors:
        findings.append({
            "severity": "warn", "code": "PERF_TELEMETRY_UNAVAILABLE",
            "message": f"GPU clocks/temperature not recorded: {telemetry_errors[0]}",
        })  # fmt: skip
    if budget is None:
        findings.append({
            "severity": "info", "code": "PERF_BUDGET_UNSET",
            "message": "configs/budgets.yaml has no latency budget yet (set in C4b)",
        })  # fmt: skip
    elif spans[budget["span"]]["p99"] > budget["p99_ms_max"]:
        findings.append({
            "severity": "error", "code": "PERF_BUDGET_EXCEEDED",
            "message": f"p99 {budget['span']} {spans[budget['span']]['p99']:.2f} ms "
                       f"> budget {budget['p99_ms_max']} ms",
            "where": budget["span"],
        })  # fmt: skip
    passed = not any(f["severity"] == "error" for f in findings)
    return metrics, findings, passed
