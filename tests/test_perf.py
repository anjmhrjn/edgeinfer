"""Performance probe (D-022). A fake clock that the fake stages advance by known amounts
makes every expected latency exact."""

import json
from dataclasses import replace

import numpy as np
import pytest

from edgeinfer.backends.base import TimedBackend
from edgeinfer.cli import main
from edgeinfer.config import ConfigError, load_model_config
from edgeinfer.perf.probe import PerfConfig, load_perf_config, run_probe
from edgeinfer.perf.stats import nearest_rank, summarize
from edgeinfer.perf.telemetry import TelemetryError
from edgeinfer.postprocess import postprocess
from edgeinfer.resize import resize_bilinear
from tests.conftest import TINY_CONFIG

CFG = load_model_config(TINY_CONFIG)  # input 32x64, output stride 4
FRAMES = [np.zeros((40, 80, 3), np.uint8)]
PCFG = PerfConfig(
    warmup_iterations=2, iterations=10, repeats=3, manifest="golden_val32", max_images=1,
    postprocess_size="label", max_repeat_p99_spread=0.10,
)  # fmt: skip
NO_BUDGET: dict = {}


class FakeClock:
    def __init__(self):
        self.ns = 0

    def __call__(self) -> int:
        return self.ns

    def advance(self, ms: float) -> None:
        self.ns += round(ms * 1e6)


class FakeBackend:
    name = "fake"

    def __init__(self, clock, infer_ms=5.0, device="cpu (test)", **meta):
        self.clock, self.infer_ms, self.device, self.meta = clock, infer_ms, device, meta
        self.calls = 0

    def load(self, artifact, cfg):
        pass

    def infer(self, x):
        self.calls += 1
        ms = self.infer_ms(self.calls) if callable(self.infer_ms) else self.infer_ms
        self.clock.advance(ms)
        return np.zeros((1, 19, 8, 16), np.float32)

    def layer_outputs(self, x, names):
        raise NotImplementedError

    def metadata(self):
        return {"backend": self.name, "precision": "fp32", "device": self.device,
                "artifact_id": "test", **self.meta}  # fmt: skip


class FakeTimedBackend(FakeBackend):
    def __init__(self, clock, device_ms, **kw):
        super().__init__(clock, **kw)
        self.device_ms = device_ms

    def infer_timed(self, x):
        return self.infer(x), dict(self.device_ms)


def probe(backend, clock, pcfg=PCFG, budgets=NO_BUDGET, post_sizes=None, **kw):
    def pre(frame, cfg):
        clock.advance(2.0)
        return np.zeros((1, 3, 32, 64), np.float32)

    def post(logits, h, w):
        clock.advance(3.0)
        if post_sizes is not None:
            post_sizes.append((h, w))
        return np.zeros((h, w), np.uint8)

    return run_probe(CFG, backend, FRAMES, pcfg, budgets, clock=clock,
                     preprocess_fn=pre, postprocess_fn=post, **kw)  # fmt: skip


def codes(findings):
    return {f["code"] for f in findings}


# --- statistics ---------------------------------------------------------------------------


def test_nearest_rank_percentiles_are_observed_values():
    values = list(range(1, 101))
    assert [nearest_rank(values, p) for p in (50, 90, 99, 100)] == [50, 90, 99, 100]
    # 10 samples: p99 is the largest, not an interpolation between the top two.
    assert nearest_rank([float(v) for v in range(10)], 99) == 9.0
    with pytest.raises(ValueError):
        nearest_rank([], 50)


def test_summarize():
    s = summarize([4.0, 1.0, 3.0, 2.0])
    assert (s["n"], s["min"], s["p50"], s["max"], s["mean"]) == (4, 1.0, 2.0, 4.0, 2.5)
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))


# --- spans --------------------------------------------------------------------------------


def test_e2e_span_is_exactly_the_sum_of_the_stages():
    clock = FakeClock()
    metrics, _, passed = probe(FakeBackend(clock), clock)
    spans = metrics["spans_ms"]
    for span, ms in {"preprocess": 2.0, "infer": 5.0, "postprocess": 3.0, "e2e": 10.0}.items():
        assert spans[span]["min"] == spans[span]["max"] == pytest.approx(ms), span
    assert spans["e2e"]["n"] == PCFG.repeats * PCFG.iterations
    assert metrics["throughput_fps"] == pytest.approx(100.0)
    assert "compute" not in spans and not metrics["device_stages_timed"]
    assert passed


def test_warmup_iterations_are_run_but_not_recorded():
    clock = FakeClock()
    backend = FakeBackend(clock)
    metrics, _, _ = probe(backend, clock)
    assert backend.calls == PCFG.warmup_iterations + PCFG.repeats * PCFG.iterations
    assert metrics["spans_ms"]["infer"]["n"] == PCFG.repeats * PCFG.iterations


def test_timed_backend_device_stages():
    clock = FakeClock()
    backend = FakeTimedBackend(clock, {"h2d": 1.0, "compute": 3.0, "d2h": 0.5})
    assert isinstance(backend, TimedBackend)
    assert not isinstance(FakeBackend(clock), TimedBackend)
    metrics, findings, passed = probe(backend, clock)
    assert metrics["spans_ms"]["compute"]["p99"] == 3.0
    assert metrics["span_inconsistencies"] == 0
    assert passed and "PERF_SPAN_INCONSISTENT" not in codes(findings)


def test_device_stages_longer_than_the_host_call_mean_missing_sync():
    clock = FakeClock()
    # 6 ms of device work inside a 5 ms host call: a timer was read before the device finished.
    backend = FakeTimedBackend(clock, {"h2d": 1.0, "compute": 4.5, "d2h": 0.5})
    metrics, findings, passed = probe(backend, clock)
    assert metrics["span_inconsistencies"] == PCFG.repeats * PCFG.iterations
    assert "PERF_SPAN_INCONSISTENT" in codes(findings)
    assert not passed


def test_timed_backend_must_report_exactly_the_device_stages():
    clock = FakeClock()
    with pytest.raises(ValueError, match="expected"):
        probe(FakeTimedBackend(clock, {"compute": 3.0}), clock)


def test_ort_cpu_is_not_a_timed_backend():
    from edgeinfer.backends.ort_cpu import OrtCpuBackend

    assert not isinstance(OrtCpuBackend(), TimedBackend)


@pytest.mark.parametrize(
    ("size", "expected"), [("label", (40, 80)), ("input", (32, 64)), ("native", (8, 16))]
)
def test_postprocess_size(size, expected):
    clock = FakeClock()
    sizes: list = []
    probe(FakeBackend(clock), clock, pcfg=replace(PCFG, postprocess_size=size), post_sizes=sizes)
    assert set(sizes) == {expected}


def test_native_postprocess_skips_an_identity_resize():
    logits = np.random.default_rng(0).standard_normal((1, 19, 8, 16)).astype(np.float32)
    # Half-pixel bilinear at scale 1 is the identity, so skipping it changes nothing.
    np.testing.assert_array_equal(resize_bilinear(logits[0], 8, 16), logits[0])
    np.testing.assert_array_equal(postprocess(logits, 8, 16), logits[0].argmax(0))


# --- findings and gates -------------------------------------------------------------------


def test_debug_engine_is_refused_without_measuring():
    clock = FakeClock()
    backend = FakeBackend(clock, debug_outputs=["layer.0"], artifact="x.engine")
    metrics, findings, passed = probe(backend, clock)
    assert backend.calls == 0 and metrics == {}
    assert codes(findings) == {"PERF_DEBUG_ENGINE"} and not passed


def test_dev_host_few_samples_no_warmup_and_unset_budget():
    clock = FakeClock()
    _, findings, passed = probe(FakeBackend(clock), clock, pcfg=replace(PCFG, warmup_iterations=0))
    assert {"PERF_DEV_HOST", "PERF_FEW_SAMPLES", "PERF_NO_WARMUP", "PERF_BUDGET_UNSET"} <= codes(
        findings
    )
    assert passed  # none of these is an error


def test_latency_budget_gate():
    budget = {"latency": {"span": "e2e", "p99_ms_max": 9.0}}
    clock = FakeClock()
    _, findings, passed = probe(FakeBackend(clock), clock, budgets=budget)
    assert "PERF_BUDGET_EXCEEDED" in codes(findings) and not passed
    budget["latency"]["p99_ms_max"] = 10.0
    clock = FakeClock()
    _, findings, passed = probe(FakeBackend(clock), clock, budgets=budget)
    assert not {"PERF_BUDGET_EXCEEDED", "PERF_BUDGET_UNSET"} & codes(findings) and passed


@pytest.mark.parametrize(
    "latency",
    [{"span": "e2ee", "p99_ms_max": 33.3}, {"span": "e2e", "p99_ms_max": 0},
     {"span": "e2e"}, {"span": "compute", "p99_ms_max": 10}],
)  # fmt: skip
def test_bad_latency_budget_is_a_config_error(latency):
    clock = FakeClock()  # "compute" is valid, but this backend doesn't report device stages
    with pytest.raises(ConfigError):
        probe(FakeBackend(clock), clock, budgets={"latency": latency})


def test_unstable_repeats():
    clock = FakeClock()
    per_repeat = PCFG.warmup_iterations + PCFG.iterations
    backend = FakeBackend(clock, infer_ms=lambda call: 5.0 if call <= per_repeat else 15.0)
    metrics, findings, _ = probe(backend, clock)
    assert metrics["repeat_p99_spread"] > PCFG.max_repeat_p99_spread
    assert "PERF_UNSTABLE" in codes(findings)


def test_gpu_telemetry_recorded_or_reported_missing():
    clock = FakeClock()
    snap = {"clocks.sm": 1590.0, "temperature.gpu": 45.0}
    metrics, findings, _ = probe(FakeBackend(clock, device="cuda:0 (Tesla T4)"), clock,
                                 telemetry_fn=lambda: snap)  # fmt: skip
    assert metrics["repeats"][0]["telemetry_before"] == snap
    assert "PERF_DEV_HOST" not in codes(findings)

    def broken():
        raise TelemetryError("nvidia-smi not found")

    clock = FakeClock()
    _, findings, _ = probe(FakeBackend(clock, device="cuda:0 (Tesla T4)"), clock,
                           telemetry_fn=broken)  # fmt: skip
    assert "PERF_TELEMETRY_UNAVAILABLE" in codes(findings)


# --- config and CLI -----------------------------------------------------------------------


def test_shipped_perf_config_loads():
    assert load_perf_config().postprocess_size == "label"


@pytest.mark.parametrize(
    "change", [{"repeats": 0}, {"postprocess_size": "full"}, {"max_repeat_p99_spread": 0},
               {"typo": 1}],
)  # fmt: skip
def test_perf_config_is_strict(change):
    d = {**PCFG.__dict__, **change}
    with pytest.raises(ConfigError):
        PerfConfig.from_dict(d, where="test")


def test_perf_cli_on_tiny_model(tiny_onnx, tmp_path):
    out = tmp_path / "perf.json"
    argv = [
        "perf",
        "--model",
        str(TINY_CONFIG),
        "--backend",
        "ort_cpu",
        "--artifact",
        str(tiny_onnx),
        "--synthetic",
        "64x128",
        "--warmup",
        "2",
        "--iterations",
        "5",
        "--repeats",
        "2",
        "--max-images",
        "2",
        "--out",
        str(out),
    ]
    assert main(argv) == 0
    report = json.loads(out.read_text())
    assert report["tool"] == "perf" and report["passed"]
    assert report["metrics"]["spans_ms"]["e2e"]["n"] == 10
    assert report["inputs"]["frame_hw"] == [64, 128]
    assert {"PERF_DEV_HOST", "PERF_FEW_SAMPLES"} <= codes(report["findings"])


def test_perf_cli_bad_override_is_a_usage_error(tiny_onnx, tmp_path):
    argv = [
        "perf",
        "--model",
        str(TINY_CONFIG),
        "--artifact",
        str(tiny_onnx),
        "--synthetic",
        "64x128",
        "--repeats",
        "0",
        "--out",
        str(tmp_path / "p.json"),
    ]
    assert main(argv) == 2
