"""Inference backends. Tools talk to a Backend, never to a runtime directly."""

from __future__ import annotations

from typing import Any

from edgeinfer.backends.base import Backend


def get_backend(name: str, **options: Any) -> Backend:
    """`options` go to the backend's constructor (e.g. ort_cpu: threads)."""
    if name == "torch_ref":
        from edgeinfer.backends.torch_ref import TorchRefBackend

        return TorchRefBackend(**options)
    if name == "ort_cpu":
        from edgeinfer.backends.ort_cpu import OrtCpuBackend

        return OrtCpuBackend(**options)
    if name == "trt":
        from edgeinfer.backends.trt import TrtBackend

        return TrtBackend(**options)
    raise ValueError(f"unknown backend {name!r} (available: torch_ref, ort_cpu, trt)")
