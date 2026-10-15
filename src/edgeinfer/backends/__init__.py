"""Inference backends. Tools talk to a Backend, never to a runtime directly."""

from __future__ import annotations

from edgeinfer.backends.base import Backend


def get_backend(name: str) -> Backend:
    if name == "torch_ref":
        from edgeinfer.backends.torch_ref import TorchRefBackend

        return TorchRefBackend()
    if name == "ort_cpu":
        from edgeinfer.backends.ort_cpu import OrtCpuBackend

        return OrtCpuBackend()
    raise ValueError(f"unknown backend {name!r} (available: torch_ref, ort_cpu)")
