"""Inference backends. Tools talk to a Backend, never to a runtime directly."""

from __future__ import annotations

from edgeinfer.backends.base import Backend


def get_backend(name: str) -> Backend:
    if name == "torch_ref":
        from edgeinfer.backends.torch_ref import TorchRefBackend

        return TorchRefBackend()
    raise ValueError(f"unknown backend {name!r} (available in C1: torch_ref)")
