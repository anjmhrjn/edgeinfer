"""Modules that run in the TensorRT container must not import torch (D-020).

torch brings its own CUDA runtime; loading it next to TensorRT puts two CUDA runtimes
in one process, and the container doesn't install torch at all.
"""

import subprocess
import sys

import pytest

HEAVY = ("torch", "transformers")
TORCH_FREE = [
    "edgeinfer.artifacts",
    "edgeinfer.backends.ort_cpu",
    "edgeinfer.validate.onnx_validator",
    "edgeinfer.analyze.parity",
    "edgeinfer.analyze.sensitivity",
    "edgeinfer.eval.harness",
    "edgeinfer.quant.ptq",
    "edgeinfer.quant.calib",
    "edgeinfer.cli",
]


@pytest.mark.parametrize("module", TORCH_FREE)
def test_module_does_not_import_torch(module):
    # A fresh interpreter: this test process has already imported torch elsewhere.
    code = f"import sys, {module}; print([m for m in {HEAVY!r} if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"{module} imports {out.stdout.strip()}"
