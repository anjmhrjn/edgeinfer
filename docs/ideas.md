# Ideas (out of scope for now)

Ideas that came up during the work but aren't part of the current roadmap.

- **SegFormer sliding-window eval** (from C1): evaluate SegFormer-B0 at 1024×2048 with 1024×1024 sliding windows, as in the paper, to test whether whole-image inference explains the 1.8-point gap to the published 76.2. Not needed for deployment (we run whole-image at 512×1024).
- **Per-layer parity against PyTorch** (from C2, D-012): map ONNX tensors to PyTorch module outputs through the dynamo exporter's `pkg.torch.onnx.name_scopes` node metadata and capture them with forward hooks. Would localize bugs introduced by the export itself; not needed while whole-model torch↔ORT parity passes.
