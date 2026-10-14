# Progress

**Current checkpoint:** C1 — Baseline and eval harness (CPU)
**Next step:** generate manifests from the dataset (`uv run edgeinfer manifests`), then run the FP32 evals. Code for all C1 steps is in place; blocked on the Cityscapes download.

## C1 — Baseline and eval harness
- [x] Repo skeleton: uv (Python 3.12), pytest, ruff, pre-commit, GitHub Actions
- [x] Cityscapes label mapping and manifest code (manifest files: pending data)
- [x] Config-driven preprocessing and postprocessing
- [x] Model registry (SegFormer-B0, DeepLabV3+ MobileNetV2) and `torch_ref` backend
- [x] Confusion-matrix mIoU, eval harness, report envelope, `edgeinfer eval`
- [ ] FP32 eval of both models on all 500 val images (+ full-resolution protocol check)
- [ ] Golden outputs for `golden_val32` with committed SHA-256 hashes (`edgeinfer golden` ready)
- [ ] Decision records D-001..D-009 drafted; Rationale fields need Anuj
- [ ] Teach-back and sign-off

**Sign-off:** _pending_
