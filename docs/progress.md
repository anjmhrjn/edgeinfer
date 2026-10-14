# Progress

**Current checkpoint:** C1 — Baseline and eval harness (CPU)
**Next step:** see the C1 checklist below.

## C1 — Baseline and eval harness
- [x] Repo skeleton: uv (Python 3.12), pytest, ruff, pre-commit, GitHub Actions
- [ ] Cityscapes label mapping and manifests
- [ ] Config-driven preprocessing and postprocessing
- [ ] Model registry (SegFormer-B0, DeepLabV3+ MobileNetV2) and `torch_ref` backend
- [ ] Confusion-matrix mIoU, eval harness, report envelope, `edgeinfer eval`
- [ ] FP32 eval of both models on all 500 val images (+ full-resolution protocol check)
- [ ] Golden outputs for `golden_val32` with committed SHA-256 hashes
- [ ] Decision records complete; teach-back; sign-off

**Sign-off:** _pending_
