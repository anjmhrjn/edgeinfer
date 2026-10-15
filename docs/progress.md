# Progress

**Current checkpoint:** C1 — Baseline and eval harness (CPU)
**Next step:** Anuj signs off C1; then plan C2 in plan mode.

## C1 — Baseline and eval harness
- [x] Repo skeleton: uv (Python 3.12), pytest, ruff, pre-commit, GitHub Actions
- [x] Cityscapes label mapping and manifests (500 val / 200 train calibration / 32 golden)
- [x] Config-driven preprocessing and postprocessing
- [x] Model registry (SegFormer-B0, DeepLabV3+ MobileNetV2) and `torch_ref` backend
- [x] Confusion-matrix mIoU, eval harness, report envelope, `edgeinfer eval`
- [x] FP32 eval of both models on all 500 val images (+ full-resolution protocol check)
- [x] Golden outputs for `golden_val32` with committed SHA-256 hashes; bitwise reproducible across runs
- [x] Decision records D-001..D-009 (D-003..D-009 rationales drafted by Claude; Anuj to review)
- [x] CI green on GitHub (run 38761999524, Python 3.10 + 3.12, commit `0d1f538`)
- [x] Teach-back: model answers in `private/teach-back-C1.md` (live quiz stopped after Q1 at Anuj's request)
- [ ] Sign-off (Anuj)

### C1 results — FP32, PyTorch CPU (`torch_ref`), Cityscapes val (500 images)
All four reports: `artifacts/reports/eval__<model>__torch_ref__eval_val__<H>x<W>__8677951.json` (git `8677951`, clean tree). Scored at full label resolution (D-004).

| Model | Input | mIoU | Published | Published protocol |
|---|---|---|---|---|
| DeepLabV3+ MobileNetV2 | 512×1024 (deployment) | **67.09** | — | — |
| DeepLabV3+ MobileNetV2 | 1024×2048 | 72.07 | 72.1 (VainF README) | whole image, 1024×2048 |
| SegFormer-B0 | 512×1024 (deployment) | **70.58** | 71.9 (paper Table 2, short side 512) | resized input |
| SegFormer-B0 | 1024×2048 | 74.43 | 76.2 (paper Table 2, short side 1024) | 1024×1024 sliding window (paper §4.1) |

**Harness check.** DeepLabV3+ at 1024×2048 gives 0.7206764740; the checkpoint's own `best_score` (VainF's eval) is 0.7206764745. They agree to 9 significant digits, so data loading, label mapping, preprocessing and the confusion matrix reproduce the reference evaluation.

**SegFormer gap (−1.8 at full resolution, −1.3 at 512×1024).** Since the harness reproduces DeepLab exactly, the gap is specific to SegFormer's protocol or weights, not to our evaluation code. Candidate causes, not yet separated:
1. Whole-image inference vs the paper's 1024×1024 sliding window. The model was fine-tuned on 1024×1024 crops, so a full 1024×2048 image is outside its training input size.
2. Resize implementation (ours: float half-pixel bilinear, D-009; mmsegmentation's: OpenCV on uint8).
3. The Hugging Face port of the original mmsegmentation checkpoint.
A sliding-window run would test cause 1 (logged in `docs/ideas.md`). It doesn't change C1: the reference for later checkpoints is our FP32 number at 512×1024, under the same protocol every later backend uses.

**Resolution cost.** Going from 1024×2048 to 512×1024 costs DeepLabV3+ 4.98 points and SegFormer 3.85. For both models the seven largest per-class drops are the thin or small classes: traffic light (−13.0 / −12.7), pole (−10.2 / −11.0), traffic sign, bicycle, person, rider, motorcycle (each −6 to −10). Large flat classes (road, sky, building) lose under 2.5. `wall` improves at the lower resolution for both (+6.1 / +4.5). Figures are IoU points, DeepLab / SegFormer.

**Golden outputs.** 32 images × 2 models; hashes in `data/golden_hashes/`. Regenerating them reproduced every hash bit for bit (CPU, 4 threads). Reports: `artifacts/reports/golden__<model>__8677951.json`.

Dev timings in the reports are from a Mac running other jobs at the same time; they aren't latency results.

**Sign-off:** _pending_
