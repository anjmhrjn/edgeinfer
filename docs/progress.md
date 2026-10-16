# Progress

**Current checkpoint:** C4 — GPU environment, TensorRT FP32/FP16, performance probe
**Next step:** C4a step 1 — AWS CLI access on the Mac (Anuj); meanwhile the GPU scripts and the performance probe.

## C4 — GPU environment, TensorRT FP32/FP16, performance probe
The GPU quota request (2026-10-14) is still pending, so C4 is split. **C4a** prepares everything that can be written and tested on the Mac; **C4b** runs it on the T4. The roadmap's C4 exit criteria are met in C4b. Plan agreed 2026-10-16: NGC TensorRT container, `cuda-python` for device buffers and timing, an optional `TimedBackend` protocol for per-stage times, SSH limited to the current IP.

C4a (Mac):
- [ ] AWS CLI access; quota status checked; budget alert, security group and key pair (with Anuj's approval)
- [x] Toolchain pinned: NGC `tensorrt:26.09-py3` (TensorRT 11.3, CUDA 13.4.1), re-checked before the first GPU session (D-020); `cuda-bindings` (D-021); `docker/Dockerfile.trt`; torch-free container environment; `supported_ops.yaml` re-seeded from onnx-tensorrt `11.3-GA`
- [ ] `scripts/gpu-{up,down,status}.sh`, `scripts/remote.sh`
- [ ] Performance probe and `edgeinfer perf` (dev runs on `ort_cpu` only)
- [ ] `trt` backend and `edgeinfer build`: builder settings, engine metadata sidecar, mismatch refusal, debug engine
- [ ] Report codes and decision records
- [ ] `scripts/c4_runs.sh` (smoke, ops, build, perf, trtexec, parity, nsys)
- [ ] CI green

C4b (T4): the `c4_runs.sh` phases, Nsight Systems summary, finalized budgets, teach-back, sign-off.

## C3 — INT8 quantization and sensitivity analysis on ORT
Plan agreed 2026-10-15: ORT `quantize_static` (ModelOpt compared in C5), TRT-style Q/DQ placement (inputs of Conv/MatMul/Gemm), medium sensitivity groups from exporter name scopes, sensitivity scored on a seeded 100-image val subset.
- [x] Spike: quantizer settings, bias handling, chunked calibration, ORT int8 kernels vs fake-quant (D-015, D-016)
- [x] Quant config, group rules, exclusion lists, `budgets.yaml` (accuracy), `sens_val100` / `val_minus_sens100` manifests
- [x] `edgeinfer quantize`: chunked calibration cache, content-addressed QDQ ONNX + sidecar, placement check, report
- [x] Validator Q/DQ checks
- [x] INT8 parity tolerances (mIoU gate, other metrics report-only; D-018)
- [x] Calibration methods (minmax / entropy / percentile) on both models; calibration-size study (D-019)
- [x] `edgeinfer sensitivity`: one-at-a-time, leave-one-out, noise floor, greedy exclusion suggestion
- [x] Exclusion list end to end + config-only faults caught on both real models
- [x] CI fixture: quantize → validate → eval → parity → sensitivity, plus config-only faults
- [x] Decision records (D-015..D-019), report codes and CLI exit codes, results
- [x] Teach-back: model answers in `private/teach-back-C3.md`
- [x] CI green on GitHub (run 38843730928, Python 3.11 + 3.12, commit `c4b5dc6`)
- [x] Sign-off (Anuj)

### C3 results — INT8 PTQ on ORT CPU (arm64), Cityscapes val, 512×1024
All runs come from `scripts/c3_runs.sh` on a clean checkout; each report records its git SHA. "Change" is INT8 minus ORT FP32 mIoU on the same images, in points. FP32 references: full val `eval__<model>__ort_cpu__eval_val__512x1024__e264b0b.json`; subsets `eval__<model>__ort_cpu__{sens_val100,val_minus_sens100}__512x1024__cb60f28.json`. These are ORT CPU results; TensorRT INT8 is measured in C5.

**Deployed configuration** (percentile 99.99 calibration on `calib_train200`, plus the exclusions suggested by sensitivity; reports `{quantize,validate}__<model>__int8_percentile__sens__da584fa.json`, `eval__<model>__ort_cpu__int8_percentile__sens__<manifest>__512x1024__da584fa.json`, `parity__<model>__ort_cpu__int8_percentile__sens__golden_val32__images__layers__da584fa.json`):

| Model | Kept in float | Full val mIoU (FP32 → INT8) | Change, full val | Change, held-out 400 | Golden parity (gate ≤ 1.0) | Validator |
|---|---|---|---|---|---|---|
| SegFormer-B0 | nothing (88/88 Conv/MatMul in INT8) | 70.58 → 70.42 | −0.16 | −0.07 | pass (−0.37; argmax agreement 0.992) | 0 errors, 0 warnings |
| DeepLabV3+ MobileNetV2 | `bb.stem_16` (3/60 Convs) | 67.09 → 66.87 | −0.22 | −0.28 | pass (−0.19; argmax agreement 0.989) | 0 errors, 0 warnings |

Both are within the provisional 1.0-point budget on full val and on the 400 held-out images that played no part in choosing the method or the exclusions.

**Calibration method** (D-019, `eval__<model>__ort_cpu__int8_<method>__<manifest>__512x1024__cb60f28.json`): percentile is best for both models; entropy (TensorRT's legacy calibrator) is worst for both and fails the parity gate on both. Full val change, minmax / entropy / percentile: SegFormer −0.53 / −2.29 / −0.16; DeepLab −3.52 / −6.53 / −1.12. Calibration size (25/50/100/200 images) doesn't change SegFormer; it moves DeepLab non-monotonically by up to 0.6 points on `sens_val100`, which mostly didn't hold up on held-out images.

**Sensitivity** (D-017; `sensitivity__segformer_b0__int8_percentile__sens_val100__cb60f28.json`, `sensitivity__deeplabv3plus_mbv2__int8_percentile__sens_val100__da584fa.json`; 100 images, margin 0.7, target ≤ 0.3):
- SegFormer: full INT8 −0.28 → `SENS_FULL_INT8_WITHIN_BUDGET`. No group recovers more than 0.15 points when kept in float (largest: `s2.mlp` 0.15, `s3.attn_matmul` 0.11, `s2.attn_matmul` 0.11).
- DeepLab: full INT8 −1.16. Quantizing only `bb.stem_16` (the first Conv and the first inverted-residual block) costs 0.95; keeping it in float recovers 1.03, and the first greedy step meets the target (−0.13). No other group recovers more than 0.17 (`bb.s24`).

**CNN vs transformer.** The expected contrast (CNN quantizes cleanly, transformer degrades) did not appear under this setup. With the same TensorRT-style Q/DQ format, the transformer loses 0.16 points in full INT8, and the CNN loses 1.12, nearly all of it in its first three Convs.
- Per-layer drift on the golden images (full INT8, `parity__<model>__ort_cpu__int8_percentile__golden_val32__images__layers__cb60f28.json`): SegFormer's worst tensors are in attention (`o_proj`, softmax and the attention MatMuls of stages 0 and 3; cosine ≥ 0.938). DeepLab's worst are the last backbone blocks (`high_level_features.14–17`, groups `bb.s160` and `bb.s320`; cosine down to 0.640).
- Drift doesn't predict accuracy impact: keeping either of those groups in float recovers nothing (`bb.s160` −0.11, `bb.s320` +0.01), and with the stem in float those blocks still have the worst drift (cosine 0.716) while the model is within budget. Per-layer drift locates where numbers diverge; sensitivity locates what costs accuracy.
- Not yet verified, so not a finding: the stem result fits the known difficulty of MobileNetV2's early depthwise layers, whose per-channel activation ranges differ widely under one per-tensor scale. SegFormer's LayerNorm, softmax and GELU stay in FP32 under this placement (D-016), so this comparison doesn't test quantizing them.

**Config-only faults** (`*__fault-*__6ec36eb.json`, `scripts/c3_runs.sh faults`), each caught on both models with a non-zero exit: calibration manifest swapped to `eval_val` → `QUANT_CALIB_ON_EVAL_SPLIT`; an exclusion list naming an unknown group → `QUANT_UNKNOWN_EXCLUSION` (lists the known groups); uint8 activation/weight types → `VALIDATE_QDQ_DTYPE` (names the config fields).

**Found and fixed during the runs:** ORT routes excluded readers of a quantized activation through the shared Q/DQ, so an excluded layer could silently run on quantized values. The placement check stopped the run, and a post-pass now keeps them in float (D-016). A crashing CLI command exited 1, the code for a failed check, so a crash could read as a gate result; crashes now exit 3 (`docs/report-codes.md`).

**Execution noise:** rescoring full INT8 with 1 thread instead of 4 changed SegFormer's subset mIoU by 0.028 points and DeepLab's by 0.000. Group effects below about 0.15 points aren't meaningful (D-017).

**Sign-off:** Anuj, 2026-10-16

## C2 — ONNX export, validator, ORT backend, parity
- [x] Python floor 3.11; `onnx`, `onnxruntime`, `onnxscript` added (D-011)
- [x] Export: dynamo exporter, opset 20, static 1×3×512×1024, content-addressed, I/O contract embedded in ONNX metadata; byte-identical re-exports (D-013)
- [x] `ort_cpu` backend with memory-bounded `layer_outputs`; per-layer reference is ORT FP32 (D-012)
- [x] Validator: 11 finding codes, each unit-tested; `configs/supported_ops.yaml` seeded from onnx-tensorrt (TensorRT 11.4, provisional until C4)
- [x] Parity analyzer: `images` / `golden` / `npz` modes, preprocessing-fault diagnosis, per-layer first-drift, golden hash verification
- [x] FP32 tolerances from measured variation (D-014)
- [x] Deliberate break (RGB→BGR) caught by validator and parity on both models
- [x] CI fixture pipeline (export → validate → ORT eval → golden → parity → breaks) in the CI subset
- [x] Teach-back: model answers in `private/teach-back-C2.md`
- [x] CI green on GitHub (run 38769664648, Python 3.11 + 3.12, commit `30ad2e9`)
- [x] Sign-off (Anuj)

### C2 results — ORT CPU FP32 vs PyTorch FP32
**Export and validation** (`export__<model>__ad46490.json`, `validate__<model>__71edac0.json`): both models export without the fallback and pass the validator with 0 errors and 0 warnings.

| Model | ONNX nodes | Size | Post-export ORT vs torch (max abs) |
|---|---|---|---|
| DeepLabV3+ MobileNetV2 | 117 | 21.1 MB | 2.2e-5 |
| SegFormer-B0 | 400 | 15.6 MB | 1.8e-5 |

**Parity on `golden_val32`** (`parity__<model>__ort_cpu__fp32__golden_val32__{images,golden}__71edac0.json`; identical in both modes because the re-preprocessed input matches golden bit for bit):

| Model | Logits max abs | 1 − cosine (worst) | Argmax agreement (stride 4) | mIoU delta (32 images) |
|---|---|---|---|---|
| DeepLabV3+ MobileNetV2 | 8.2e-5 | 2.9e-12 | 1.0 | −1e-5 points |
| SegFormer-B0 | 8.8e-5 | 1.5e-13 | 1.0 | −2e-5 points |

**Full val, 500 images, 512×1024** (`eval__<model>__ort_cpu__eval_val__512x1024__e264b0b.json` vs the C1 `torch_ref` reports): DeepLabV3+ 67.0863 (delta +2.0e-6 points), SegFormer-B0 70.5770 (delta −6.6e-8 points). Per-class IoUs differ by up to 7e-5 points, which confirms the ORT run computed its own outputs.

**Deliberate break** (config `channel_order: BGR`; `validate__<model>__break-bgr__e264b0b.json`, `parity__<model>__…__layers__break-bgr__e264b0b.json`): the validator fails with `VALIDATE_PREPROC_MISMATCH` on `input.channel_order`. Parity fails (exit 1) with `PARITY_INPUT_MISMATCH` ("channels are reversed … check input.channel_order"), output drift, argmax disagreement and an mIoU drop on the golden images of −11.27 points (DeepLabV3+) and −5.15 (SegFormer). Per-layer mode names the first Conv (#0 in graph order) as the first drifting tensor. With the original config, the same runs pass (reports above).

**Tolerance evidence** (`tolerance_study__<model>__71edac0.json`): ORT with graph optimizations off or one thread stays within 6.5e-5 of golden; per-layer drift between ORT optimization levels ≤ 1.5e-5 relative; near-tie pixels ≤ 4.3e-4 per image.

**Sign-off:** Anuj, 2026-10-15

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
- [x] Sign-off (Anuj)

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

**Sign-off:** Anuj, 2026-10-15
