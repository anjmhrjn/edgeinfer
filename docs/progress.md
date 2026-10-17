# Progress

**Current checkpoint:** C5 — TensorRT INT8 and mixed precision (GPU)
**Next step:** finish the ModelOpt placement run on the Mac, push to GitHub (CI), then GPU session 1 (checklist under C5). The instance is stopped; terminate it if C5's GPU work is more than ~2 weeks away (D-023).

## C5 — TensorRT INT8 and mixed precision
Plan agreed 2026-10-16:
- **INT8+FP16 models.** C3's Q/DQ models are converted to FP16 outside the Q/DQ (ModelOpt `convert_to_f16`), keeping C3's placement and percentile scales. Precision label `int8_fp16`, using the `int8` tolerance set.
- **TensorRT sensitivity.** Leave-one-out on the top 5 ORT groups per model, with every subset of them generated on the Mac in advance. Greedy selection scored for real (D-017), with a noise floor from two builds of full INT8.
- **ModelOpt comparison.** Placement only: ModelOpt `quantize()` placement against ours, both max-calibrated and FP16 elsewhere.
- **Pareto latency.** trtexec compute p50 at the 990 MHz lock for every config; the probe only for finalists and FP16.
- **Final-engine rule, fixed before the results.** The lowest probe `infer` p99 within the accuracy budget on full val, checked held out. Differences under ~3.5% are ties (D-029), and a tie goes to the config with fewer INT8 groups. If no INT8 config beats FP16 beyond that, FP16 is the final engine.
- **Plotting.** matplotlib in a Mac-only `plots` dependency group.
- **GPU time.** Two GPU sessions (accuracy, then latency), ~6 GPU-hours.

Spike (Mac, 2026-10-16):
- [x] **TensorRT 11 INT8.** TensorRT 11.x removes `IInt8Calibrator` and its subclasses and `BuilderFlag.INT8`; explicit Q/DQ is the documented replacement (D-031). The 11.4 `ITensor` API lists no dynamic-range methods. To confirm on 11.3 in the container during smoke.
- [x] **INT8+FP16 route** (D-031). ModelOpt `convert_to_f16` keeps the C3 placement and scales. It drops the embedded metadata, which is restored, as in `edgeinfer fp16`.
- [x] **ORT CPU.** ORT 1.31's CPU graph optimizations break FP16-scale Q/DQ models: SegFormer fails to load (MatMul+Add fused into `QGemm`, which rejects FP16 scales), and DeepLab loads but gives wrong outputs. `ort_cpu` therefore runs `int8_fp16` unoptimized.
- [x] **ModelOpt `quantize()` placement** (8 images, scratch run, placement only; it doesn't depend on the calibration images):
  - DeepLab: Q/DQ also on residual `Add`, `Resize` and `ReduceMean`.
  - SegFormer: also on `LayerNormalization`, `Add`, `Transpose`, `Reshape` and `Resize`.
  - Calibration memory grows with the image count, so the 200-image run needs per-node calibration.

Mac:
- [x] `int8_fp16` artifacts: `edgeinfer quantize --float-dtype fp16`, placement and scale-rounding checks, `VALIDATE_FLOAT_PRECISION`; `int8_fp16` tolerance set = `int8` (D-031)
- [x] Precheck on ORT CPU (`scripts/c5_runs.sh precheck`, reports `*int8_fp16*__439d506.json`): validator clean, golden parity passes, `sens_val100` within 0.055 points of the C3 INT8 models (D-031 Evidence)
- [x] Candidate sets (`analyze/candidates.py`), engine layer summary and per-layer profile attribution (`backends/trt_layers.py`); code and tests, not yet wired into the CLI
- [x] Candidate sets (`edgeinfer candidates`): 32 INT8 + INT8+FP16 configurations per model, every one validated (`candidates__<model>__int8_percentile__top5__95ef750.json`); TensorRT scorer (`edgeinfer sensitivity --candidates`, D-032)
- [x] Build: engine layer summary and Q/DQ fusion check (`BUILD_QDQ_UNFUSED`); trtexec `--profile` (per-layer time by layer kind)
- [x] Pareto front and the final-engine rule, fixed before any INT8 result (`analyze/pareto.py`, D-033); `scripts/pareto.py` (matplotlib, `plots` group)
- [x] `scripts/c5_runs.sh` (Mac and GPU phases), `scripts/c5_gpu.py` (session-1 phases in the container)
- [x] ModelOpt placement artifact (`edgeinfer quantize --tool modelopt`); SegFormer min/max with our placement done (`quantize__segformer_b0__int8_minmax_fp16__ort__3064e28.json`)
- [ ] ModelOpt placement for both models, DeepLab min/max with ours (`scripts/c5_runs.sh placement`): ModelOpt's per-node calibration on 200 images ran > 60 min for SegFormer on the Mac without finishing, so it runs as the last Mac job
- [ ] CI green on GitHub for the C5 Mac commits

GPU session 1 (accuracy, ~2.5 h), in order:
1. `scripts/gpu-up.sh`; `scripts/c5_runs.sh push`; `push-data` isn't needed (val and golden are on the instance from C4)
2. `scripts/c5_runs.sh smoke`: the image rebuilds once (new `uv.lock`). Then: GPU tests including `test_implicit_int8_is_gone` (D-031 on 11.3) and `test_int8_engine_fuses_qdq`, whose printed layer summary is the first real check of the inspector format; `trtexec --help` for the profile flags
3. `scripts/c5_runs.sh gpu build`: check every INT8 build report for `BUILD_QDQ_UNFUSED` before going on
4. `scripts/c5_runs.sh gpu parity`, then `gpu sub` (sens_val100 for FP32, FP16, C3 INT8, INT8+FP16, placement pair)
5. `scripts/c5_runs.sh gpu sens` (both models)
6. Pull reports, log the session in `private/gpu-log.md`, `scripts/gpu-down.sh`

## C4 — GPU environment, TensorRT FP32/FP16, performance probe
The GPU quota request (2026-10-14) is still pending, so C4 is split. **C4a** prepares everything that can be written and tested on the Mac; **C4b** runs it on the T4. The roadmap's C4 exit criteria are met in C4b. Plan agreed 2026-10-16: NGC TensorRT container, `cuda-python` for device buffers and timing, an optional `TimedBackend` protocol for per-stage times, SSH limited to the current IP.

C4a (Mac):
- [x] AWS CLI access (`aws login`), region us-east-1; GPU quota applied (4 vCPUs, request record still `CASE_OPENED`); budget email alerts exist (credit filter skipped for now)
- [x] Toolchain pinned: NGC `tensorrt:26.09-py3` (TensorRT 11.3, CUDA 13.4.1), re-checked before the first GPU session (D-020); `cuda-bindings` (D-021); `docker/Dockerfile.trt`; torch-free container environment; `supported_ops.yaml` re-seeded from onnx-tensorrt `11.3-GA`
- [x] `scripts/gpu-{create,up,status,down,destroy}.sh`, `scripts/remote.sh` (D-023); instance not created yet
- [x] Performance probe and `edgeinfer perf` (D-022): stage spans, nearest-rank percentiles, repeats, missing-sync check, nvidia-smi telemetry; `TimedBackend` protocol; end to end at label resolution
- [x] FP16 ONNX for TensorRT 11 strong typing: `edgeinfer fp16` with ModelOpt AutoCast in `tools/modelopt/` (D-024). Both models: every node FP16, validator clean; ORT CPU full-val pre-check within 0.02 points of FP32 (ORT runs them in mixed precision, so this checks the conversion, not TensorRT FP16)
- [x] `trt` backend and `edgeinfer build` (D-025): builder settings, engine sidecar, mismatch refusal, debug engine, CUDA-event stage timing; TensorRT code untested until the smoke phase
- [x] Report codes and decision records (D-020..D-025)
- [x] `scripts/c4_runs.sh` (push, smoke, ops, build, perf, trtexec, parity, nsys); `scripts/check_supported_ops.py`
- [x] CI green on GitHub (run 38943467616, Python 3.11 + 3.12, commit `865eb70`)

C4b (T4, first GPU session 2026-10-16, 3.94 GPU-hours):
- [x] Instance created (D-023); container re-checked (26.10 still absent); smoke 46/46 on the T4
- [x] Ops check: all 201 listed ops agree with the TensorRT 11.3 parser (`ops_check__trt11.3.0.99__c6ee043.json`)
- [x] FP32 and FP16 engines for both models (`build__<model>__<precision>__d2bf208.json`)
- [x] Golden parity for all four engines; FP16 tolerances (D-030)
- [x] Full-val mIoU on TensorRT
- [x] Perf probe at a locked clock (D-027); trtexec cross-check; iteration counts confirmed (D-028); builder optimization level measured (D-029)
- [x] Budgets finalized: latency on the engine's `infer` span, memory on GPU-wide memory used (D-026)
- [x] Nsight Systems trace of SegFormer FP16, summarized below
- [x] Teach-back: model answers in `private/teach-back-C4.md`
- [x] CI green on GitHub for the C4b commits (run 38974421217, Python 3.11 + 3.12, commit `aa50457`)
- [x] Sign-off (Anuj)

### C4 results — TensorRT 11.3 on the T4 (g4dn.xlarge), Cityscapes, 512×1024, batch 1
Perf: `perf__<model>__trt__<precision>__3a086ea.json` (20 warmup + 3 × 400 frames, SM clock locked at 990 MHz and confirmed in every 100 ms sample, no throttle reason). trtexec: `trtexec__<model>__<precision>__990MHz__3a086ea.json`. Parity: `parity__<model>__trt__<precision>__golden_val32__golden__d188731.json`. Full val: `eval__<model>__trt__<precision>__eval_val__512x1024__<sha>.json`. Engines: `build__<model>__<precision>__d2bf208.json`.

| Engine | `infer` p50 / p99 (budget ≤ 33.3) | compute p50 | trtexec compute p50 | GPU memory used (≤ 953 MiB) | Full val mIoU | Golden parity |
|---|---|---|---|---|---|---|
| SegFormer-B0 FP32 | 23.77 / 25.92 ms | 20.40 ms | 20.26 ms | 427 MiB | 70.577 | pass (max abs 6.1e-5) |
| SegFormer-B0 FP16 | 9.40 / 11.82 ms | 6.07 ms | 5.99 ms | 299 MiB | 70.583 | pass (max abs 0.21, argmax 0.99915) |
| DeepLabV3+ MBv2 FP32 | 19.56 / 21.74 ms | 16.28 ms | 15.91 ms | 239 MiB | 67.086 | pass (max abs 5.4e-5) |
| DeepLabV3+ MBv2 FP16 | 8.41 / 10.36 ms | 5.09 ms | 4.99 ms | 193 MiB | 67.100 | pass (max abs 0.26, argmax 0.99799) |

- **Accuracy.** TensorRT FP32 matches ORT FP32 exactly on full val (70.577, 67.086). FP16 is 0.006 / 0.013 points *above* FP32: argmax near-ties resolving differently, not a gain.
- **FP16 speed.** Against FP32 at the same clock, compute is 3.4× faster for SegFormer and 3.2× for DeepLab, and `infer` p99 2.2× / 2.1×. H2D (6.3 MB input, 1.07 ms) and D2H (2.5 MB logits, 0.39 ms) don't change with precision.
- **Engines.** FP16 halves activation memory (SegFormer 275.7 → 133.8 MiB, DeepLab 87.5 → 43.8 MiB). SegFormer's FP16 engine file is about the same size as FP32 (16.4 vs 16.5 MB), unlike DeepLab's (23.6 → 10.9 MB); why isn't established.
- **trtexec cross-check.** The probe's compute is 0.7–2.3% above trtexec's at the locked clock, as it should be: trtexec uses CUDA graphs and no copies, so it's a lower bound.
- **End to end is the Python pipeline, not the engine.** e2e p99 is 781.6–795.8 ms for every engine: postprocess (upsampling 19 logit channels to 1024×2048, then argmax, single-threaded NumPy) p99 699.7–707.8 ms, preprocess p99 80.4–83.4 ms. That is why the budget gates `infer` (D-026). Moving pre/postprocessing to the GPU is the downstream C++/CUDA runtime's job (D-003).

**Nsight Systems** (`nsys__segformer_b0__fp16__1f0ceb2.json`; 10 warmup + 50 frames captured between `cudaProfilerStart/Stop`, engine loading excluded; the trace is in `artifacts/nsys/`, not committed):
- The GPU is busy 1.0% of a frame: 6.09 ms of kernels and 1.41 ms of copies per frame (H2D 72% of copy time, D2H 28%) against a 754 ms mean frame. The rest is CPU work.
- 88 distinct kernels. The three kernel-launch calls in the report's top five CUDA API entries alone make 10,260 calls over 60 frames, so at least 171 launches per frame. The largest kernels by name: an FP16 tensor-core GEMM with fused ReLU (9.4% of kernel time, once per frame), a 3×3 FP16 convolution (7.9%, 8 per frame), cuTENSOR permutation kernels (7.1% + 3.0%), fused multi-head-attention GEMMs (5.0% + 3.0%) and bilinear resize (4.0%, 3 per frame). Which layers they belong to isn't established yet; that needs the engine's layer information (C5).
- `cudaEventSynchronize` accounts for 330 ms over 60 frames (5.5 ms per frame): the host waiting for the GPU inside `infer`, as expected with one stream and synchronous inference.

**Clock policy** (D-027): with free clocks the GPU sat idle 97–99% of each frame and ran inference at the 1590 MHz boost, while trtexec's continuous load drove it into the 70 W power cap, so trtexec measured *slower* than the probe (`trtexec__*__0bb207f.json`). At the 990 MHz lock, the highest clock all four engines hold without throttling, compute is 20–48% slower than at boost (`perf__*__950b85d.json` vs `perf__*__3a086ea.json`), and the two tools agree.

**Builder optimization level** (D-029): level 5 was 7.5% slower than level 3 for SegFormer FP16 and 0.5% faster for DeepLab FP16, at twice the build time. Two level-3 builds of the same DeepLab engine differ by 3.4%, so TensorRT builds aren't reproducible at the plan level. Reports now record the plan's sha256 next to the engine id.

**Run-to-run variation:** DeepLab FP16 warns `PERF_UNSTABLE`: its per-repeat `infer` p99 is 10.74 / 10.59 / 8.54 ms, while its device stages' p99 are identical across repeats. That is host jitter inside an 8 ms span, a third of the budget (D-028).

**Found and fixed during the runs:**
- `remote.sh` didn't pull a failed check's report (exit 1) until the next run, because `set -e` stopped it first. Reports are now always pulled, and the exit status is passed on.
- Reports written from a working tree with uncommitted edits are marked `git_dirty` and were re-run, never used.
- The before/after nvidia-smi snapshots missed throttling, because the GPU is idle at those moments. Clocks are now sampled throughout each run.

**Sign-off:** Anuj, 2026-10-16

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
