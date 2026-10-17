# Decision records

One entry per non-trivial decision. Newest at the bottom.

<!-- Template:
## D-NNN: <title>  (C<N>, <YYYY-MM-DD>)
**Context:** what problem forced a choice
**Options:**
1. <option> — pros / cons
2. <option> — pros / cons
**Decision:** <chosen>
**Rationale:** why this option over the others
**Evidence:** links to reports or benchmarks, if any
**Consequences / revisit if:** what this commits us to; what would change the decision
-->

## D-001: CNN model — DeepLabV3+ MobileNetV2 (VainF)  (C1, 2026-10-14)
**Context:** The project needs a Cityscapes-trained CNN to contrast with SegFormer-B0 under INT8. It must have public Cityscapes weights, a permissive license and a size that can plausibly meet a real-time budget on a T4.
**Options:**
1. VainF DeepLabV3+ MobileNetV2, OS16 — MIT; published 72.1 mIoU at 1024×2048; lightweight backbone / not pip-installable, weights on Dropbox.
2. VainF DeepLabV3+ ResNet-101 — 76.2 mIoU / much heavier, unlikely to fit the latency budget.
3. mmsegmentation real-time model (BiSeNetV2, STDC) — designed for real time / mmseg is a heavy dependency with fragile installs on current PyTorch.
4. torchvision LRASPP MobileNetV3 — clean dependency / no Cityscapes weights; would need training.
**Decision:** Option 1. Network code vendored (3 files, commit `4e1087d`); checkpoint pinned by SHA-256.
**Rationale:** The purpose of this project is experimenting and comparing results with SegFormer-B0, which is a lightweight transformer model. Using a heavier model doesn't give any additional advantage that a lightweight and budget-friendly model already has.
**Evidence:** Checkpoint `best_score` field = 0.7207 (matches the published 72.1).
**Consequences / revisit if:** Depends on a Dropbox URL staying up (hash-pinned, cached locally). Revisit if the URL dies or the model can't approach the budget.

## D-002: Input resolution — 512×1024  (C1, 2026-10-14)
**Context:** Published Cityscapes numbers use 1024×2048. A real-time budget (p99 ≤ 33.3 ms on a T4, end to end) pushes lower.
**Options:**
1. 512×1024 — deployment-realistic, 4× fewer pixels / lower mIoU than published; not directly comparable to papers.
2. 1024×2048 — comparable to published numbers / likely over the latency budget, especially for the transformer.
**Decision:** Option 1. The FP32 baseline at 512×1024 is the reference for every later comparison. A one-off 1024×2048 run per model checks the harness against published numbers.
**Rationale:** Latency has higher priority than mIoU. Runs that complete within the given budget with a bit of lower mIoU tradeoff is acceptable than runs that complete over the budget.
**Evidence:** C1 eval reports (pending data).
**Consequences / revisit if:** Static 512×1024 engines. Revisit if C4 shows ample latency headroom.

## D-003: Upsampling outside the model graph  (C1, 2026-10-14)
**Context:** Both models predict at 1/4 input resolution. Upsampling to label size can be inside the exported graph or in postprocessing. VainF's upstream `forward()` upsamples inside the graph; SegFormer's doesn't.
**Options:**
1. Outside, in `postprocess.py` — both models export the same kind of output (stride-4 logits); upsampling cost is visible as its own stage; resolution isn't baked into the engine / postprocess must be reimplemented by every runtime.
2. Inside the graph — one fewer stage for runtimes / hides the cost inside "compute"; the engine output becomes 19×1024×2048 floats (~160 MB to copy back per frame).
**Decision:** Option 1. The DeepLabV3+ wrapper skips VainF's final interpolate.
**Rationale:** Keeping upsampling out of the graph makes both models produce the same kind of output, so one postprocess and one parity check cover both. It also keeps the engine output small (19×128×256 instead of 19×1024×2048, which would be ~160 MB to copy back per frame) and makes the upsample cost visible as its own stage instead of hiding it inside compute.
**Evidence:** —
**Consequences / revisit if:** End-to-end timing in C4 must include postprocess. The C++ runtime implements upsample + argmax (a natural custom-kernel candidate).

## D-004: mIoU protocol — score at full label resolution  (C1, 2026-10-14)
**Context:** Predictions come from a 512×1024 input; labels are 1024×2048.
**Options:**
1. Upsample logits (bilinear) to 1024×2048, argmax, compare to untouched labels — the standard protocol; labels never altered / more compute per image.
2. Downsample labels to 512×1024 (nearest) — cheaper / non-standard; thin structures (poles, signs) are lost from the ground truth and scores inflate.
**Decision:** Option 1. One confusion matrix accumulated over all images; IoU per class = TP/(TP+FP+FN); classes absent from both GT and predictions are excluded (NaN), as in cityscapesScripts.
**Rationale:** Scoring at full label resolution is the standard Cityscapes protocol, so our numbers mean the same thing as published ones. Downsampling the labels would erase thin classes like poles and traffic signs from the ground truth and quietly inflate the score.
**Evidence:** `tests/test_miou.py` (hand-computed cases).
**Consequences / revisit if:** —

## D-005: Golden set — 32 val images, input + logits per image  (C1, 2026-10-14)
**Context:** Later backends (ORT, TensorRT, the C++ runtime) are compared against FP32 reference outputs.
**Options:**
1. Per-image uncompressed `.npz` with FP32 preprocessed input and FP32 logits at native stride — runtimes can be checked independently of preprocessing; simple to read from C++ / ~6 MB input + ~2.5 MB logits per image.
2. Logits only — smaller / every parity check then also depends on reproducing preprocessing exactly.
3. Larger set (e.g. 100) — tighter statistics / more disk and per-layer analysis time.
**Decision:** Option 1 with 32 images (`golden_val32`, seeded). SHA-256 hashes committed in `data/golden_hashes/`; the files themselves are gitignored (Cityscapes license).
**Rationale:** Storing the preprocessed input next to the logits lets any runtime be checked on the model alone, without also having to match our preprocessing exactly; preprocessing gets its own check. 32 images are enough to see systematic drift while keeping per-layer analysis fast, and committing only the hashes respects the Cityscapes license.
**Evidence:** —
**Consequences / revisit if:** Preprocessing parity must be tested separately (the input array is the reference for it). Revisit size if parity statistics are too noisy.

## D-006: Python 3.12 for development  (C1, 2026-10-14)
**Context:** The Mac's system Python is 3.14; code must also run inside an NGC container on the GPU instance.
**Options:**
1. Pin 3.12 via `uv python pin` — matches recent NGC containers / not the newest.
2. Use 3.14 — newest / risk of missing wheels and behavior differences from the container.
**Decision:** Option 1. `requires-python >= 3.10` stays; CI tests 3.10 and 3.12. (Floor raised to 3.11 in D-011.)
**Rationale:** 3.12 matches what recent NGC containers ship, so code tested on the Mac behaves the same on the GPU instance. 3.14 is too new to count on wheels for every library we'll need (ONNX, ORT, TensorRT bindings).
**Evidence:** —
**Consequences / revisit if:** Revisit when the C4 container's Python version is known.

## D-007: Own labelId→trainId table instead of cityscapesScripts  (C1, 2026-10-14)
**Context:** Cityscapes labels need mapping from 34 label IDs to 19 train IDs with an ignore index.
**Options:**
1. 34-row table in `data/labels.py`, checked against the official `labels.py` and unit-tested — no dependency.
2. `cityscapesscripts` package — the official source / a dependency (plus its own dependencies) for ~35 lines.
**Decision:** Option 1.
**Rationale:** The mapping is 34 fixed rows that haven't changed since the dataset was released, so a dependency buys nothing; a tested table is easier to read and to explain than a package import.
**Evidence:** `tests/test_labels.py`.
**Consequences / revisit if:** Revisit if we need the official evaluation scripts themselves.

## D-008: Small dependencies — pyyaml, jsonschema, argparse  (C1, 2026-10-14)
**Context:** Configs, report validation and the CLI each need a library choice.
**Options:**
1. `pyyaml` + hand-written strict dataclass loader; `jsonschema` for the report envelope; stdlib `argparse` — few dependencies.
2. `pydantic` (configs + reports) and `typer` (CLI) — less boilerplate / two more dependencies, more magic to explain.
**Decision:** Option 1.
**Rationale:** Each dependency is something to install in the container and explain in an interview. A strict hand-written loader is a few dozen lines and every check is visible; jsonschema is kept because the report envelope is a contract with other tools and a standard schema makes that explicit.
**Evidence:** —
**Consequences / revisit if:** Revisit if the config schema grows enough that hand validation becomes error-prone.

## D-009: Exact resize definition — half-pixel bilinear, no antialiasing, in NumPy  (C1, 2026-10-14)
**Context:** Preprocessing must be reproducible exactly by other runtimes, including a C++/CUDA one. Libraries disagree on "bilinear": PIL antialiases when downscaling, OpenCV uses fixed-point arithmetic on uint8, PyTorch has `align_corners` and `antialias` flags.
**Options:**
1. Own NumPy implementation of half-pixel bilinear (equal to PyTorch `align_corners=False, antialias=False`), applied to float32 — a one-line mathematical definition; tested against PyTorch; for the exact 2× downscale it equals a 2×2 box average / no antialiasing for non-integer downscales.
2. PIL `resize(BILINEAR)` — common / antialiasing filter is hard to reproduce exactly in CUDA.
3. `torch.nn.functional.interpolate` — same math as 1 / makes preprocessing depend on torch.
**Decision:** Option 1 (`src/edgeinfer/resize.py`).
**Rationale:** The C++/CUDA runtime has to reproduce preprocessing bit-for-bit, so the resize must have a definition simple enough to write down in one line. Half-pixel bilinear without antialiasing has that, and the test against PyTorch proves our implementation matches it; PIL's antialiasing filter has no such easy definition.
**Evidence:** `tests/test_preprocess.py::test_resize_matches_torch_half_pixel_bilinear`.
**Consequences / revisit if:** Neither model was trained on resized images, so resolution is a domain shift either way (see D-002).

## D-010: Commit JSON reports to git  (C1, 2026-10-15)
**Context:** The integrity rules require every published number to trace to a JSON report with the git SHA and environment. `artifacts/` was fully gitignored, so a reader of the repo couldn't follow a number back to its report.
**Options:**
1. Commit `artifacts/reports/*.json`; keep golden outputs, ONNX models and engines ignored — numbers traceable from a clone / reports expose the hostname and library versions; repo grows by a few KB per report.
2. Keep reports local; rely on reruns for reproduction — nothing extra in git / numbers in docs can't be checked without the dataset and hours of compute.
**Decision:** Option 1. Reports contain metrics only, no Cityscapes images, labels or tensors, so the license allows it.
**Rationale:** Every number in the docs should be checkable by someone who has only the repo. Reports hold metrics without any Cityscapes data, so committing them is license-safe and costs a few KB; keeping them local would leave the traceability rule unenforceable.
**Evidence:** —
**Consequences / revisit if:** Smoke runs (`--limit`) should write outside `artifacts/reports/` (or be deleted) so only reports that back documented numbers get committed. Reports from before this decision lack the `transformers` version in `env`; it's recoverable from `uv.lock` at the report's git SHA.

## D-011: Raise the Python floor to 3.11  (C2, 2026-10-15)
**Context:** C2 adds onnxruntime. Releases from 1.25 on publish wheels for Python ≥ 3.11 only; 1.23.2 is the newest with 3.10 wheels (checked on PyPI 2026-10-15). CI tested 3.10 and 3.12, and Python 3.10 reaches end-of-life in October 2026.
**Options:**
1. Raise `requires-python` to ≥ 3.11; CI tests 3.11 and 3.12 — one onnxruntime version (1.31) everywhere / drops a Python version nobody on this project uses.
2. Keep 3.10 with environment markers: onnxruntime 1.31 on ≥ 3.11, 1.23.2 on 3.10 — keeps the floor / CI would run two ORT versions whose numerics can differ, so parity tolerances would have to cover both.
3. Pin onnxruntime 1.23.2 everywhere — one version, keeps 3.10 / about a year older, and C3 quantization would build on older ORT tooling.
**Decision:** Option 1.
**Rationale:** The 3.10 floor was a default, not a requirement: the Mac runs 3.12 (D-006) and NGC containers ship 3.12. Keeping it would mean either two onnxruntime versions in CI, which muddies parity, or an old onnxruntime for the whole project, all for a Python version that goes end-of-life this month.
**Evidence:** PyPI metadata: onnxruntime 1.23.2 `requires_python >=3.10`, 1.25.0–1.31.0 `>=3.11`.
**Consequences / revisit if:** Ruff targets py311. Revisit if the C4 NGC container ships Python < 3.11 (unlikely).

## D-012: Per-layer reference is ORT FP32 on the FP32 ONNX model  (C2, 2026-10-15)
**Context:** Per-layer parity compares intermediate tensors of a candidate (ORT INT8 in C3, TensorRT debug engines in C4–C5) against a reference, keyed by ONNX tensor names. PyTorch modules don't have ONNX names.
**Options:**
1. ORT FP32 running the FP32 ONNX file — the same graph, so tensor names match exactly with no mapping / doesn't localize bugs introduced by the export itself.
2. `torch_ref` with forward hooks, mapped to ONNX tensors through the dynamo exporter's `pkg.torch.onnx.name_scopes` node metadata — can localize export bugs / covers module boundaries only (not functional ops inside Hugging Face `forward()`), and the metadata format is exporter-version specific.
**Decision:** Option 1. `TorchRefBackend.layer_outputs` raises an error that names this decision.
**Rationale:** Whole-model parity (torch vs ORT, on every golden image) already checks the export. What the later checkpoints need from per-layer mode is to compare different executions of the same graph, and for that, a reference built from the same graph gets the names right by construction. The torch mapping would be the most fragile code in the analyzer, and it would only be worth it if an export bug showed up that whole-model parity couldn't explain.
**Evidence:** —
**Consequences / revisit if:** The per-layer reference inherits any ORT FP32 deviation from torch, which is bounded by whole-model parity. Revisit if an export bug needs per-layer localization (option 2 is in `docs/ideas.md`).

## D-013: ONNX export — dynamo exporter, opset 20, static shape, single file  (C2, 2026-10-15)
**Context:** Both models must go PyTorch → ONNX for ORT now and TensorRT in C4. Exporter, opset, shape mode and file layout all affect what the TensorRT parser accepts and whether the artifact is reproducible.
**Options:**
1. `torch.onnx.export(dynamo=True)` (the default since torch 2.9) — traces with `torch.export`, emits through onnxscript, runs the onnxscript optimizer (constant folding, BatchNorm folding) / needs the `onnxscript` dependency.
2. Legacy TorchScript exporter (`dynamo=False`) — long track record / many of its options are deprecated in torch 2.14; the fallback if 1 fails.
3. Hugging Face Optimum — knows HF architectures / heavy dependency, only covers SegFormer, and our wrappers already return plain tensors.
Opset: pin 20 (torch 2.14's `ONNX_DEFAULT_OPSET`; ≥ 17 gives native `LayerNormalization`, 20 gives native `Gelu`; the TensorRT 11.4 ONNX parser supports opsets 9–24) vs leave unpinned (would change silently with torch upgrades).
Shapes: static 1×3×512×1024 vs dynamic batch/size.
**Decision:** Option 1, opset 20 pinned, static shape, weights inside the file (`external_data=False`). Input `input`, output `logits`. The exporter's `pkg.torch.onnx.stack_trace` node metadata is stripped.
**Rationale:** The dynamo exporter is PyTorch's supported path and exported both models on the first try, so the fallback wasn't needed. Pinning the opset means the graph changes only when we decide it should, and 20 is inside what both ORT and the TensorRT parser accept. Static shapes match a fixed camera resolution and let TensorRT optimize for exactly one shape. Both files are under 25 MB, so splitting weights out would only add a second file to track. The stack-trace metadata held absolute paths of the local checkout: it leaked the path, and it made the file bytes depend on where the repo lives, which breaks content addressing.
**Evidence:** Export reports `export__<model>__<sha>.json`: ORT vs torch on a seeded random input, max abs error 2.2e-5 (DeepLabV3+, max |logit| 9.7) and 1.8e-5 (SegFormer, 15.0). Re-exporting gives byte-identical files for both models. DeepLabV3+: 117 nodes (BatchNorm folded into Conv, ReLU6 as `Clip`); SegFormer-B0: 400 nodes, including 30 `LayerNormalization` and 8 `Gelu`.
**Consequences / revisit if:** Adds `onnx`, `onnxscript` (alternative: Optimum, rejected above). The I/O contract (input spec, preprocessing, postprocessing) is embedded in the ONNX metadata, so the validator can catch a config that doesn't match the model. Revisit the opset if the C4-pinned TensorRT parser rejects any op at opset 20, and static shapes if a second resolution is ever needed.

## D-014: FP32 parity tolerances from measured variation  (C2, 2026-10-15)
**Context:** Parity needs pass/fail thresholds. Two correct FP32 runtimes still differ: different operation order, fused kernels and thread partitioning change the rounding. A threshold below that noise fails good models, and one far above it hides real regressions.
**Options:**
1. Derive from measurement: 10× headroom over the worst legitimate FP32 variation observed, applied to error magnitude (so 100× on 1−cosine, which grows with the square of the error). Derive argmax and input bounds from what those errors can actually change — tight enough to catch small regressions / measured on one platform (Mac, ORT); TensorRT FP32 may need its own values.
2. Round conventional values (max abs 1e-2, cosine 0.99999, mIoU 0.01) — less likely to need revisiting / 10–100× looser than the observed noise, so a real FP32 regression could pass.
**Decision:** Option 1 (`configs/tolerances.yaml`, `fp32`): input max abs 1e-5; logits max abs 1e-3; cosine ≥ 1−1e-9; argmax agreement ≥ 0.999; mIoU delta ≤ 0.001 points on the golden images; per layer cosine ≥ 1−1e-8 and relative max abs ≤ 2e-4.
**Rationale:** The measured FP32 noise is tiny (logits max abs 8.8e-5, 1−cos 2.9e-12), so tolerances a few orders of magnitude looser would pass FP32 regressions that should fail. Tying each bound to a measurement makes each number defensible: argmax can only flip where the top-2 margin is under twice the logit tolerance, which is at most 4.3e-4 of pixels per image, so 0.999 leaves about 2× room. The input bound sits between float rounding (one float32 ulp ≈ 2.4e-7 at this range) and the smallest real preprocessing fault (one 8-bit intensity step ≈ 0.017 after normalization).
**Evidence:** `parity__<model>__ort_cpu__fp32__golden_val32__golden__<sha>.json` (worst over 32 images: max abs 8.25e-5 DeepLabV3+, 8.77e-5 SegFormer; 1−cos 2.9e-12 / 1.5e-13; stride-4 argmax agreement 1.0; mIoU delta −1e-5 / −2e-5 points). `tolerance_study__<model>__<sha>.json`: ORT with no graph optimizations or one thread stays within 6.5e-5 of golden; per-layer drift between ORT optimization levels ≤ 1.5e-5 relative, 1−cos ≤ 2.4e-11; near-tie pixel fraction ≤ 4.3e-4 per image.
**Consequences / revisit if:** These tolerances apply to any `fp32` candidate. TensorRT FP32 in C4 uses different kernels; if it exceeds these values without a real defect, new values need their own decision record and measurement (never a silent loosening). INT8/FP16 get separate tolerance sets in C3/C4.

## D-015: Quantization tool — ONNX Runtime `quantize_static`; ModelOpt compared in C5  (C3, 2026-10-15)
**Context:** C3 needs a PTQ tool that writes a QDQ ONNX model TensorRT can build in explicit-quantization mode (C5) and that runs on the Mac. The roadmap asked to prefer NVIDIA ModelOpt if its ONNX PTQ runs on macOS, so one QDQ file serves both backends.
**Options:**
1. ONNX Runtime `onnxruntime.quantization.quantize_static` — already a dependency (no new package), runs on the Mac, every setting explicit (format, dtypes, symmetry, per-channel, op types, node include/exclude lists) / defaults are not TensorRT's (asymmetric activations, int32 bias), so each must be set and checked.
2. NVIDIA ModelOpt ONNX PTQ (`nvidia-modelopt[onnx]` 0.47) — TensorRT-oriented Q/DQ placement by default / documented as "Linux (Beta)"; its `[onnx]` extra pins `onnxruntime~=1.24` and `onnx~=1.21` on macOS, below the onnxruntime 1.31 and onnx 1.23 that every C2 report was produced with. Using it would mean a second environment for the quantize step, or downgrading the main one.
**Decision:** Option 1 for C3. ModelOpt runs in C5 on the Linux GPU container, where it's supported, and its QDQ model is compared with ours.
**Rationale:** The QDQ format is the interface TensorRT consumes, not the tool that wrote it, so an ORT-produced model set to TensorRT's conventions builds the same way. Option 2 would trade a supported, already-installed tool for a beta one with conflicting version pins, mainly to get its placement heuristics, and placement is something C5 can compare directly on the GPU where it matters for speed.
**Evidence:** PyPI metadata for `nvidia-modelopt` 0.47.0 (`requires_dist` markers for Darwin); ModelOpt ONNX PTQ guide ("Linux (Beta)"). Checked 2026-10-15.
**Consequences / revisit if:** Q/DQ placement is our choice (D-016), not ModelOpt's heuristic. Revisit if the C5 comparison shows ModelOpt's placement is clearly better on TensorRT in accuracy or latency.

## D-016: QDQ format — signed INT8, symmetric, inputs of Conv/MatMul/Gemm, FP32 bias  (C3, 2026-10-15)
**Context:** The same QDQ model must mean the same thing on ORT now and TensorRT in C5. TensorRT's explicit quantization expects signed INT8 with zero point 0. Its guidance (TensorRT 11.4 docs, "Explicit Quantization") says: quantize all inputs of weighted operations, don't quantize their outputs by default, per-tensor activations, per-channel weights. ORT's `quantize_static` defaults differ in several ways (asymmetric activations, Q/DQ on outputs, Conv bias quantized to int32).
**Options:**
1. TensorRT-style: Q/DQ on the inputs of Conv, MatMul and Gemm only (this includes the activation×activation attention MatMuls), no output Q/DQ, weights per-channel, activations per-tensor, bias left FP32 — follows the documented guidance; LayerNorm, Softmax and GELU stay FP32 / their effect is seen only through the MatMuls that consume their outputs.
2. ORT default op set (Q/DQ around LayerNorm, Resize, Clip, Relu, Concat …) — more of the graph in INT8, LayerNorm sensitivity measurable directly / placement TensorRT may run inefficiently, so ORT and TensorRT results become less comparable.
**Decision:** Option 1. Settings: `QuantFormat.QDQ`, `activation_type=weight_type=QInt8`, `per_channel=True`, `ActivationSymmetric=True`, `WeightSymmetric=True`, `op_types_to_quantize=[Conv, MatMul, Gemm]`, `OpTypesToExcludeOutputQuantization=[Conv, MatMul, Gemm]`, `QuantizeBias=False`.
**Rationale:** The guidance comes from the target runtime, and following it keeps C3's accuracy results meaningful for C5. Bias stays FP32 because neither the TensorRT docs nor NVIDIA's own tools put bias through Q/DQ, and ORT's int32 bias path also adjusts weight scales so the bias fits in int32, which would change the weights' quantization for a reason specific to ORT.
**Evidence:** Spike on the tiny fixture, SegFormer-B0 and DeepLabV3+ (2026-10-15): every Q/DQ is int8 with zero point 0; weights per-channel on axis 0 (Conv) and axis 1 (MatMul, the output-channel axis of a `[K, N]` weight); every Conv/MatMul keeps its node name and output tensor name, so per-layer parity keys still match (D-012); the embedded I/O contract and node metadata survive. `QuantizeBias` is an undocumented extra option (read in `onnxruntime/quantization/qdq_quantizer.py`, ORT 1.31).
**Consequences / revisit if:** TensorRT also recommends quantizing the residual input of skip-connection Adds (for Conv+Add fusion); we don't, which may cost speed in C5 but not accuracy. A Q/DQ pair can feed several consumers (ORT's default; `DedicatedQDQPair` off); check that TensorRT builds this in C5. ORT also routes *excluded* readers of a quantized activation through the shared pair (found in C3 when DeepLab's low-level features, read by both `bb.s32` and `lowlevel_proj`, were quantized for one group only). `DedicatedQDQPair` doesn't help because it only splits pairs between quantized readers, so `quant/ptq.py` points excluded readers back at the float tensor before the placement check; the count is in the sidecar as `placement.unshared_inputs`. MatMul per-channel on axis 1 must be verified against the TensorRT parser in C5. Revisit if C5 shows a placement-driven accuracy or speed problem.

## D-017: Sensitivity analysis — medium layer groups, one-at-a-time and leave-one-out, greedy exclusions scored for real  (C3, 2026-10-15)
**Context:** When full INT8 misses the accuracy budget (mIoU drop ≤ 1.0 point vs FP32, provisional in `configs/budgets.yaml` until C4), we need to know which layers to keep in float, and to show where each architecture loses accuracy. Scoring a configuration means quantizing and evaluating it, so the number of configurations and the images per configuration set the cost.
**Options:**
- Granularity: per node (60 Convs in DeepLab, 88 Conv/MatMul nodes in SegFormer; each node's effect is near the noise) / medium groups from the exporter's module scopes (18 for SegFormer: per stage, patch embedding, attention MatMuls, attention projections, MLP, plus 2 head groups; 10 for DeepLab: per backbone stage, ASPP, low-level projection, head) / per stage (too coarse to name a layer type).
- Modes: one-at-a-time (only group g quantized: its cost in isolation) and leave-one-out (all but g quantized: what keeping g in float recovers). Both, because they disagree when groups interact.
- Exclusion list: rank by leave-one-out recovery and assume the recoveries add up / exclude in that order but quantize and score every cumulative step, stopping at the first within target.
- Scoring set: full val (500 images; ~30 configurations would take days on the Mac) / a seeded 100-image subset of val (`sens_val100`) with its 400-image complement held out.
**Decision:** Medium groups (`configs/quant/groups/<model>.yaml`, regex over the innermost module scope; every quantizable node must match exactly one group), both modes, greedy steps scored for real, scored on `sens_val100`. Target on the subset: budget − margin, with margin 0.7 points. The analyzer writes the suggested list to `configs/quant/exclusions/<model>__sens.yaml`, the same mechanism a user or a fault injection uses.
**Rationale:** Medium groups are coarse enough for effects to stand out and fine enough to name a layer type (attention MatMuls separate from their projections). Scoring each greedy step removes the additivity assumption, and `SENS_NON_ADDITIVE` reports when it would have been wrong. The margin is the largest gap measured between the subset and data it didn't see: 0.70 points (DeepLab, 50-image calibration, D-019). The greedy search picks configurations because they score well on the subset, the same selection effect, so a smaller margin would expect the subset to be more representative than it has proven.
**Evidence:** `sensitivity__segformer_b0__int8_percentile__sens_val100__cb60f28.json`: full INT8 −0.282 on the subset, within the 0.3 target (`SENS_FULL_INT8_WITHIN_BUDGET`); the largest single-group recovery is 0.151 (`s2.mlp`). `sensitivity__deeplabv3plus_mbv2__int8_percentile__sens_val100__da584fa.json`: full INT8 −1.157; quantizing only `bb.stem_16` (the first Conv and the first inverted-residual block, 3 of 60 Convs) costs 0.953, keeping it in float recovers 1.027, and the first greedy step meets the target (−0.131). Execution noise floor (full INT8 rescored with 1 thread instead of 4): 0.028 (SegFormer), 0.000 (DeepLab; identical to every printed digit).
**Consequences / revisit if:** The thread-count noise floor understates the variation between configurations: keeping a group in float made some scores worse by more than the floor (SegFormer `s0.mlp` −0.073 vs a floor of 0.028; DeepLab `bb.s96` −0.125 vs 0.000). Each configuration is quantized from the same calibration cache, so this isn't sampling noise; it's how 100 images' mIoU responds to many small rounding changes. Group effects below about 0.15 points shouldn't be read as real, and the report's `distinguishable` flag (based on the thread-count floor alone) is too permissive. Revisit the margin if the held-out check of an exclusion list misses the budget, and the budget itself in C4.

## D-018: INT8 parity tolerances — gate on accuracy, report drift  (C3, 2026-10-15)
**Context:** Parity needs a tolerance set for INT8 candidates. The FP32 set (D-014) bounds numerical noise between correct runtimes. INT8 differs from FP32 by design: logits move by whole quantization steps. In the spike, even an unchanged INT8 model moved noticeably when only the ORT thread count changed, because an FP32 rounding difference upstream can push a value across a rounding boundary, which then shifts it by a full step.
**Options:**
1. Gate only on accuracy (mIoU delta on the golden images, limit = the 1.0-point budget); report logits max abs, cosine, argmax agreement and per-layer drift without gating them (`null` in `configs/tolerances.yaml`) — the gate measures what the budget is about / a model whose logits drift a lot but whose classes don't change passes, which is the intent.
2. Set elementwise and per-layer limits from the measured drift of an accepted INT8 model — catches a later regression in drift / circular: the limits would encode whatever the current quantizer produced, and would need re-deriving for every calibration method and exclusion list.
**Decision:** Option 1. `int8`: input max abs 1e-5 (preprocessing is unchanged from FP32), mIoU delta ≤ 1.0 point, all other entries `null`. Parity treats `null` as report-only.
**Rationale:** The accuracy budget is the actual requirement; elementwise drift is how INT8 works, not a defect. Gating drift would mean picking numbers after seeing them, which is the circularity the integrity rules forbid. The drift is still in every report, and per-layer mode still ranks where it accumulates, which is what C3 needs it for.
**Evidence:** The INT8 noise floor (same model, different thread count) is measured and reported by the sensitivity analyzer (D-017); results are in `docs/progress.md` C3.
**Consequences / revisit if:** The golden set has only 32 images, so its mIoU delta is a noisy proxy; the budget is decided on full val (`eval` reports). `abs()` also fails an INT8 model that gains more than 1.0 point on the golden images, which deserves investigation rather than a pass. Revisit for FP16 in C4 (FP16 drift is small and can be gated like FP32).

## D-019: Calibration — percentile 99.99 on 200 train images, for both models  (C3, 2026-10-15)
**Context:** Static INT8 needs a range per activation tensor, taken from calibration data. The method that turns observed values into a range decides how much of the tail is clipped versus how coarse the steps are. ORT offers min/max, entropy (KL divergence, the method behind TensorRT's legacy calibrator) and percentile. ORT's entropy defaults (`num_bins=128`, `num_quantized_bins=128`) leave a single candidate threshold, so entropy with defaults is min/max under another name; our configs use 2048/128. The plan chose the method per model on `sens_val100` and checked it on full val and the held-out `val_minus_sens100`.
**Options:**
1. Min/max — no clipping; one outlier stretches the range and coarsens every step.
2. Entropy, 2048 bins — clips to the threshold that keeps the INT8 histogram closest to FP32.
3. Percentile 99.99 — clips the top 0.01% of values.
For the set size: 200 images (the whole `calib_train200`) or a smaller prefix (25/50/100).
**Decision:** Option 3 for both models, calibrated on all 200 images.
**Rationale:** Percentile is best for both models on the scoring subset, on full val and on the held-out 400 images, and it is the only method that passes the golden parity gate for DeepLab. Entropy is worst for both, so TensorRT's calibrator default isn't the safe choice for these models. For the size, a 100-image calibration looked 0.6 points better than 200 on `sens_val100` but only 0.17 better held out, and 50 images went from best on the subset to worst held out. A size picked by its score is partly picked on sampling noise, so we keep the planned 200.
**Evidence:** mIoU change vs ORT FP32 (points); full val / `sens_val100` / `val_minus_sens100` from `eval__<model>__ort_cpu__int8_<method>__<manifest>__512x1024__cb60f28.json` against the FP32 `eval__<model>__ort_cpu__<manifest>__…` reports; golden from `parity__<model>__ort_cpu__int8_<method>__golden_val32__images__layers__cb60f28.json`.

| Model / method | Full val | sens_val100 | Held-out 400 | Golden parity (≤ 1.0) |
|---|---|---|---|---|
| SegFormer-B0 minmax | −0.53 | −0.30 | −0.61 | pass |
| SegFormer-B0 entropy | −2.29 | −1.79 | −2.45 | fail |
| SegFormer-B0 percentile | −0.16 | −0.28 | −0.07 | pass |
| DeepLabV3+ minmax | −3.52 | −3.98 | −3.31 | fail |
| DeepLabV3+ entropy | −6.53 | −6.27 | −6.72 | fail |
| DeepLabV3+ percentile | −1.12 | −1.16 | −1.16 | pass |

Calibration size, percentile (`eval__<model>__ort_cpu__int8_percentile_calib<N>__<manifest>__512x1024__cb60f28.json`; N = 200 is the row above):

| Images | 25 | 50 | 100 | 200 |
|---|---|---|---|---|
| SegFormer-B0, sens_val100 | −0.20 | −0.29 | −0.27 | −0.28 |
| DeepLabV3+, sens_val100 | −1.10 | −0.59 | −0.52 | −1.16 |
| DeepLabV3+, held-out 400 | — | −1.29 | −0.99 | −1.16 |

Across the four DeepLab calibration caches, no single activation tensor's range separates the good and bad sizes by more than 3%; 18 of 109 tensors move by up to 25% between sizes, and not in order of size.
**Consequences / revisit if:** DeepLab's INT8 accuracy depends on which images calibrate it by roughly 0.3 points held out (0.6 on the 100-image subset); single-run differences of that size between calibration settings aren't meaningful. The gap between the scoring subset and held-out data reached 0.70 points (50 images), which set the sensitivity margin (D-017). Revisit in C5: TensorRT builds from these Q/DQ scales, so the method carries over, and ModelOpt's calibrators are compared there.

## D-020: Toolchain — NGC TensorRT container 26.09 (TensorRT 11.3), pinned by digest; torch-free container environment  (C4, 2026-10-16)
**Context:** C4 pins the toolchain to the latest TensorRT that supports the T4 (sm_75), in a matching NGC container. Checked 2026-10-16: TensorRT 11.4.0 is the latest release and supports SM 7.5 and up ([support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)). The newest NGC TensorRT container is `26.09-py3`, which ships TensorRT 11.3 with CUDA 13.4.1, Ubuntu 24.04, Python 3.12 and Nsight Systems 2026.5.1 ([container release notes](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html), [NGC tags](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags)). No container ships 11.4 yet. The newest Deep Learning Base GPU AMI found (20260717) has driver 595.71.05; CUDA 13.x runs on drivers ≥ 580 under minor-version compatibility ([CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)), while CUDA 13.4's own driver branch is R615.
**Options:**
1. NGC container 26.09 (TensorRT 11.3) — TensorRT, CUDA, cuDNN, `trtexec` and Nsight tested together by NVIDIA / one minor version behind the latest TensorRT.
2. 26.09 with TensorRT upgraded to 11.4 inside — latest TensorRT / a combination NVIDIA didn't ship; `trtexec` and samples stay at 11.3 unless rebuilt.
3. CUDA base image + PyPI `tensorrt` wheels — smaller image / CUDA and library versions matched by hand, no `trtexec`.
**Decision:** Option 1, pinned by digest (`sha256:651006ce…db25`) in `docker/Dockerfile.trt`. Re-check the day before the first GPU session: if a newer container with TensorRT 11.4 has shipped, bump to it, since nothing has been built yet. Freeze once the first engines are built. `configs/supported_ops.yaml` is re-seeded from onnx-tensorrt `11.3-GA` (`37fbf78`); its op table is identical to 11.4's.
The container installs no PyTorch. torch and transformers move to a `models` dependency group (installed by default on the Mac and in CI). ONNX-only helpers move from `export/onnx_export.py` to `edgeinfer/artifacts.py`, and `tests/test_torch_free.py` checks that the modules the container runs never import torch.
**Rationale:** The value of the NGC container is that its parts are released and tested as a set; upgrading TensorRT inside it would give that up for one minor version. Since no engine exists yet, waiting for a newer container costs nothing, so the pin stays open until the first build. torch is kept out of the container because importing it loads its own CUDA runtime next to TensorRT's, and nothing on the GPU needs it: export and golden outputs are made on the Mac.
**Evidence:** Sources linked above; image digest from the nvcr.io registry.
**Consequences / revisit if:** The AMI's 595 driver runs CUDA 13.4 through minor-version compatibility, so a feature needing a newer driver would fail with `cudaErrorCallRequiresNewerDriver`. The smoke phase checks this first. The fix would be a newer AMI or the container whose CUDA matches the driver. ONNX models and golden outputs are produced on the Mac and copied to the instance. Engines are tied to this TensorRT version and the T4.

## D-021: CUDA from Python — `cuda-bindings` (cuda-python)  (C4, 2026-10-16)
**Context:** The `trt` backend and the performance probe need device buffers, host↔device copies, a stream and CUDA events for timing. TensorRT's Python API doesn't allocate memory or time anything itself.
**Options:**
1. `cuda-bindings` (NVIDIA cuda-python, low-level module) — official; calls map 1:1 to the CUDA runtime API (`cudaMalloc`, `cudaMemcpyAsync`, `cudaEventRecord`) / more code; a new dependency.
2. `torch.cuda` — already a dependency; convenient tensors and events / a second CUDA runtime in the process; hides allocation and stream details.
3. Polygraphy — least code / hides exactly what is being timed, which the probe has to show.
**Decision:** Option 1, `cuda-bindings~=13.4.4` (matches the container's CUDA 13.4), in a Linux-only `gpu` dependency group. The `cuda-python` metapackage isn't used because it also pulls in `cuda-core`, which we don't need.
**Rationale:** The probe has to make its timing span explicit, and the downstream C++ runtime will make the same CUDA calls, so writing them out in Python keeps the two comparable. torch was ruled out for the same reason as in D-020: a second CUDA runtime next to TensorRT.
**Evidence:** PyPI `cuda-bindings` 13.4.4 publishes manylinux x86_64 wheels.
**Consequences / revisit if:** Nothing on the Mac can import it; GPU code imports it lazily and its tests are marked `gpu`. Bump the pin together with the container's CUDA minor version.

## D-023: GPU instance access and lifecycle — SSH from the current IP only, idle auto-stop, terminate between phases  (C4, 2026-10-16)
**Context:** The T4 instance (`g4dn.xlarge`, us-east-1) is reached from the Mac to sync code, run jobs in the TensorRT container and pull reports back. It must be reachable only by us, must not bill while forgotten, and must fit a ~$100 credit budget. Prices from the AWS Pricing API (2026-10-16): g4dn.xlarge $0.526/hour; gp3 storage $0.08/GB-month, billed whenever the volume exists, including while the instance is stopped. The latest Deep Learning Base GPU AMI (Ubuntu 24.04, build 20261006: driver 595.91.07, G4dn supported) needs a root volume of at least 75 GB.
**Options:**
Access:
1. SSH with a security group that allows only the current public IP — rsync works directly; minimal setup / port 22 is open to that one address while a session runs.
2. SSM Session Manager — no inbound port at all / needs an IAM instance profile and the session-manager plugin; rsync has to tunnel SSH through SSM.
Root volume: 100 GB / 75 GB (the AMI minimum); kept for the whole project / terminated between work phases.
Idle auto-stop on average CPU < 5% for 15, 30 or 60 minutes.
**Decision:** SSH (option 1). `gpu-up.sh` replaces the port-22 rule with this machine's current /32; `gpu-down.sh` removes it. The key is a dedicated ed25519 key generated on the Mac and imported, so the private key never leaves it. The host key is pinned under a fixed alias, because the public DNS name changes on every start. IMDSv2 is required.
Root volume 100 GB gp3, about $8/month while the instance exists. Terminate the instance (`gpu-destroy.sh`) during gaps of more than about two weeks and after C5; `gpu-create.sh` recreates it.
Idle alarm: stop the instance when average CPU stays below 5% for six 5-minute periods (30 minutes), using EC2's free basic monitoring.
Scripts: `scripts/gpu-{create,up,status,down,destroy}.sh` and `scripts/remote.sh`. The instance is found by its `Name` tag; create and destroy only print their plan unless given `--yes`. `remote.sh run` syncs code only (private files, data and artifacts are excluded). Cityscapes val goes to the instance's `/data/cityscapes` only through `remote.sh push-data`, and ONNX models and golden outputs through `push-artifacts`. The container runs as the instance's login user.
**Rationale:** SSH limited to one IP for the length of a session is enough exposure control for a single-user dev box, and it keeps the setup something we can explain end to end. The disk is the main cost that grows with calendar time rather than GPU time: 100 GB for two idle months costs more than all the planned GPU hours, so the instance is deleted when no GPU work is coming. Recreating it costs about 15 minutes of GPU time, and nothing on it is irreplaceable. We chose 100 GB over 75 because running out of disk mid-session wastes paid time, and the AMI's four CUDA toolkits leave unknown free space. Thirty minutes of idle CPU catches a forgotten instance for about $0.30 without stopping it during normal pauses.
**Evidence:** Prices from the AWS Pricing API; AMI facts from the [DLAMI 20261006 release notes](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpubaseoss-ul2404-2026-10-08.html).
**Consequences / revisit if:** The CPU threshold assumes GPU work keeps a CPU core busy, which is true while CUDA spin-waits on synchronization but isn't guaranteed. At the first session, check the instance's CPU metric during an engine build and a benchmark; if it gets close to 5%, lower the threshold or move to a GPU-utilization metric (needs the CloudWatch agent). After termination, engines must be rebuilt, so report comparisons across phases rely on the toolchain pin (D-020). The day-to-day AWS profile with narrow permissions (start/stop/describe the tagged instance, edit its security group) is still to be created; until then the scripts run with `aws login` credentials.

## D-022: Performance probe — stage timing, nearest-rank percentiles, end to end at label resolution  (C4, 2026-10-16)
**Context:** The probe has to produce latency numbers that hold up: a distribution rather than a mean, separate stages, and checks against the usual benchmarking mistakes (missing synchronization, no warmup, too few samples, an unsteady machine). It must work on every backend, including `ort_cpu` on the Mac, while the TensorRT backend can time its device stages with CUDA events. The 33.3 ms budget says "end to end", and what that span includes decides whether it can be met. Dev measurements on the Mac (NumPy, 1024×2048 frame) put preprocessing at about 13 ms and upsampling the logits back to 1024×2048 plus argmax at about 175 ms, against 0.7 ms for argmax alone at the model's output stride. These are rough timings from this discussion, not reported results.
**Options:**
Device stage timing:
1. Add `infer_timed()` to the `Backend` protocol — every backend must implement it, including `torch_ref` and `ort_cpu`, which have no device stages.
2. A separate, optional `TimedBackend` protocol — `Backend` stays the contract downstream consumers use; only backends with device stages implement it.
Postprocess output size for the end-to-end span: label resolution (1024×2048, what eval scores) / model input resolution (512×1024) / native stride (argmax only).
Percentiles: linear interpolation (NumPy's default) / nearest rank.
**Decision:** Option 2 (`TimedBackend.infer_timed()` returns the logits plus milliseconds for `h2d`, `compute` and `d2h`). Label resolution for the end-to-end span. Nearest-rank percentiles.
How each iteration is timed: a host clock (`perf_counter_ns`) is read between preprocess, infer and postprocess, so `e2e` is exactly their sum. Frames are decoded into memory before timing. The device stages must fit inside the host-measured infer call (with 0.05 ms slack); any iteration where they don't is an error (`PERF_SPAN_INCONSISTENT`).
How a run is structured: warmup iterations, then `repeats` independent runs of `iterations` each. The headline statistics pool all repeats; the spread of the repeats' p99 is reported as run-to-run variation. Python's garbage collector stays enabled (a real Python pipeline runs with it) and is run before each repeat. GPU clocks, temperature and power are read with `nvidia-smi` before and after each repeat; no NVML dependency.
The probe refuses debug engines.
The latency gate reads an optional `latency: {span, p99_ms_max}` section of `configs/budgets.yaml`. Until C4b adds it, the probe reports `PERF_BUDGET_UNSET` and passes.
Settings live in `configs/perf.yaml`: warmup 20, 400 iterations × 3 repeats, 8 frames from `golden_val32`, label resolution, PERF_UNSTABLE above a 10% p99 spread.
**Rationale:** The probe is only as credible as its span definitions. Making the device stages fit inside a host span turns the most common GPU benchmarking error into a failed check, and nearest rank never reports a p99 better than a latency that actually happened. Label resolution keeps latency and mIoU describing the same output. If that makes the Python pipeline miss the budget, the numbers will say so and show which stage to move to the GPU, which is the downstream C++/CUDA runtime's job, rather than hiding it by measuring a smaller output.
**Evidence:** `tests/test_perf.py` (exact spans under a fake clock, missing-synchronization detection, gates and findings).
**Consequences / revisit if:** Iteration counts are provisional and are set in C4b from the T4's measured variation. Which span the 33.3 ms budget gates (`e2e` or a device span) is decided in C4b with the measured stage breakdown, as a decision record. `ort_cpu` reports no device stages, so its `infer` span includes everything inside the ORT call. Clock locking isn't decided yet; telemetry is recorded either way.

## D-024: FP16 under TensorRT 11 strong typing — ModelOpt AutoCast in an isolated environment  (C4, 2026-10-16)
**Context:** TensorRT 11 removed `BuilderFlag.FP16` and the other precision flags. Every network is strongly typed: each layer runs in the type the ONNX model gives it, so an FP16 engine needs an FP16 model ([TensorRT 10→11 Python migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-patterns.html)). NVIDIA's documented replacement is ModelOpt AutoCast ([guide](https://nvidia.github.io/Model-Optimizer/guides/8_autocast.html)), which runs on CPU through ONNX Runtime. `nvidia-modelopt[onnx]` 0.47.0 pins `onnx~=1.21` and `onnxruntime~=1.24` on macOS; this project uses onnx 1.23.2 and onnxruntime 1.31.0, the versions behind every C2/C3 result.
**Options:**
1. ModelOpt AutoCast in its own uv project (`tools/modelopt/`, own lockfile), called as a subprocess — NVIDIA's path; the main environment is untouched / a second environment to maintain; heavy (torch).
2. `onnxconverter-common` plus our own overflow check using the ORT layer-output tooling — light, fully our code / reimplements AutoCast.
3. Go back to TensorRT 10.x, where the FP16 flag still exists — simplest build code / reverses D-020 and uses the deprecated path.
**Decision:** Option 1. `edgeinfer fp16` (config `configs/precision/fp16_autocast.yaml`) does four things:
1. Preprocesses the first 8 images of `calib_train200` into NPZ inputs. The train split is enforced with the same check as INT8 calibration.
2. Runs `tools/modelopt/autocast_fp16.py` through `uv run --project tools/modelopt` with `keep_io_types: true` (input and logits stay float32, so the I/O contract and the C++ runtime don't change), `data_max: 512` and `init_max: 65504`.
3. Restores the model's embedded metadata, which AutoCast drops (without it, the I/O contract would be lost).
4. Writes a content-addressed FP16 artifact. Its id hashes the FP32 id, the config, the calibration images and `tools/modelopt/uv.lock`. The sidecar keeps AutoCast's log and a per-node precision summary.
The subprocess's package versions are recorded under a `modelopt_env.` prefix, so a report never confuses them with the main environment's.
**Rationale:** The FP16 conversion is the part of the deployment flow that TensorRT 11 now expects to be done by NVIDIA's tool, so using it keeps the pipeline on the current, supported path. Isolating it costs one extra lockfile and protects the onnx/ORT versions that all earlier results depend on. Writing our own converter would duplicate a maintained tool for no accuracy benefit we can show.
**Evidence:** AutoCast converted all nodes of both models to FP16 (SegFormer 400/400, DeepLab 117/117): no activation exceeded 512 on the calibration images. AutoCast keeps every calibration image's intermediate tensors until it aggregates them, so its memory grows with the image count. With 32 images the SegFormer conversion was killed for lack of memory on the 16 GB Mac; with 8, the peak resident memory of the subprocess was 8.5 GB for SegFormer and 9.6 GB for DeepLab (`autocast_peak_rss_bytes` in `fp16__<model>__fp16_autocast__3b24f8d.json`), so the limit is 8. Both 8-image conversions kept every node in FP16, and both FP16 models pass the validator (`validate__<model>__fp16__3b24f8d.json`). The converted SegFormer runs on ORT CPU on the Mac, which makes a pre-GPU accuracy check possible.
**Consequences / revisit if:** Eight images is a thin sample for a range check, and everything runs in FP16, including SegFormer's LayerNorm and softmax, so FP16 accuracy has to be measured, not assumed. The ORT CPU eval is a pre-check, and TensorRT FP16 parity in C4b is the real one: ORT's CPU backend runs these models in mixed precision, not FP16 (for SegFormer it inserts 117 Cast nodes and runs Conv, MatMul, Add and Mul in FP32; only Gemm, LayerNorm and data movement stay FP16), so the pre-check tests the conversion (FP16-rounded weights, a valid graph) but not TensorRT's FP16 arithmetic. Its full-val results are within 0.02 points of FP32 for both models (`eval__<model>__ort_cpu__fp16__eval_val__512x1024__e3b683e.json`). FP16 parity tolerances don't exist yet; they are set in C4b from TensorRT measurements. For C5: under strong typing, the layers our C3 INT8 Q/DQ models leave unquantized run in FP32, not FP16, and AutoCast doesn't accept quantized models, so an INT8+FP16 mix needs a different route (e.g. ModelOpt's ONNX quantization with an FP16 high-precision type). Revisit the AutoCast thresholds if TensorRT FP16 parity fails.

## D-025: TensorRT engines — content-addressed per TensorRT version and GPU, refused on mismatch, timed with CUDA events  (C4, 2026-10-16)
**Context:** A serialized TensorRT engine is only valid for the TensorRT version and GPU it was built for. The downstream C++ runtime needs engines that describe themselves (I/O names, shapes, dtypes, precision, TensorRT version), and the performance probe needs device-stage timing that can't be read early. TensorRT 11's Python API: builder settings through `IBuilderConfig` (`set_memory_pool_limit`, `builder_optimization_level` 0–5 with default 3, timing cache, `profiling_verbosity`); inference through `set_tensor_address` + `execute_async_v3` on a stream; `mark_output`/`mark_debug` prevent fusion of the tensors they mark ([TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html)).
**Options:**
Mismatch policy:
1. Refuse when TensorRT version, GPU name or compute capability differ; warn on CUDA driver/runtime differences — matches what actually breaks / an engine is rebuilt after any TensorRT upgrade.
2. Try to load and let deserialization fail — less code / a failure inside TensorRT, or an engine that loads but picks different kernels, with no clear message.
Per-layer access:
1. A separate debug engine with marked outputs, never benchmarked — the deployed engine stays fused.
2. Mark debug tensors in the deployed engine — one engine / `mark_debug` prevents fusion, so the measured engine wouldn't be the deployed one.
**Decision:** Mismatch policy option 1, and per-layer option 1.
Engine id: a hash of the ONNX artifact id, the build config (`configs/build.yaml`), the TensorRT version, the GPU name, the compute capability and any debug outputs. The CUDA driver isn't in the id, because a driver update doesn't invalidate an engine.
Each engine has a sidecar holding: the runtime fingerprint; its I/O (checked against the model's contract at build and at load); the preprocessing contract; device memory; build settings; the timing-cache state; and a `.layers.json` from the engine inspector.
Precision comes from the ONNX model (strong typing, D-024).
Inference: pinned host buffers, one stream, and CUDA events around H2D, `execute_async_v3` and D2H. The call returns after the last event completes, so the device stages fit inside the host-measured call (D-022).
Device memory is recorded two ways: the engine's activation size, and the drop in free device memory across loading.
Debug engines get `__debug__` in their file name, and the probe refuses them.
Errors are specific codes (`TRT_*`, `BUILD_*`). A backend that can't load in this environment exits 2.
Builder defaults: workspace 2 GiB, optimization level 3 (TensorRT's default), timing cache on, detailed profiling verbosity.
**Rationale:** Refusing on exactly the fields that make a plan invalid turns a confusing deserialization failure into one actionable message, and the same fields in the id mean a rebuild on a new TensorRT never overwrites an old engine. Pinned buffers and one stream are what a real-time runtime does, so the Python numbers stay comparable with the C++ runtime's. Keeping debug engines separate is the only way per-layer access can't distort the latency of the deployed engine.
**Evidence:** `tests/test_trt_meta.py` (ids, refusal, I/O checks, errors without TensorRT). `tests/test_trt.py` (build, ORT agreement, event timing, refusal, debug outputs) is marked `gpu` and first runs in the C4b smoke phase.
**Consequences / revisit if:** None of the TensorRT/CUDA code has run yet. The smoke phase checks every API call against TensorRT 11.3 before any measurement. The builder optimization level and the effect of the timing cache on reproducibility are decided in C4b from measurements. A timing cache makes rebuilds faster but can make two builds pick different kernels than a cold build would, so its hash before each build is recorded.

## D-026: Latency budget gates the engine's `infer` span; memory budget on GPU-wide memory used  (C4, 2026-10-16)
**Context:** D-022 left open which span the 33.3 ms budget gates. On the T4 (reports `perf__<model>__trt__<precision>__950b85d.json`, unlocked clocks), the Python pipeline's end-to-end p99 is 771.6–794.9 ms for all four engines. Postprocess (bilinear upsampling of 19×128×256 logits to 1024×2048, then argmax, single-threaded NumPy on the g4dn.xlarge's 2 physical cores) takes p99 696.5–705.7 ms, and preprocess takes 79.0–82.4 ms. `infer` (H2D + compute + D2H, host-timed through the TensorRT backend) is 9.1–22.0 ms. No engine, at any precision, could meet 33.3 ms end to end in this pipeline. D-003 and the roadmap already give upsampling, argmax and GPU preprocessing to the downstream C++/CUDA runtime, and they keep CPU preprocessing here so that runtime's gain can be measured. The "≤ 1 GB device memory" budget also needs a metric.
**Options:**
Latency:
1. Gate the engine's `infer` span at p99 ≤ 33.3 ms. e2e is reported as the Python baseline. No split of the frame between engine and pre/post / weak until C5: every engine passes today.
2. Gate `infer` at part of the frame, keeping the rest for GPU pre/postprocessing. A realistic gate / the split would be a guess until the downstream runtime measures GPU pre/post.
3. Move pre/postprocessing to the GPU here (e.g. small TensorRT pre/post engines) and gate a real e2e. The most realistic / a large scope increase that overlaps the downstream project and reverses the roadmap default.
4. Keep e2e at 33.3 ms. Literal / every engine fails by about 23×, so the gate can't tell precisions apart.
Memory metric:
1. GPU-wide `memory.used` from nvidia-smi with the engine loaded. Includes the CUDA context and anything else on the GPU, so it can't undercount the process / counts whatever else happens to be on the GPU (nothing, on this instance).
2. `cudaMemGetInfo` difference around engine load (`device_used_delta_bytes`). The engine's own footprint / leaves out the CUDA context, which a deployed process also pays for.
3. TensorRT's activation memory plus engine size. Static, no measurement / misses runtime allocations.
**Decision:** Latency option 1 (Anuj). `configs/budgets.yaml`: `latency: {span: infer, p99_ms_max: 33.3}`. Memory option 1 (Claude's choice when finalizing the budgets): `memory: {device_used_mib_max: 953}` (1 GB = 10⁹ bytes = 953.7 MiB, rounded down so the limit is the stricter reading of "1 GB"), the maximum of the snapshots taken before and after each repeat. Neither budget is gated on a development host (`PERF_DEV_HOST`). The run-to-run stability check (`PERF_UNSTABLE`) now uses the p99 of the budgeted span instead of e2e, and each repeat records the p99 of every span.
**Rationale:** The engine is what this project optimizes and what the precision choices change. `infer` is the part of the frame the Python pipeline and the C++ runtime share: the same pinned copies and the same `execute_async_v3` on one stream. Gating e2e would fail every engine because of CPU stages no engine choice can affect. Splitting the frame would put an invented number in a gate. The full 33.3 ms is the only value measured evidence supports today, and it's loose on purpose until the downstream runtime measures its GPU pre/postprocessing. For memory, the conservative metric can only overstate the footprint, so passing it means the process fits.
**Evidence:** `perf__*__trt__*__950b85d.json` (stage breakdown above; GPU memory used 193–427 MiB with an engine loaded). `tests/test_perf.py` (`test_stability_is_checked_on_the_budgeted_span`, `test_memory_budget_gates_gpu_memory_used`, `test_dev_host_latency_is_not_gated`).
**Consequences / revisit if:** The budget leaves no room for pre/postprocessing. When the downstream runtime measures them on the GPU, tighten `p99_ms_max` to the frame minus their p99, with a new record. e2e is still reported in every perf report and in the results, labelled as the Python-pipeline baseline. Earlier reports' `repeat_p99_spread` is on e2e; reports from this change on record `repeat_p99_spread_span`.

## D-027: Benchmark at a locked SM clock of 990 MHz, sampled during every run  (C4, 2026-10-16)
**Context:** The T4 is a 70 W card. Under continuous load it lowers its own SM clock to stay under the power limit (nvidia-smi throttle reason `0x4`, software power cap). In the first T4 perf runs (`perf__*__950b85d.json`) the GPU sat idle for about 97% of each frame, waiting on the CPU stages, so every inference ran at the full 1590 MHz. trtexec's back-to-back load instead held the clock near the cap. As a result trtexec's compute time came out *above* the probe's (probe/trtexec 0.72–0.90 in `trtexec__*__0bb207f.json`), the reverse of what a lower bound should show. The probe's before/after telemetry snapshots couldn't show this, because they're taken while the GPU is idle. A real 30 FPS deployment would sit somewhere between these two duty cycles. So unless the clock is fixed, the same engine's latency depends on how busy the rest of the pipeline keeps the GPU.
**Options:**
1. Lock the SM clock to the highest supported value that every engine sustains under continuous load without a throttle reason, for both perf and trtexec. Numbers are consistent and comparable with trtexec, and conservative for sustained load / numbers below the card's peak; a host-side sudo step.
2. Leave clocks free and sample them during each run. Realistic for this pipeline's duty cycle / numbers shift whenever the pipeline's speed changes, and trtexec stays incomparable.
3. Both, free and locked. The most information / double the perf GPU time.
**Decision:** Option 1 (Anuj). `configs/perf.yaml` `gpu_clock_mhz: 990`. `scripts/c4_runs.sh` locks it on the host (`nvidia-smi --lock-gpu-clocks=990,990`) before the perf, trtexec and nsys phases. Persistence mode is on, so the lock holds between processes until `clocks-reset` or a reboot (`gpu-down.sh` stops the instance, which resets it). The probe and the trtexec wrapper sample SM clock, power, temperature and throttle reasons every 100 ms during each measured run (`GpuSampler`), and report:
- `PERF_CLOCK_NOT_LOCKED` (error) if the clock goes above the lock, or if there are no samples to confirm it;
- `PERF_GPU_THROTTLED` (warn) if a throttle reason appears;
- `PERF_CLOCK_UNLOCKED` (warn) if no lock is configured.
How 990 was chosen: the T4 supports SM clocks from 720 to 1590 MHz in 15 MHz steps (`nvidia-smi -q -d SUPPORTED_CLOCKS`). A diagnostic search (20 s of continuous trtexec per engine, not kept as a report) put the limit between 990 and 1035. The reports below confirm it.
**Rationale:** A gate has to give the same answer for the same engine, and with free clocks it wouldn't: the same engine measures faster or slower depending on how busy the rest of the pipeline keeps the GPU. Locking at a clock the card can always hold makes the number a property of the engine. It's also the conservative reading for sustained 30 FPS operation, and it makes the probe and trtexec directly comparable, so the trtexec cross-check works as intended again.
**Evidence:** `trtexec__deeplabv3plus_mbv2__fp32__1035MHz__3a086ea.json`: at 1035 MHz the power cap engaged (down to 1020 MHz, 70.2 W). `trtexec__*__990MHz__3a086ea.json`: all four engines at 990 MHz in every sample, no throttle reason, highest power 68.2 W (DeepLab FP32). Probe/trtexec compute is now 1.007–1.023, so the probe sits just above the trtexec lower bound, as it should. `perf__*__3a086ea.json`: 990 MHz in every sample of every repeat, no throttle reason.
**Consequences / revisit if:** Latencies are measured below the card's boost clock (1590 MHz), so compute times are 20–48% higher than at boost (compute p50 locked vs boost: SegFormer FP32 20.4 vs 16.5 ms, DeepLabV3+ FP32 16.3 vs 11.0 ms). The power margin at 990 is under 2 W. Any `PERF_GPU_THROTTLED` at the lock means stepping down to 975 MHz with an update to this record. The `perf__*__950b85d.json` reports were measured at boost and are superseded for gating.

## D-028: Probe iteration counts — 20 warmup + 3 × 400, confirmed on the T4  (C4, 2026-10-16)
**Context:** D-022 set 20 warmup iterations and 3 repeats of 400 as provisional values. On the T4 each frame takes about 0.75 s, almost all of it CPU pre/postprocessing, so a four-engine perf sweep takes about an hour (`perf__*__3a086ea.json`: 20:21–21:24 UTC).
**Options:**
1. Keep 20 + 3 × 400. Pooled p99 rests on 12 samples above it / about 1 GPU-hour per sweep.
2. 20 + 3 × 200. Half the cost / p99 rests on 6 samples, and it trips `PERF_FEW_SAMPLES` (threshold 1000), which would need its own record.
3. Keep the counts but time postprocessing at native stride, so each iteration is about 10× cheaper / reverses D-022's label-resolution choice.
**Decision:** Option 1 (Anuj).
**Rationale:** Measured run-to-run variation is small with these counts: the repeats' e2e p99 spread was 1.1–3.1% at boost (`perf__*__950b85d.json`) and 0.2–3.6% for the budgeted `infer` span at the lock, except DeepLab FP16 (see below). Fewer samples would make p99 depend on even fewer outliers. An hour per sweep fits the GPU budget for the sweeps C5 needs.
**Evidence:** `perf__*__950b85d.json`, `perf__*__3a086ea.json` (`repeat_p99_spread`, per-repeat `p99_ms`).
**Consequences / revisit if:** DeepLab FP16 warns `PERF_UNSTABLE`: `infer` p99 per repeat was 10.74 / 10.59 / 8.54 ms, while its device stages' p99 were identical across repeats (compute 5.10 ms each). The variation is host-side jitter inside an 8 ms span (Python, the copy into pinned memory, the synchronize wake-up on a 4-vCPU VM), where each repeat's p99 is its 4th-largest sample. The pooled p99 (10.36 ms) is a third of the budget. If a short engine's p99 ever approaches its budget, more iterations per repeat is the fix, not a looser spread limit.

## D-029: Builder optimization level 3 (TensorRT's default), measured against 5  (C4, 2026-10-16)
**Context:** TensorRT's builder optimization level (0–5) sets how much kernel search the builder does: higher levels try more tactics and build more slowly. D-025 left the level at the default (3) until it could be measured.
**Options:**
1. Level 3 — TensorRT's default; faster builds.
2. Level 5 — the most search / slower builds; a gain only if measured.
**Decision:** Option 1, from measurement (Anuj chose to measure). FP16 engines for both models were built at level 5 (`configs/build_opt5.yaml`, into `artifacts/engines_opt5/`). To separate the level from the build conditions, level 3 was rebuilt under the same conditions: at the 990 MHz lock with a fresh timing cache, into `artifacts/engines_opt3_lock/`. trtexec compute p50 at 990 MHz:
- SegFormer: level 3 5.97 ms, level 5 6.42 ms (7.5% slower); build time 90 s vs 187 s.
- DeepLab: level 3 4.82 ms, level 5 4.80 ms (0.5% faster); build time 33 s vs 61 s.
**Rationale:** Level 5 made SegFormer slower and changed DeepLab by less than the variation between two level-3 builds. The deployed level-3 DeepLab engine (built unlocked) measures 4.99 ms against 4.82 ms for its locked rebuild, a 3.4% difference with identical settings. Doubling build time buys nothing here. That level 5 can be slower at all shows its search isn't guaranteed to find a faster plan: the tactic timings it measures during the build are noisy too.
**Evidence:** `build__<model>__fp16__opt5__9c80a8e.json`, `trtexec__<model>__fp16__opt5__990MHz__9c80a8e.json`, `build__<model>__fp16__opt3lock__69a6f9c.json`, `trtexec__<model>__fp16__opt3lock__990MHz__69a6f9c.json`, `trtexec__<model>__fp16__990MHz__3a086ea.json`.
**Consequences / revisit if:** Builds aren't reproducible at the plan level: two builds with the same inputs, and therefore the same engine id, chose different kernels. The SegFormer rebuild has 189 layers instead of 190, and the DeepLab rebuild's compute differs by 3.4%. Reports now record the plan's sha256 (`inputs.artifact_sha256`) next to the engine id, so a measurement names the exact plan. Latency comparisons between precisions in C5 should allow for this few-percent build-to-build variation. Revisit for INT8 engines, where the larger tactic space might change the result.

## D-030: FP16 parity tolerances — 10× headroom over measured TensorRT FP16 drift  (C4, 2026-10-16)
**Context:** FP16 changes the logits by design, but by far less than INT8, and the 1-point accuracy budget is far above anything a correct FP16 engine should cost. TensorRT FP16 against the FP32 golden outputs on the 32 golden images (`parity__<model>__trt__fp16__golden_val32__golden__tolfp32__3a086ea.json`, run with the FP32 tolerance set, which it fails as expected): logits max abs 0.212 / 0.260, cosine min 0.99999909 / 0.99996837, argmax agreement min 0.99915 / 0.99799, golden mIoU Δ +0.0002 / +0.0027 points (SegFormer / DeepLabV3+). Full val: +0.006 / +0.013 points (`eval__<model>__trt__fp16__eval_val__*`).
**Options:**
1. Same method as D-014: 10× the worst observed error magnitude (100× on 1−cos). Tight enough to catch an FP16 fault far below the budget, and every bound traces to a report / measured on one TensorRT version.
2. Accuracy only, as for INT8 (D-018): golden mIoU |Δ| ≤ 1.0 point. Consistent with INT8 / an FP16 regression of up to a full point would pass.
3. Round sanity bounds (cosine ≥ 0.999, argmax ≥ 0.99, mIoU ≤ 0.1 points). Less tied to one version / judgement calls rather than measurements.
**Decision:** Option 1 (Anuj). `configs/tolerances.yaml` `fp16`: input max abs 1e-5 (preprocessing unchanged), logits max abs ≤ 2.6, cosine ≥ 0.99684, argmax agreement ≥ 0.98, golden mIoU |Δ| ≤ 0.026 points. Each bound is rounded toward the stricter side. Per-layer bounds are null (reported, not gated) until FP16 per-layer drift is measured on a debug engine.
**Rationale:** FP16 should be close to lossless, so its gate should fail an FP16-specific fault (an overflow, a layer that loses range) long before the accuracy budget would notice. Deriving the bounds the same way as FP32's makes them defensible in the same terms and keeps them tied to measurements rather than round numbers.
**Evidence:** The parity reports above, and `parity__<model>__trt__fp16__golden_val32__golden__<sha>.json` once the gate runs.
**Consequences / revisit if:** The bounds come from one TensorRT version and one build of each engine. Re-measure them with a new record if a TensorRT upgrade or a rebuild changes FP16 kernel choices enough to fail them without a real defect. Never loosen them silently.

## D-031: INT8 with FP16 outside the Q/DQ — ModelOpt `convert_to_f16` on our Q/DQ models  (C5, 2026-10-16)
**Context:** TensorRT 11 networks are strongly typed (D-024), and TensorRT 11.x removes the implicit-quantization path: `IInt8Calibrator` and its subclasses and `BuilderFlag.INT8` are gone, and explicit Q/DQ (ModelOpt or manual) is the documented replacement ([10→11 migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-patterns.html)). An INT8 engine is therefore built from a Q/DQ model, and every layer outside the Q/DQ pairs runs in the model's own float type. C3's Q/DQ models are FP32 outside the pairs, so their LayerNorm, softmax, Resize, residual Adds and any excluded group would run in FP32: the INT8 engine would be compared against an FP16 baseline with part of it in FP32. AutoCast's top-level entry point (D-024) refuses quantized models.
**Options:**
1. ModelOpt `convert_to_f16` on our C3 Q/DQ models. This is the step ModelOpt's own quantizer runs after inserting Q/DQ (`quantization/precision_utils.py`, 0.47.0). It keeps C3's placement and percentile scales, so the ORT results of C3 still describe the quantization / scales are rounded to FP16.
2. ModelOpt `quantize()` end to end (its placement, `high_precision_dtype=fp16`). NVIDIA's whole path / its calibrators are only entropy or max, both worse than percentile on these models (D-019). Placement and calibration would change at once, so the C3 results wouldn't carry over.
3. Our own FP32→FP16 pass over the Q/DQ graph. No ModelOpt at this step / reimplements a maintained tool, the argument D-024 rejected.
**Decision:** Option 1 (Anuj). `edgeinfer quantize --float-dtype fp16` quantizes as before, then runs `tools/modelopt/qdq_fp16.py` (`keep_io_types: true`) and writes a content-addressed `int8_fp16` artifact. Its id hashes the parent INT8 id, the converter settings and `tools/modelopt/uv.lock`. The conversion is checked before the artifact is written:
- the quantized nodes and the Q/DQ nodes must be the same as the parent's (`QUANT_PLACEMENT_MISMATCH`, `QUANT_FP16_PLACEMENT_CHANGED`);
- the scale rounding is recorded (`scale_rounding`);
- the validator fails an `int8_fp16` model with FP32 compute (`VALIDATE_FLOAT_PRECISION`).
Parity gates `int8_fp16` like `int8` (D-018: mIoU only). ORT CPU runs `int8_fp16` models with graph optimizations off (see Consequences).
**Rationale:** The question C5 asks is what INT8 buys over the FP16 engine, so the float parts of an INT8 engine must be FP16, or the comparison mixes two changes. Converting our own models changes only the float types. The placement and scales whose accuracy C3 measured stay, so ORT and TensorRT sensitivity can be compared group by group. The comparison with ModelOpt's placement still happens, separately (D-015).
**Evidence:** Reports at `439d506` (`scripts/c5_runs.sh precheck`):
- Conversion (`quantize__<model>__int8_percentile_fp16__{none,sens}__439d506.json`): every node outside the Q/DQ pairs is FP16, with 2 boundary Casts and 0 FP32 compute nodes. Quantized nodes are unchanged.
- Scale rounding: SegFormer max relative change 0.05% (2 of 15,823 values subnormal in FP16). DeepLab 1.48% (42 of 17,747 subnormal); the worst is a weight channel with scale 2e-6.
- Validator: 0 errors, 0 warnings (`validate__<model>__int8_fp16__*__439d506.json`). Golden parity passes on ORT CPU (`parity__<model>__ort_cpu__int8_fp16__*__golden__439d506.json`).
- mIoU on `sens_val100`, ORT CPU unoptimized (`eval__<model>__ort_cpu__int8_fp16__*__sens_val100__512x1024__439d506.json`) against the C3 INT8 models:
  - SegFormer full INT8: 68.818 vs 68.873 (−0.055 points).
  - DeepLab full INT8: 65.663 vs 65.639 (+0.024).
  - DeepLab with `bb.stem_16` in float: 66.647 vs 66.665 (−0.018; C3 value from the greedy step of `sensitivity__deeplabv3plus_mbv2__int8_percentile__sens_val100__da584fa.json`).
  All three are inside the ~0.15-point level below which D-017 doesn't read effects as real.
**Consequences / revisit if:** ORT 1.31's CPU graph optimizations break FP16-scale Q/DQ models: they fuse MatMul+Add into `QGemm`, which rejects FP16 scales, so SegFormer doesn't load, and DeepLab loads but gives wrong outputs. `ort_cpu` therefore runs `int8_fp16` unoptimized, and its numbers check the conversion, not FP16 arithmetic: ORT CPU computes FP16 ops through casts (D-024). TensorRT is the real test. Scales below FP16's normal range lose precision. If TensorRT accuracy of `int8_fp16` falls short of `int8`, check `scale_rounding` first.

## D-032: TensorRT sensitivity — top 5 ORT groups, every subset generated on the Mac, leave-one-out, cold-rebuild noise floor  (C5, 2026-10-16)
**Context:** C5 re-runs sensitivity on TensorRT to see whether the ORT ranking (D-017) holds on the deployment runtime, within the GPU-hour budget. Each configuration needs an INT8+FP16 model (D-031), whose conversion runs in the ModelOpt environment on the Mac, and an engine build plus a 100-image score on the T4 (a few minutes each). The ORT ranking has one group per model clearly above the ~0.15-point level that D-017 treats as real (DeepLab `bb.stem_16`, recovery 1.03), and none for SegFormer (largest 0.15).
**Options:**
Groups to test: top 3 / top 5 / all (18 + 10).
Modes: leave-one-out only / also one-at-a-time.
Where configurations are made: on the Mac before the session / on the GPU (needs the ModelOpt environment, with PyTorch, in the container).
Noise floor: a second scoring with different threads (C3's method) / a second, cold build of the same model.
**Decision:** Top 5 per model (Anuj), leave-one-out only, configurations made on the Mac, cold-rebuild noise floor. `edgeinfer candidates` quantizes and converts every subset of the 5 groups (32 per model) and validates each. It writes a `candidates` report that lists each configuration's INT8 and INT8+FP16 artifact. `edgeinfer sensitivity --candidates` builds and scores them on TensorRT on `sens_val100`, against the FP32 engine on the same images. Greedy selection with margin 0.7 is unchanged (D-017). A configuration that wasn't generated, or a file that isn't the listed artifact, is an error (`SENS_CANDIDATE_MISSING`, `SENS_CANDIDATE_MISMATCH`).
**Rationale:** Five groups cover the only group with a real effect, plus four that let a different TensorRT ranking show up. The rest are inside the noise on ORT, and scoring them would cost GPU time to rank noise. One-at-a-time would need configurations with every other group excluded, which the subsets of five don't include. Leave-one-out is what greedy selection uses. Generating all 2⁵ subsets takes minutes on the Mac, so greedy can follow whatever order TensorRT gives without a trip back to the Mac mid-session. On TensorRT, what varies between runs of the same model is the builder's kernel choice (D-029), not thread scheduling. A second build without the timing cache, into a separate directory, measures that variation.
**Evidence:** The ORT ranking: `sensitivity__<model>__int8_percentile__sens_val100__{cb60f28,da584fa}.json`. `tests/test_candidates.py` covers subsets, lookup, refusals, and the scorer path with ORT standing in for TensorRT.
**Consequences / revisit if:** Groups below the top 5 stay INT8 in every TensorRT configuration. If even all five in FP16 can't reach the target, the report says `SENS_BUDGET_UNREACHABLE` for these groups, and the remaining groups need a new candidate set. A one-build noise floor is a single sample of the build-to-build variation, so read it as an order of magnitude.

## D-033: Final-engine rule and Pareto inputs, fixed before the INT8 results; matplotlib for plots  (C5, 2026-10-16)
**Context:** C5 ends with a final engine per model. FP16 already meets every budget (C4), so the choice is a trade-off between speed and accuracy among configurations that all pass. If the rule were chosen after seeing the numbers, it could be fitted to them. The Pareto plots need one accuracy axis and one latency axis for every configuration, and the repo has no plotting library.
**Options:**
Rule: lowest latency within budget / smallest accuracy loss within latency budget / weighted score.
Latency axis: trtexec compute p50 for every configuration (about 1 min each) / the probe's budgeted `infer` p99 for every configuration (about 15 min each).
Plotting: matplotlib / hand-written SVG / Vega-Lite JSON.
**Decision (Anuj, at plan approval):**
- **Rule.** Among finalists within the accuracy budget on full val and on `val_minus_sens100`, take the lowest probe `infer` p99. Latencies within 3.5% of the fastest are ties (build-to-build variation, D-029), and a tie goes to the configuration with the fewest quantized nodes. FP16 has none, so if no INT8 engine is faster beyond 3.5%, FP16 is the final engine. The rule is implemented in `analyze/pareto.py:select_final` and unit-tested.
- **Pareto axes.** y = mIoU change vs the FP32 engine on `sens_val100`. x = trtexec compute p50 at the 990 MHz lock, matched by engine artifact id. The probe runs for finalists and FP16 only.
- **Plotting.** matplotlib, in a `plots` dependency group used on the Mac only.
**Rationale:** Writing the rule down first makes the final choice an output of the measurements, not an argument about them. The tie margin is the measured build-to-build variation: a gap smaller than that between two engines can come from rebuilding one of them. Preferring fewer quantized nodes in a tie takes less accuracy risk for the same measured speed. trtexec agrees with the probe's compute span within 0.7–2.3% at the lock (C4), and the copies don't change with precision, so it ranks configurations correctly at a fifteenth of the cost. The probe still gates the finalists. matplotlib is the standard tool and is used only to draw a PNG from reports; SVG by hand would be more code for the same picture.
**Evidence:** `tests/test_pareto.py`; C4 probe/trtexec agreement in `trtexec__*__990MHz__3a086ea.json`.
**Consequences / revisit if:** Pareto accuracy is on 100 images, so differences below ~0.15 points aren't meaningful (D-017). Finalists are confirmed on full val and held out before the rule applies. The rule ignores memory, because every engine is far below the 953 MiB budget (C4). Revisit that if an INT8 engine comes close to it.
