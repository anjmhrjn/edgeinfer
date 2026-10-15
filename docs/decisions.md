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
