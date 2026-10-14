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
**Decision:** Option 1. `requires-python >= 3.10` stays; CI tests 3.10 and 3.12.
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
