# Roadmap

Checkpoints C1–C7. C1–C3 run entirely on a CPU dev machine; C4–C5 need the GPU; C6 spans both. If the GPU becomes available early, C4 can start after C2.

Each checkpoint lists tasks, exit criteria, decisions to discuss (with the default recommendation), and domain pitfalls.

---

## C1 — Baseline and eval harness (CPU)
**Goal:** trustworthy FP32 reference numbers and a frozen eval harness.

Tasks:
1. Repo skeleton: `uv`, `pytest`, `ruff`, pre-commit, GitHub Actions (lint + unit tests), `.gitignore` covering data, artifacts and private files.
2. Cityscapes loader, labelIds → 19 trainIds mapping (`cityscapesScripts` or an equivalent tested mapping), ignore index 255, manifests (eval, calib, golden). Seeded and deterministic.
3. Model registry: load both models in PyTorch from configs.
   - SegFormer-B0: `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` (verify availability; record the license).
   - CNN: candidate DeepLabV3+ MobileNet with Cityscapes weights (e.g. `VainF/DeepLabV3Plus-Pytorch`). Verify weights and license.
4. Config-driven preprocessing and postprocessing.
5. mIoU via one confusion matrix accumulated over the whole dataset (not averaged per image).
6. FP32 eval on CPU for both models; write eval reports.
7. Golden outputs: FP32 logits for `golden_val32` as `.npz` in `artifacts/golden/`; commit their SHA-256 hashes.
8. Unit tests: mIoU against a hand-computed example, label mapping, preprocessing on a known input, manifest determinism.

**Exit criteria:** both models evaluated on all 500 val images at the chosen resolution; numbers plausible versus published results after accounting for resolution and protocol (explain any gap); golden outputs saved with hashes; CI green.

**Decisions:** CNN model choice; input resolution (default 512×1024 — published numbers use higher resolution, a real-time budget pushes lower); where upsampling lives (default: outside the graph in postprocess, included in end-to-end timing); golden set size and format.

**Pitfalls:** SegFormer outputs logits at 1/4 input resolution — upsample (bilinear) before argmax. Published SegFormer Cityscapes numbers use higher resolution and sliding-window inference, so a 512×1024 baseline will be lower; the FP32 baseline at deployment resolution is the reference.

---

## C2 — ONNX export, validator, ORT backend, parity (CPU)
**Goal:** a validated ONNX model per architecture that matches PyTorch, plus the validator and parity analyzer.

Tasks:
1. Export both models to ONNX (static batch-1 shape by default).
2. Validator: ONNX checker, shape inference, op inventory, ops not in `configs/supported_ops.yaml` (seeded from the TensorRT ONNX parser's supported-operator list; provisional until C4), I/O shape and dtype checks, NaN/Inf in initializers, preprocessing config vs model input consistency.
3. `Backend` protocol + `torch_ref` and `ort_cpu` backends, including `layer_outputs` (ORT: add intermediate tensors as graph outputs on a copy of the model).
4. Parity analyzer against golden outputs: max abs error, mean abs error, cosine similarity on logits, pixel-level argmax agreement, mIoU delta. Per-layer mode reports the first layer whose drift exceeds tolerance.
5. Deliberate-break test: swap RGB↔BGR via config → parity fails loudly with the right finding code → revert.
6. Integration test with a tiny fixture model (random small CNN, no Cityscapes data) so CI exercises export → validate → ORT → parity.

**Exit criteria:** both ONNX models pass the validator; ORT FP32 matches PyTorch within tolerance; the deliberate break is caught; per-layer mode works; CI runs the fixture pipeline.

**Decisions:** exporter (`torch.onnx.export(dynamo=True)` vs legacy TorchScript exporter vs Hugging Face Optimum — default: try dynamo, fall back with a recorded reason); opset (≥17 for native LayerNormalization, and supported by the pinned TensorRT parser); static vs dynamic shapes (default static — the camera resolution is fixed); parity tolerances (set from the observed FP32-vs-FP32 distribution, then justified).

**Pitfalls:** Hugging Face models may need `return_dict=False` or Optimum to export cleanly — verify outputs right after export. Use ONNX tensor names as the shared key across backends.

---

## C3 — INT8 quantization and sensitivity analysis on ORT (CPU)
**Goal:** working INT8 PTQ and a sensitivity analyzer, developed cheaply on CPU in a form that transfers to TensorRT.

Tasks:
1. PTQ producing a **QDQ ONNX** model, calibrated on `calib_train200`.
   - TensorRT expects **signed INT8, symmetric, zero-point 0**. ORT's `quantize_static` defaults to asymmetric uint8 activations — configure int8/symmetric explicitly.
   - Check whether NVIDIA ModelOpt's ONNX PTQ runs on macOS; if so, prefer it so the same QDQ model goes to both backends.
   - Weights per-channel symmetric; activations per-tensor.
2. Run INT8 through ORT CPU; evaluate mIoU on val; per-layer parity vs FP32.
3. Sensitivity analyzer:
   - Group layers into blocks (SegFormer: encoder stages/blocks, decode head; CNN: backbone stages, ASPP/decoder, classifier).
   - Modes: **one-at-a-time** (quantize only group X) and **leave-one-out** (quantize all except group X).
   - Output: groups ranked by mIoU impact + a suggested exclusion list that meets the accuracy budget.
4. Compare calibration methods (at least minmax vs entropy or percentile) on both models.

**Exit criteria:** INT8 QDQ models for both architectures; accuracy and parity reports; sensitivity rankings; a report-backed finding on CNN-vs-transformer quantization behavior; the exclusion-list mechanism working end to end.

**Decisions:** quantization tool (ModelOpt ONNX PTQ vs ORT `quantize_static`); calibration method; sensitivity granularity (cost grows with the number of groups, and on TensorRT each configuration is a new engine build); calibration set size.

**Pitfalls:** ORT CPU INT8 speed is meaningless and its accuracy may differ from TensorRT INT8 — expect and document it. Expect transformer sensitivity in LayerNorm, softmax/attention, patch embedding and the classifier, but let the analyzer show it.

---

## C4 — GPU environment, TensorRT FP32/FP16, performance probe (GPU)
**Goal:** a reproducible GPU setup, TensorRT engines and trustworthy latency numbers.

Tasks:
1. `scripts/gpu-up.sh`, `gpu-down.sh`, `gpu-status.sh` (AWS CLI, instance found by tag); `scripts/remote.sh "<cmd>"` rsyncs the repo (excluding data and artifacts) to the instance and runs inside the container. Reports come back to the Mac; engines stay on the instance.
2. Confirm the CloudWatch idle auto-stop alarm and budget alerts exist (ask; don't assume).
3. Pin the toolchain: latest TensorRT supporting sm_75 (check NVIDIA's support matrix), matching NGC container, `docker/Dockerfile.trt`, env report.
4. `trt` backend: build FP32 and FP16 engines with explicit builder settings (workspace limit, optimization level, timing cache); engine metadata sidecar (TensorRT version, GPU, CUDA, driver, builder flags). Refuse to load an engine whose metadata doesn't match the runtime. Validate `supported_ops.yaml` against the real parser.
5. Performance probe: warmup, measured iterations, CUDA events with proper synchronization; p50/p90/p99/max/mean/std and throughput; separate timing for preprocess, H2D copy, compute, D2H copy, postprocess and end to end; device memory and engine size; GPU clocks and temperature. Cross-check compute latency with `trtexec`.
6. Nsight Systems trace of the end-to-end path for one model; document where the time goes.
7. Parity for TRT FP32 and FP16. Per-layer debugging uses a **separate debug engine** with marked outputs (marking outputs disables fusion, so it is not the deployed engine); never benchmark the debug engine.
8. Finalize budgets in `configs/budgets.yaml` from measured baselines (decision record).

**Exit criteria:** FP32 and FP16 engines for both models; perf and parity reports; Nsight Systems summary in docs; budgets finalized.

**Decisions:** TensorRT version/container; builder optimization level; whether GPU clocks are locked for benchmarking (consistency vs realism); iteration counts; CPU vs GPU preprocessing in the Python pipeline (default CPU, so the C++/CUDA runtime's improvement is measurable).

**Pitfalls:** synchronize before reading timers; warm up; report the distribution, not just the mean; the T4 is a 70 W card and can throttle — record clocks and temperature and report run-to-run variation. Engines are tied to the GPU model, TensorRT version and often the driver.

---

## C5 — TensorRT INT8 and mixed precision (GPU)
**Goal:** a mixed-precision engine per model that meets the budgets, plus the trade-off curve.

Tasks:
1. Build INT8 engines from the C3 QDQ models (explicit quantization). Check whether implicit/calibrator-based INT8 is deprecated in the pinned TensorRT version and record the finding.
2. Re-run sensitivity on TensorRT within the GPU-hour budget: start from the ORT ranking and test only the top-N groups. Differences between the ORT and TRT rankings are a finding.
3. Mixed-precision selection: start from full INT8, move the most sensitive groups back to FP16 until the accuracy budget is met, then check latency. Record each step.
4. Pareto plot of latency vs mIoU for every configuration tried, per model.
5. Explain from the traces why INT8 speedup over FP16 is usually below 2× (reformat layers between precisions, memory-bound ops, non-quantized layers).

**Exit criteria:** a final engine per model meeting the budgets, or a documented explanation of why it can't plus the best trade-off; Pareto plots generated from reports; findings written up.

**Decisions:** top-N groups to test on TensorRT; selection algorithm (greedy vs exhaustive over a small set); acceptable trade-off if both budgets can't be met.

---

## C6 — Gated deployment pipeline (CPU + GPU)
**Goal:** rollouts that are fast and predictable.

Tasks:
1. Stages `export → validate → quantize (optional) → build → parity gate → perf gate → publish`, each idempotent and skipped when its content-addressed output already exists.
2. Gates from `configs/gates.yaml`; a failed gate stops the pipeline with a clear report and non-zero exit.
3. Publish: register artifact + reports + metadata (MLflow model registry with a local file store unless decided otherwise); support rollback to the previous passing version.
4. Airflow DAG wrapping the CLI (local Airflow standalone with the ORT backend; GPU stages via the CLI on the instance, or an SSH operator if decided).
5. CI: lint, unit tests and the fixture-model pipeline on every push; GPU tests skipped in CI and run manually.
6. Demonstrate a blocked rollout: run the pipeline on a deliberately broken config and show the gate stopping it.

**Exit criteria:** one command deploys a model end to end with gates; a broken model is blocked; the Airflow DAG runs; rollback works; CI green.

**Decisions:** orchestrator (default Airflow; alternative Temporal); registry (MLflow vs an artifact directory + index); how GPU stages are triggered.

---

## C7 — Results and write-up
Tasks:
1. `scripts/make_results.py` generates `docs/results.md` tables and plots from reports only.
2. README: problem, architecture diagram, results table (per model × precision: mIoU, p50/p99, memory, engine size), Pareto plots, key findings, limitations, reproduction steps, licenses.
3. Limitations: the T4 is a datacenter Turing GPU, not vehicle hardware. Explain what would differ on an Orin-class target: unified memory (no host↔device copy), DLA, lower clocks and power modes, a different TensorRT build.

**Exit criteria:** README and results complete and reproducible from a clean clone plus data access.

---

## Definition of done
- [ ] C1–C7 exit criteria met and recorded in `docs/progress.md`
- [ ] Every number in README/results traces to a report with git SHA + env
- [ ] Both models have a final engine and an explained trade-off
- [ ] The pipeline blocks a broken model; CI green
- [ ] No Cityscapes data or derived tensors in git
- [ ] `docs/decisions.md` complete, with no TODO markers
- [ ] GPU instance stopped
