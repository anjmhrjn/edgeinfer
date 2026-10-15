# CLAUDE.md — edgeinfer

Toolkit and experiments for taking PyTorch perception models to TensorRT under real-time latency, accuracy and memory budgets, with the tooling to prove the optimized model is still correct.

- Roadmap and per-checkpoint plans: `docs/roadmap.md`
- Current status: `docs/progress.md`
- Decision records: `docs/decisions.md`

## What this project builds
- **Validator** — checks an exported ONNX model before it's built (ops, shapes, dtypes, weights, preprocessing/input consistency).
- **Performance probe** — latency distribution (p50/p90/p99), throughput, per-stage timing, device memory, engine size.
- **Parity analyzer** — compares any backend's outputs against FP32 golden outputs, whole-model and per-layer.
- **Sensitivity analyzer** — ranks layer groups by accuracy impact under INT8 and proposes a mixed-precision config.
- **Gated deployment pipeline** — export → validate → quantize → build → parity gate → perf gate → publish, with rollback.

**Models:** SegFormer-B0 (transformer) and a Cityscapes-trained CNN. Two architectures on purpose: CNNs usually quantize to INT8 cleanly, while transformers degrade at LayerNorm, softmax and attention. The analyzers must be able to show this contrast.

**Task:** Cityscapes semantic segmentation, batch 1, RGB input (default 3×512×1024), 19 classes, metric mIoU.

**Precisions:** FP32 (reference) → FP16 → INT8 PTQ (explicit QDQ) → sensitivity-guided mixed precision.

**Budgets** (finalized in C4 from measured baselines; see `configs/budgets.yaml`):
- p99 ≤ 33.3 ms end-to-end at batch 1 on the T4 (one frame at 30 FPS)
- mIoU drop ≤ 1.0 point vs FP32 at the same resolution
- ≤ 1 GB device memory

## How to work in this repo
- At session start, read `docs/progress.md` and state the current checkpoint and next step in 1–3 lines.
- Each checkpoint is planned in plan mode from its section in `docs/roadmap.md`. A plan lists steps, files, tests, exit criteria and **"Decisions needed"** (options, trade-offs, recommendation). Don't make non-trivial decisions silently.
- Record every non-trivial decision in `docs/decisions.md` using the template there.
- Work in small steps; run tests after each; commit at each green step with a clear message.
- Verify exit criteria by running them and showing the output.
- Version-specific facts (TensorRT APIs, ModelOpt flags, ONNX opset support, NGC container contents) change between releases. Verify them against official docs before relying on them.
- Out-of-scope ideas go in `docs/ideas.md`. No new dependency without stating why and what the alternative was.

## Integrity rules
- Every reported number must trace to a JSON report in `artifacts/reports/` that records the git SHA and environment. Never write estimated numbers into docs.
- Never loosen a tolerance or gate to make something pass without a decision record justifying the new value.
- Investigate results that look too good before reporting them (missing GPU synchronization, caching, measuring the wrong span, evaluating on calibration data).
- Don't silence failures: no bare `except`, no skipped tests to get CI green, no `|| true`.

## Environment
- **Dev machine:** macOS arm64, 16 GB unified memory, no NVIDIA GPU. Runs everything except TensorRT/CUDA through the ORT CPU backend. Reference runs use CPU, not MPS. Stream data; keep eval batch size small. Mac latency numbers are for development only.
- **GPU:** AWS EC2 `g4dn.xlarge` — 1× NVIDIA T4 (Turing, compute capability 7.5 / sm_75, 16 GB). Deep Learning Base GPU AMI (Ubuntu), work runs inside a pinned NGC container. Managed with `scripts/gpu-up.sh`, `gpu-down.sh`, `gpu-status.sh`, `remote.sh`.
  - Prepare and test on the Mac first; GPU sessions are for running, not writing code.
  - **Stop the instance at the end of every GPU session** (`scripts/gpu-down.sh`). Don't create other billable AWS resources without asking.
- **Toolchain:** pinned in C4 to the latest TensorRT supporting sm_75. Versions recorded in `docs/decisions.md` and in every report's `env` block.
- **Python:** ≥ 3.11 (D-011; must run on both the Mac and the container).
- **Data:** Cityscapes. The license prohibits redistribution: never commit images, labels or derived tensors (golden outputs, calibration caches), and never upload them anywhere public. Data lives at `$CITYSCAPES_ROOT` on the Mac and `/data/cityscapes` on the instance. Only manifests (image IDs) are committed:
  - `data/manifests/eval_val.txt` — full val split (500 images)
  - `data/manifests/calib_train200.txt` — 200 seeded train images; never calibrate on val
  - `data/manifests/golden_val32.txt` — 32 val images for parity

## Architecture

```
configs/           models/*.yaml (weights, input shape, pre/postprocessing), budgets.yaml, gates.yaml, tolerances.yaml, supported_ops.yaml
data/manifests/    image ID lists only
src/edgeinfer/
  data/            Cityscapes loader, manifests, label mapping
  models/          model registry, PyTorch loading
  preprocess.py    config-driven; single source of truth
  postprocess.py   upsample + argmax
  eval/            confusion-matrix mIoU, eval harness
  export/          PyTorch -> ONNX
  validate/        ONNX validator
  backends/        base.py, torch_ref.py, ort_cpu.py, trt.py
  quant/           PTQ -> QDQ ONNX, calibration reader, exclusion lists
  analyze/         parity.py, sensitivity.py
  perf/            performance probe
  pipeline/        gated stages, artifact store
  reports.py       report envelope + schema validation
  cli.py
pipelines/airflow/dags/deploy_model.py
scripts/           gpu-*.sh, remote.sh, make_results.py
docker/Dockerfile.trt
tests/             unit + integration; GPU tests marked @pytest.mark.gpu
artifacts/         golden/, onnx/, engines/ gitignored; reports/ committed (D-010)
docs/              roadmap.md, progress.md, decisions.md, results.md, report-codes.md, ideas.md
```

### Design principles
1. **Backend abstraction.** Tools talk to a `Backend`, never to ORT or TensorRT directly (`ort_cpu` on the Mac, `trt` on the GPU).
2. **Every tool is a library function plus a CLI command that writes a JSON report.** Outputs are machine-readable, stable and deterministic.
3. **Preprocessing is config-driven:** resize size and interpolation, mean/std, channel order, layout, dtype. No hidden constants in code.
4. **Artifacts are content-addressed:** ID = hash of (weights, model config, quantization config, toolchain versions), with a `*.meta.json` sidecar.
5. **Analysis tools are read-only.** They never modify their inputs.

```python
class Backend(Protocol):
    name: str                                    # "torch_ref" | "ort_cpu" | "trt"
    def load(self, artifact: Path | None, cfg: ModelConfig) -> None: ...  # torch_ref: None
    def infer(self, x: np.ndarray) -> np.ndarray: ...   # NCHW float32 in, logits out
    def layer_outputs(self, x: np.ndarray, names: list[str]) -> dict[str, np.ndarray]: ...
    def metadata(self) -> dict: ...              # precision, versions, device, artifact id
```

### Report envelope (every report)
```json
{
  "schema_version": "1", "tool": "parity", "created_at": "...", "git_sha": "...",
  "env": {"host": "...", "device": "...", "versions": {}},
  "model": "segformer_b0", "backend": "trt", "precision": "int8", "artifact_id": "...",
  "inputs": {"manifest": "golden_val32"}, "thresholds": {}, "metrics": {}, "passed": true,
  "findings": [{"severity": "error|warn|info", "code": "PARITY_LAYER_DRIFT", "message": "...", "where": "..."}]
}
```
Finding codes are a stable, documented enum in `docs/report-codes.md`.

### CLI
`edgeinfer eval | export | validate | quantize | build | parity | perf | sensitivity | deploy | report`
Common flags: `--model --backend --precision --artifact --manifest --out`. Non-zero exit code on failed checks.

## Downstream consumers (not built in this repo)
- **C++ real-time runtime + custom CUDA kernel** (separate repo). Loads these TensorRT engines in a frame loop under a deadline. Requires:
  - self-describing engine metadata (I/O names, shapes, dtypes, precision, TensorRT version)
  - a fully specified preprocessing config it can reimplement exactly
  - `edgeinfer parity --from-npz` to check its outputs against the golden set
- **Agentic deployment-triage tool** (separate repo). Calls these tools through an MCP server to root-cause seeded faults. Requires:
  - the stable report envelope and finding codes
  - faults injectable through configuration only (preprocessing overrides, exclusion lists, calibration manifest swaps, precision flags)
  - deterministic, read-only tools with specific error messages

## Commands
- Setup: `uv sync && uv run pre-commit install`
- Lint/format: `uv run ruff check . && uv run ruff format --check .`
- Tests (CI subset): `uv run pytest -m "not gpu and not data and not weights"`
- All CPU tests: `uv run pytest -m "not gpu"` (needs `$CITYSCAPES_ROOT` and downloaded weights)
