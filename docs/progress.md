# Progress

**Current checkpoint:** C6 — Gated deployment pipeline (CPU + GPU)
**Next step:** C6 write-up: teach-back (`private/teach-back-C6.md`), then Anuj signs off C6 below. C7 is next. The GPU instance is stopped, not terminated; its engines back the registry's `production` versions (D-037).

## C6 — Gated deployment pipeline
Plan agreed 2026-10-17:
- **Registry (D-037).** Our own file registry, `artifacts/registry/<model>.json`, metadata only and committed. It holds versions and a `production` alias; rollback moves the alias to the previous passing version after re-checking its sha256.
- **Orchestration (D-038).** `edgeinfer deploy` runs on the Mac: local stages in-process, GPU stages through `scripts/remote.sh run`, one call per stage. The Airflow DAG runs one BashOperator per stage (`edgeinfer deploy --stage <name>`). Airflow sits in an isolated `tools/airflow` environment, and the pipeline never imports it.
- **Gates (D-039).**
  - Every precision: validate and golden parity.
  - `int8` / `int8_fp16`: also full-val accuracy, drop ≤ 1.0 against the FP32 artifact on the same backend.
  - Perf: the probe against the budgets.
  - `configs/gates.yaml` refers to budgets and tolerances and repeats no numbers.
- **Gate reuse (D-040).** A passing gate report is reused when all of these are unchanged: artifact id (plus engine sha256), resolved thresholds, manifest hash, and code since the report's SHA (`src/edgeinfer` without `pipeline/`, plus `uv.lock`). The report must also come from a clean tree. Failed reports are never reused.
- **Channels.** An `ort_cpu` deploy, where perf isn't enforced, publishes only to `dev`; `production` needs every gate enforced. Publish refuses a dirty tree. Exit codes: 0 published, 1 blocked, 2 config/environment.
- **Blocked-rollout demos.** A SegFormer INT8+FP16 recipe without D-034's exclusion (GPU) and a BGR fixture recipe (CI).
- **GPU time.** One session, ~2 GPU-hours (est.).

Mac:
- [x] Airflow spike (`tools/airflow`, 2026-10-17):
  - Airflow 3.3.2 is the latest stable (PyPI, released 2026-09-17; 3.4.0 is in beta). It is tested on Python 3.10–3.14. Macs are supported for development only, and Linux is the only supported production environment ([prerequisites](https://airflow.apache.org/docs/apache-airflow/stable/installation/prerequisites.html)).
  - The official install uses a constraints file per Airflow and Python version ([installing from PyPI](https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-pypi.html)). `uv add -c` applies it when resolving but records it neither in `pyproject.toml` nor in `uv.lock`. So the official pins for the 125 resolved non-Airflow packages are written into `[tool.uv] constraint-dependencies`, and a later relock keeps the tested set.
  - Airflow 3 import paths: `from airflow.sdk import DAG`; `from airflow.providers.standard.operators.bash import BashOperator`. The standard provider (1.19.0 in the constraints) ships with `apache-airflow-core`.
  - `airflow dags test` on macOS arm64 runs a DAG in-process without a scheduler, and `{{ params.* }}` templating works. A non-zero exit fails the task and the run. BashOperator's `skip_on_exit_code` defaults to 99, so our exit codes 0/1/2 keep their meaning.
- [x] Recipe and gates configs and loaders (`pipeline/config.py`, `configs/gates.yaml`, `configs/deploy/`)
- [x] Pipeline core: stages, executors, gate reuse, deploy report (`pipeline/runner.py`, `executor.py`, `reuse.py`)
  - Producing commands (export, quantize, fp16, build) now report `reused`: the output file predates the command, because an artifact with the same id already existed.
  - `remote.sh sha256 <path>` checks a file on the instance, for rollback.
  - Found while writing it: a committed registry would mark every later report `git_dirty`. After a publish it is a modified tracked file, and on the instance, where it isn't synced, git sees it as deleted. `reports.git_state` now leaves `artifacts/registry/` out of the dirty check: it records results and is never an input to a measurement (D-037).
- [x] Registry: publish, aliases, rollback (`pipeline/registry.py`); tests `test_registry.py`, `test_reuse.py`, and `test_deploy.py` on the fixture
- [x] `edgeinfer deploy run|rollback|status` (`run --stage --run-id` for Airflow, `--dry-run` judged with the recipe's last run) and the `DEPLOY_*`/`REGISTRY_*` codes in `docs/report-codes.md`
- [x] Fixture end to end in CI (`tests/test_deploy.py`): pass → `dev`; rerun all skipped/reused; changed code reruns the gates; BGR blocked at parity (`PARITY_INPUT_MISMATCH`); failed checks never reused; one stage at a time; dirty tree blocks publish; CLI dry run, rollback, status
- [x] DeepLab's final model on ORT CPU (`deploy run --recipe deeplabv3plus_mbv2__ort_cpu`), published as v1 to `dev` (`artifacts/registry/deeplabv3plus_mbv2.json`):
  - **Cold run** (`deploy/deeplabv3plus_mbv2__ort_cpu/20261017T182358Z/deploy.json`): 27 min 49 s. Export and convert were skipped: the commands reused C5's FP32 export and INT8+FP16 model `0fc604c5e1e0`. The gates ran: validate 0.1 s, golden parity 17.4 s, accuracy 1230.9 s (two full-val evals), perf 419.7 s (not enforced, `DEPLOY_GATE_NOT_ENFORCED`).
  - **Accuracy gate.** FP32 67.086 (as in C2/C4, `e264b0b`, `7ac2253`), INT8+FP16 66.812: a drop of 0.274 points against the 1.0 budget. TensorRT's engine of the same model scored 66.758 in C5, 0.054 points away.
  - **Warm rerun** (`…/20261017T185213Z/deploy.json`): 1.1 s. Every stage was skipped or reused, and publish found v1 already registered.
  - **TensorRT recipes, Mac stages run in advance** (`deploy run --stage export|convert|validate`, runs `20261017T1852*Z`). Each resolves to the C5 artifact it is meant to deploy: DeepLab `0fc604c5e1e0` (INT8+FP16, stem FP16), SegFormer `e15dc3063aac` (FP16), DeepLab runner-up `d3dcca5f82e9` (C3 INT8+FP32), SegFormer fault `4031ccefacfb` (INT8+FP16, no exclusions). `--dry-run` of these recipes can only say every GPU gate runs: there is no earlier GPU run.
- [x] Airflow DAG (`pipelines/airflow/dags/deploy_model.py`), import test (`tests/test_airflow_dag.py`, CI job `airflow-dag`), and two local runs (`airflow dags test`, Airflow 3.3.2, Mac):
  - **`deeplabv3plus_mbv2__ort_cpu`.** All nine tasks succeeded in one pipeline run (`deploy/deeplabv3plus_mbv2__ort_cpu/20261017T185619Z/deploy.json`). Every stage was skipped or reused, and v1 was already registered.
  - **`faults/deeplabv3plus_mbv2__bgr_ort_cpu`** (DeepLab FP32 with BGR preprocessing). Export ran (4.2 s, a new id `06beefdf8b6a` with a BGR contract) and validate passed, because the contract matches the config. **Parity failed**: `PARITY_INPUT_MISMATCH` ("channels are reversed… check input.channel_order"), and mIoU on the golden images fell from 67.53 to 56.26. Then accuracy, perf and publish were `upstream_failed`, the DAG run failed, and the registry was unchanged (`deploy/deeplabv3plus_mbv2__bgr_ort_cpu/*/deploy.json`).
- [x] `scripts/rsync-excludes.txt` excluded only the root `.venv/`, so `remote.sh` copied `tools/modelopt/.venv` (929 MB), and now would copy `tools/airflow/.venv`, to the instance. It now excludes `.venv/` at any depth.
- [x] The TensorRT path of the runner tested on the Mac with a stub GPU executor (`test_trt_target_with_a_stub_gpu`). It checks that:
  - the candidate and FP32 ONNX models are pushed with their sidecars, and both are built;
  - the gates measure the engine;
  - the deployment reaches `production` at its `instance:` location;
  - a rerun only re-invokes the builds, which reuse their engines.
- [x] Decision records D-037–D-040; `docs/ideas.md`: persistent engine store, MLflow mirror, Temporal

GPU session (2026-10-17, 19:16–21:44 UTC, 2.47 GPU-hours; plan ~2):
- [x] Smoke 60/60.
- [x] **Found on the T4: `remote.sh` overwrote the runs' state.**
  - What happened: `sync_code` pulls the instance's reports before every remote call. The instance's copy of a run's `deploy.json` was the one synced at the run's first GPU stage, and `rsync -rlptz` replaces a file whose size or mtime differs, so each pull replaced the Mac's newer copy. The runner rewrote the file from memory after every stage, which hid this until the next run's first remote call reset it for good.
  - Effect on the first pass of five runs: every gate and publish worked, but four `deploy.json` files kept only the stages through `push`. So the warm rerun found no reuse keys and re-ran parity and accuracy, about 14 GPU-minutes. Its perf stage then hit an SSH timeout and stopped with exit 2 (`DEPLOY_STAGE_ERROR`), publishing nothing.
  - Fix (`2602921`): the pull uses `--update`, so a file newer on the Mac is never replaced. Reproduced and checked locally with the same flags first. On the instance, each `--stage` call reloads `deploy.json` from disk, and the state survived every remote call.
  - Why CI missed it: the stub GPU executor replaces `remote.sh`.
  - Recovery (Anuj's choice): the registry was reset to its committed state, and the gated runs were repeated and published in the original order (about 45 extra GPU-minutes). The four incomplete runs (`deeplabv3plus_mbv2/20261017T191813Z`, `…/20261017T200309Z`, `segformer_b0/20261017T194737Z`, `segformer_b0__int8fp16_no_unfused/20261017T201815Z`) were deleted on the Mac (Anuj), so no committed label points at a truncated state file. Their records up to `push` were intact, so the repeated runs' validate checks name them in `reused_from`; the validate reports those checks use are committed (`deploy/<recipe>/20261017T1852*Z/validate.json`, `…__ort_cpu/20261017T182358Z/validate.json`). Only the last run of the first pass, the INT8 runner-up's, was intact (`deploy/deeplabv3plus_mbv2__int8/20261017T201932Z/deploy.json`), and it is kept.
- [x] **DeepLab final** (INT8+FP16, stem in FP16) → **v2 `production`** (`deploy/deeplabv3plus_mbv2/20261017T205607Z/deploy.json`).
  - Run stage by stage (`--stage`), with publish held until the registry was reset.
  - Export, convert and build were skipped: the C5 artifacts (`0fc604c5e1e0` ONNX, `8ff4db1b2872` engine).
  - Gates, 1351.7 s in all:
    - golden parity 55.9 s, −0.360 points;
    - accuracy 413.2 s: full val 66.758, a drop of 0.328 against the 1.0 budget, the same as C5. The FP32 reference eval (67.086) was reused from the INT8 run, since it uses the same FP32 engine and the same key;
    - perf 861.3 s: `infer` p99 6.486 ms (≤ 33.3), 179 MiB.
- [x] **Warm rerun, one command** (`…/20261017T213611Z/deploy.json`): 19.6 s across stages (20 s wall), against 1351.7 s cold.
  - Every stage was skipped or reused, and publish found v2 already registered.
  - 14.0 s of it is the build stage: two remote calls that confirm the engines exist. One `remote.sh run` costs about 7 s of sync and pull-back (SegFormer's single build call took 7.1 s; push took 3.7–6.3 s) (D-038).
- [x] **SegFormer final** (FP16) → **v1 `production`** (`deploy/segformer_b0/20261017T211841Z/deploy.json`).
  - Gates, 929.3 s: golden parity +0.0002 points; accuracy skipped (FP16, D-039); perf `infer` p99 9.466 ms, 299 MiB.
- [x] **SegFormer fault** (INT8+FP16 without D-034's exclusions) **blocked**, exit 1 (`deploy/segformer_b0__int8fp16_no_unfused/20261017T213435Z/deploy.json`).
  - `DEPLOY_GATE_FAILED` where = `parity`, from `PARITY_MIOU_DELTA`: mIoU on the golden images 71.45 → 0.06.
  - Accuracy, perf and publish `not_run`, registry unchanged. 74.3 s in all.
- [x] **INT8 runner-up** (C3's INT8+FP32, stem in FP32) → **v3 `production`**.
  - One cold command, 1807.8 s (`deploy/deeplabv3plus_mbv2__int8/20261017T201932Z/deploy.json`): golden parity −0.306; full val FP32 67.086, INT8 66.875, a drop of 0.211 (as in C5); `infer` p99 6.950 ms, 203 MiB.
  - Republished after the reset by a rerun that reused every gate, 20.0 s (`…/20261017T213412Z/deploy.json`).
- [x] **Rollback** `production` v3 → v2 (`registry/rollback__deeplabv3plus_mbv2__production__20261017T213644Z.json`). It checked v2's engine sha256 on the instance first (`remote.sh sha256`), and `deploy status` shows v3 "rolled back from production".
- [x] **Airflow DAG on the TensorRT target** (`airflow dags test deploy_model -c '{"recipe": "segformer_b0"}'`, Mac → instance). All nine tasks succeeded in 15 s (`deploy/segformer_b0/20261017T213659Z/deploy.json`, 11.0 s across stages, every stage skipped or reused).
- [ ] **Next GPU session, before any `remote.sh` call:** delete those four run directories on the instance (`scripts/remote.sh host 'cd ~/edgeinfer/artifacts/reports/deploy && rm -r …'`), or the first sync's pull brings them back.
- [x] `deploy status` printed `vNone` for an alias's first move; it now prints `-`.

### C6 exit criteria
- **One command deploys end to end with gates.**
  - `deploy run --recipe configs/deploy/deeplabv3plus_mbv2__int8.yaml` ran every gate cold and exited 0 with the version in `production` (`…__int8/20261017T201932Z`). So did the ORT CPU run, to `dev` (`…__ort_cpu/20261017T182358Z`).
  - The two finals ran the same stages one at a time (`20261017T205607Z`, `20261017T211841Z`), which is also what the Airflow tasks do.
- **Fast and idempotent.** DeepLab final rerun 19.6 s vs 1351.7 s, every stage skipped or reused (`20261017T213611Z`). ORT CPU 1.1 s vs 27 min 49 s.
- **A broken model is blocked.** SegFormer fault on the T4 and the BGR fault on ORT CPU: both exit 1 with `DEPLOY_GATE_FAILED` where = `parity`, nothing published. In CI: `test_preprocessing_fault_is_blocked_at_parity`.
- **Rollback works.** v3 → v2 after the sha check; `deploy status` above.
- **The Airflow DAG runs.** Locally on ORT CPU (pass, and the BGR fault blocked) and on the TensorRT target (pass). The import test passes in CI (`airflow-dag` job).
- **CI green.** GitHub Actions run 39061685590 (`9b09beb`): `lint-and-test` on 3.11 and 3.12 and `airflow-dag`.

**Teach-back:** model answers in `private/teach-back-C6.md`.

**Sign-off:** pending (Anuj)

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
- [x] ModelOpt placement for both models, DeepLab min/max with ours (`quantize__<model>__int8_minmax_fp16__{ort,modelopt}__2507a9e.json`, validator clean). ModelOpt's per-node calibration on 200 images took 4.4 h (SegFormer) and 2.0 h (DeepLab) on the Mac. Quantized nodes, ours vs ModelOpt:
  - DeepLab: Conv 60 vs Conv 60 + Add 10 + Resize 2 + ReduceMean 1
  - SegFormer: Conv 20 + MatMul 68 vs those + LayerNormalization 26 + Add 16 + Transpose 6 + Resize 4 + Reshape 3
- [x] CI green on GitHub for the C5 Mac commits (latest run 39001542370, `2507a9e`)

GPU session 1 (2026-10-17, 00:30–02:05 UTC, 1.57 GPU-hours):
- [x] Smoke: 59/60 GPU tests at first. The INT8 fixture test assumed full fusion, but the fixture has one Conv whose shared DQ doesn't fuse; the test was corrected. `test_implicit_int8_is_gone` confirms calibrator INT8 is removed in 11.3 (D-031). trtexec 11.3: `--separateProfileRun` is a no-op, so it's dropped from the profile flags.
- [x] INT8 engines (`build__<model>__<label>__5efd53c.json`): DeepLab fuses every quantized node. SegFormer leaves 9 of 88 unfused in INT8+FP32 and 17 of 88 in INT8+FP16.
- [x] Golden parity on TensorRT (`parity__<model>__trt__<label>__golden_val32__golden__*.json`): both C3 INT8+FP32 models pass (SegFormer −0.38, DeepLab −0.69 full / −0.31 with the stem in float). DeepLab INT8+FP16 passes (−0.60 / −0.36). **SegFormer INT8+FP16 fails completely (mIoU 71.45 → 0.06).**
- [x] Diagnosis and fix (D-034): in the FP16 network, the INT8 Convs TensorRT doesn't fuse compute wrong values. Keeping the 17 unfused nodes in FP16 passes (−0.27, `parity__segformer_b0__trt__int8fp16_unfused__…__943d788.json`). A first fix with only the 8 depthwise convs wasn't enough.
- [x] Found and fixed on the way: per-layer parity read NaN/Inf tensors as perfect matches (`PARITY_LAYER_NONFINITE`). The one report written before the fix was replaced by a corrected rerun.
- [x] `sens_val100` on TensorRT for FP32, FP16, C3 INT8 and INT8+FP16 (`eval__<model>__trt__<label>__sens_val100__*.json`)
- [x] TensorRT sensitivity, both models (`sensitivity__<model>__trt__int8_fp16__sens_val100__{204bef4,8baee58}.json`)
- [x] Placement pair (ours vs ModelOpt): moved to session 2, because ModelOpt's calibration was still running on the Mac
- [x] Session 2 start: delete `parity__segformer_b0__trt__int8fp16_none__debug__…__layers__204bef4.json` on the instance (written before the NaN fix; the corrected rerun is `…__9cb5e07.json`), or the next sync brings it back

GPU session 2 (2026-10-17, 13:36–17:13 UTC, 3.62 GPU-hours; `scripts/c5_runs.sh` and `scripts/c5_gpu.py`, in this order):
- [x] Phases written and checked against the session-1 reports on the Mac: `clean`, `gpu build`/`sub` (placement pair), `trtexec`, `finalists`, `perf`, `parity`, `eval`, `final`, `profile`, `nsys`. `edgeinfer eval --subset` scores the held-out images and `sens_val100` in the same full-val pass, so no image is inferred twice per engine.
- [x] `clean`, `push` (placement pair), `smoke` (60/60)
- [x] Placement pair: build, `sens_val100`. SegFormer's pair is reported on ORT only (D-036): both engines leave the 17 D-034 nodes unfused, and ours scores 0.10 mIoU on TensorRT.
- [x] trtexec at 990 MHz for every engine with a `sens_val100` score (Pareto x axis); Pareto plots `docs/figures/pareto__<model>.png` (`pareto__<model>__dd84833.json`)
- [x] Finalists: within the subset target, on the Pareto front, the 3 fastest (D-035; `finalists__<model>__6efac63.json`)
- [x] Probe for finalists and FP16; golden parity and full val (with held-out) for finalists, FP16 and FP32
- [x] Final engine per model: the gates and golden parity, then the D-033 rule (D-035, `final__<model>__acaa81b.json`)
- [x] Per-layer profile of FP16 and the best INT8 engine (`trtexec__<model>__<label>__c5profile__990MHz__ab77361.json`); nsys trace of SegFormer's best INT8 engine (`nsys__segformer_b0__int8_fp16__1bd25bf.json`)
- Found during session 2: session 1's lookup of `build__segformer_b0__int8fp16_none__*.json` picked up the newer debug build, so SegFormer's full-INT8+FP16 `sens_val100` score came from the debug engine. Lookups now skip `__debug__` reports and match the engine id; the real engine was re-scored (mIoU 0.05, broken by D-034 as well), and no conclusion changes.

### C5 session 2 results — TensorRT 11.3 on the T4, full val (500 images), 512×1024
Finalists and FP16. Change is in mIoU points against the FP32 engine on the same images: full val 70.577 / 67.086, held out (`val_minus_sens100`, 400 images) 70.203 / 66.507. Latency is the probe (20 warmup + 3 × 400 frames at the 990 MHz lock).
- Reports: `final__<model>__acaa81b.json`, which lists each row's sources.
  - Probe: `perf__<model>__trt__<label>__c5__6efac63.json`.
  - Full val: `eval__<model>__trt__<label>__eval_val__c5__512x1024__ad4bfcf.json`.
  - Golden parity: `parity__…`.

| Model | Engine | Full val change | Held-out change | `infer` p99 (≤ 33.3 ms) | compute p50 | GPU memory | Engine file |
|---|---|---|---|---|---|---|---|
| SegFormer-B0 | **FP16 (final)** | +0.006 | +0.007 | 9.548 ms | 6.004 ms | 299 MiB | 15.7 MiB |
| SegFormer-B0 | INT8+FP16, `s0.patch_embed` in FP16 (best INT8) | −0.103 | +0.005 | 9.349 ms | 5.817 ms | 269 MiB | 8.4 MiB |
| SegFormer-B0 | INT8+FP16, `s2.attn_matmul` in FP16 | −0.153 | −0.099 | 9.705 ms | 6.156 ms | 269 MiB | 8.6 MiB |
| SegFormer-B0 | INT8+FP16, `s2.mlp` in FP16 | −0.066 | +0.011 | 9.711 ms | 6.173 ms | 267 MiB | 8.5 MiB |
| DeepLabV3+ MBv2 | **INT8+FP16, `bb.stem_16` in FP16 (final)** | −0.328 | −0.428 | 6.547 ms | 3.078 ms | 179 MiB | 5.8 MiB |
| DeepLabV3+ MBv2 | INT8+FP32, `bb.stem_16` in FP32 (C3 deployed) | −0.211 | −0.315 | 7.057 ms | 3.579 ms | 203 MiB | 5.9 MiB |
| DeepLabV3+ MBv2 | FP16 | +0.013 | +0.015 | 8.486 ms | 4.986 ms | 193 MiB | 10.4 MiB |

- **Final engines** (D-033, every row passes the perf gates and golden parity):
  - **SegFormer: FP16.** The best INT8 engine's `infer` p99 is 2.1% below FP16's, inside the 3.5% tie band, and a tie goes to fewer quantized nodes. INT8 halves the engine file and saves 30 MiB of device memory, but neither is a budget constraint.
  - **DeepLab: INT8+FP16 with the stem in FP16.** `infer` p99 is 23% below FP16, and compute 38% below. The cost is 0.33 points on full val and 0.43 on the held-out images, within the 1.0 budget.
- **SegFormer's INT8 engines are within 0.16 points of FP32 on full val.** That fits session 1: full INT8 was already within the subset target.
- **SegFormer FP16's `infer` p99 is 9.548 ms here against 11.82 ms in C4**, while p50 (9.42 vs 9.40 ms) and compute p50 (6.00 vs 6.07 ms) barely move. The p99 difference is host jitter in the Python `infer` span (D-028), which is why the tie band exists.

**Why INT8 is far below 2× FP16** (C5 task 5).
- Sources:
  - trtexec per-layer profiles at 990 MHz: `trtexec__<model>__<label>__c5profile__990MHz__ab77361.json`.
  - Layer counts: `build__<model>__<label>__*.json`.
  - SegFormer INT8 nsys trace: `nsys__segformer_b0__int8_fp16__1bd25bf.json`, against C4's FP16 trace.
- How to read the times: they are sums of per-layer medians. SegFormer's engines run layers on 2–3 auxiliary streams in parallel, so its sums exceed the wall-clock compute time.
- "Matrix layers" are TensorRT's `gemm`, `CaskConvolution`, `CaskGemmConvolution` and `fusion` layer types: the convolutions and matrix multiplies, which are the only ops our placement quantizes.

| Per-layer time (ms) | SegFormer FP16 | SegFormer best INT8 | DeepLab FP16 | DeepLab final INT8 |
|---|---|---|---|---|
| Matrix layers | 3.077 (41%) | 2.504 (36%) | 4.776 (93%) | 2.802 (87%) |
| `kgen` (TensorRT-generated kernels) | 3.254 (43%) | 3.267 (47%) | — | — |
| Reformat | 0.984 (13%) | 0.874 (13%) | 0.239 (5%) | 0.321 (10%) |
| Resize, pooling | 0.254 | 0.254 | 0.107 | 0.101 |
| Sum of layers | 7.569 | 6.898 (1.10×) | 5.123 | 3.224 (1.59×) |
| trtexec compute p50 | 5.988 | 5.679 (1.05×) | 4.967 | 3.067 (1.62×) |

- **SegFormer: most of the time is in layers INT8 doesn't touch.**
  - The matrix layers are 41% of FP16 time and get 1.23× faster.
  - TensorRT's generated `kgen` kernels are 43% and don't change (3.254 → 3.267 ms). The INT8+FP16 build with the D-034 exclusions has 86 of them (`build__segformer_b0__int8fp16_unfused__943d788.json`). Which ONNX nodes they cover isn't broken down here; the engine layer files stayed on the instance. By elimination, they are the non-matrix ops: LayerNorm, softmax, activations, adds.
  - Reformats stay at 13%.
  - Amdahl's law sets the ceiling. Even if the matrix layers ran at 2× and nothing else changed, the layer sum would only drop to 6.03 ms (1.26×).
  - The trace agrees with the profile. GPU kernel time per frame is 5.73 ms against 6.09 ms for FP16 in C4 (−6%).
    - The largest single kernel in the INT8 engine is FP16: the 3×3 depthwise conv kernel `sm50_xmma_fprop_conv2d_c1_k1_f16…`, 8 calls per frame, 0.455 ms (7.9%). These are the depthwise convs D-034 keeps in FP16.
    - Third is a cuTENSOR permutation that reads FP16 and writes INT8: a layout change plus quantization, 3 per frame, 0.343 ms (6.0%).
  - So for SegFormer-B0, INT8 doesn't pay. Its float ops cost as much time as its matrix work, its matrices are small (channel widths 32–256), and the layers it can't fuse stay in FP16.
- **DeepLab: almost everything is convolution, and INT8 still gets 1.62×, not 2×.**
  - The convolutions are 93% of FP16 time and get 1.70× faster.
  - Reformat time rises 34% (0.239 → 0.321 ms) with the precision boundaries: 7 reformat layers, 3 of them FP16→INT8 at the boundaries where FP16 layers feed INT8 ones.
  - The layers left in FP16 (stem, resize, pooling) take 0.338 ms, against 2.565 ms in INT8 layers.
  - Why the convolutions themselves stop at 1.70× isn't measured per layer here. MobileNetV2 is mostly depthwise and 1×1 convolutions, which move a lot of data for little arithmetic, so memory bandwidth rather than INT8 math likely limits them. Per-layer times would show it; they need the layer files from the instance.

**Placement comparison: ours vs ModelOpt's** (D-015, D-036). Both are min/max-calibrated INT8+FP16 with no exclusions. Change on `sens_val100` against FP32 (66.796 / 69.155).

| | DeepLab, TensorRT | DeepLab, ORT CPU | SegFormer, ORT CPU |
|---|---|---|---|
| Our placement (Conv/MatMul only) | −4.483 | −4.433 | −0.334 |
| ModelOpt placement (also Add, Resize, ReduceMean; on SegFormer also LayerNorm, Transpose, Reshape) | −4.239 | −4.029 | −0.890 |

- Reports:
  - TensorRT: `eval__deeplabv3plus_mbv2__trt__int8fp16_minmax_{ort,modelopt}__sens_val100__512x1024__de889fd.json` and the matching `trtexec__…__c5__990MHz__de889fd.json`.
  - ORT: `eval__<model>__ort_cpu__int8_fp16__minmax_{ort,modelopt}__sens_val100__512x1024__3f96e51.json`.
- **DeepLab: ModelOpt's placement is slightly better on both axes.**
  - It is 0.24 points more accurate on TensorRT (0.40 on ORT).
  - Its trtexec compute p50 is 3.8% lower: 2.819 vs 2.930 ms. That is at the edge of build-to-build variation (3.4%, D-029).
  - Its engine has no FP16 layers and 4 reformats; ours has 2 FP16 layers and 6 reformats. Quantizing the residual Adds keeps the data in INT8 between convolutions.
- **SegFormer: ModelOpt's placement costs 0.56 points more** on ORT. Quantizing LayerNorm and the other non-matrix ops hurts the transformer, which is the contrast between the two architectures this project is built to show.
- On TensorRT, both SegFormer engines are broken by D-034 (mIoU 0.10 and 28.40), which says nothing about placement.
- **Calibration matters far more than placement.** For DeepLab on TensorRT, our placement loses 1.306 points with percentile calibration and 4.483 with min/max: a 3.2-point gap, against 0.24 points between placements.

**C5 exit criteria:**
- A final engine per model that meets the budgets. SegFormer: FP16, because INT8 ties. DeepLab: INT8+FP16, stem in FP16.
- Pareto plots generated from the reports by `scripts/pareto.py`.
- TensorRT vs ORT ranking compared (session 1 results).
- INT8 < 2× explained with profile numbers (above).
- The implicit-INT8 removal recorded (D-031).
- CI green on `f1e6212` (run 39052865459).

The results tables for the README and `docs/results.md` come from `scripts/make_results.py` in C7.

**Teach-back:** model answers in `private/teach-back-C5.md`.

**Sign-off:** Anuj, 2026-10-17

### C5 session 1 results — TensorRT 11.3 on the T4, sens_val100 (100 val images), 512×1024
Change is in mIoU points against the FP32 engine on the same images. The FP32 engines match ORT FP32 (SegFormer 69.155, DeepLab 66.796).

| Engine | SegFormer-B0 | DeepLabV3+ MBv2 |
|---|---|---|
| FP16 | +0.006 | +0.011 |
| INT8+FP32, C3 model as is, full INT8 | −0.304 | −1.123 |
| INT8+FP32, C3 deployed exclusions | (= full) | −0.032 (`bb.stem_16` in float) |
| INT8+FP16, full INT8 | −0.291 (17 unfused nodes in FP16, D-034) | −1.306 |
| INT8+FP16, `bb.stem_16` in FP16 | — | −0.174 |

**TensorRT vs ORT sensitivity** (top 5 ORT groups, leave-one-out; target ≤ 0.3 on this subset):
- DeepLab: full INT8 −1.306 (ORT −1.157). `bb.stem_16` recovers 1.132 (ORT 1.027) and the first greedy step meets the target (−0.174), the same suggestion as ORT. Next, `bb.s24` recovers 0.308 (ORT 0.174); `head`, `bb.s64` and `bb.s32` are 0.09–0.16. Noise floor from a cold rebuild (different plan hash): 0.000.
- SegFormer: full INT8 −0.291 (ORT −0.282), within the target: `SENS_FULL_INT8_WITHIN_BUDGET`, as on ORT. `s2.mlp` ranks first on both runtimes (0.128 vs 0.151). The other four are 0.02–0.08. Noise floor 0.005.
- The rankings agree where effects are above the noise: one group for DeepLab, none for SegFormer. Below ~0.15 points the order isn't meaningful (D-017). SegFormer's TensorRT configurations differ from ORT's in the 17 nodes kept in FP16 (D-034).

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
