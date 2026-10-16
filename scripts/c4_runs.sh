#!/usr/bin/env bash
# C4b runs on the T4, phase by phase, from the Mac. The checklist for the GPU sessions:
#
#   scripts/c4_runs.sh push     ONNX models (FP32 + FP16) and the tiny test fixture
#   scripts/c4_runs.sh push-data  golden outputs and Cityscapes val (perf frames, parity)
#   scripts/c4_runs.sh smoke    GPU, driver, container, TensorRT/CUDA imports, gpu tests
#   scripts/c4_runs.sh ops      configs/supported_ops.yaml vs the real ONNX parser
#   scripts/c4_runs.sh build    FP32 and FP16 engines for both models
#   scripts/c4_runs.sh perf     performance probe on every engine
#   scripts/c4_runs.sh trtexec  trtexec cross-check of compute latency
#   scripts/c4_runs.sh parity   golden parity for every engine
#   scripts/c4_runs.sh eval     full-val mIoU for every engine (500 images, ~10 min each)
#   scripts/c4_runs.sh nsys     Nsight Systems trace of the end-to-end path (one model)
#
# --print shows the commands without running them. Start the instance with gpu-up.sh
# first and stop it with gpu-down.sh after. Reports come back to artifacts/reports/.
set -euo pipefail
cd "$(dirname "$0")/.."

PRINT=0
if [ "${1:-}" = "--print" ]; then
    PRINT=1
    shift
fi
phase=${1:?usage: scripts/c4_runs.sh [--print] push|push-data|smoke|ops|build|perf|trtexec|parity|eval|nsys}
MODELS=(segformer_b0 deeplabv3plus_mbv2)

run() {  # run <remote.sh args...>
    if [ "$PRINT" -eq 1 ]; then
        printf 'scripts/remote.sh'
        printf ' %q' "$@"
        printf '\n'
    else
        scripts/remote.sh "$@"
    fi
}

# Newest deployable engine of a model and precision, resolved on the instance. Debug
# engines (marked outputs, no fusion) are never picked: they aren't the deployed engine.
engine() {  # engine <model> <precision>
    echo "\$(ls -t artifacts/engines/$1/$1__$2__*.engine | grep -v __debug__ | head -1)"
}
perf_report() {  # perf_report <model> <precision>: newest full perf report on TensorRT
    echo "\$(ls -t artifacts/reports/perf__$1__trt__$2__*.json | grep -v __nsys | head -1)"
}
onnx() {  # onnx <model> <precision>
    echo "\$(ls -t artifacts/onnx/$1/$1__$2__*.onnx | head -1)"
}

case "$phase" in
    push)
        # The tiny fixture is exported here: the container has no PyTorch (D-020).
        [ "$PRINT" -eq 1 ] || uv run edgeinfer export --model tests/fixtures/tiny_fixture.yaml >/dev/null
        files=(artifacts/onnx/tiny_fixture)
        for m in "${MODELS[@]}"; do
            for p in fp32 fp16; do
                f=$(ls -t artifacts/onnx/$m/${m}__${p}__*.onnx | head -1)
                files+=("$f" "${f%.onnx}.meta.json")
            done
        done
        run push-artifacts "${files[@]}"
        ;;
    push-data)
        run push-artifacts artifacts/golden
        run push-data
        ;;
    smoke)
        run host "nvidia-smi && df -h / && docker --version"
        run run "nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.sm,clocks.max.sm,temperature.gpu,power.limit --format=csv \
&& python -c 'import tensorrt, cuda.bindings; print(\"tensorrt\", tensorrt.__version__, \"cuda-bindings\", cuda.bindings.__version__)' \
&& command -v trtexec nsys && pytest -q tests/test_trt.py tests/test_trt_meta.py tests/test_perf.py"
        ;;
    ops)
        run run "python scripts/check_supported_ops.py"
        ;;
    build)
        for m in "${MODELS[@]}"; do
            for p in fp32 fp16; do
                run run "edgeinfer build --model $m --artifact $(onnx "$m" "$p")"
            done
        done
        ;;
    perf)
        for m in "${MODELS[@]}"; do
            for p in fp32 fp16; do
                run run "edgeinfer perf --model $m --backend trt --artifact $(engine "$m" "$p")"
            done
        done
        ;;
    trtexec)
        # Compute-only latency from NVIDIA's tool, to cross-check the probe's compute span.
        # The wrapper records trtexec's flags and writes a report with the SHA and environment.
        for m in "${MODELS[@]}"; do
            for p in fp32 fp16; do
                run run "python scripts/trtexec_report.py --engine $(engine "$m" "$p") \
--perf-report $(perf_report "$m" "$p")"
            done
        done
        ;;
    parity)
        for m in "${MODELS[@]}"; do
            run run "edgeinfer parity --model $m --backend trt --artifact $(engine "$m" fp32) --inputs golden"
        done
        # FP16 parity needs FP16 tolerances, set in C4b from these measurements (D-024).
        ;;
    eval)
        # The default report name has no precision, so FP32 and FP16 would collide.
        for m in "${MODELS[@]}"; do
            for p in fp32 fp16; do
                run run "edgeinfer eval --model $m --backend trt --artifact $(engine "$m" "$p") \
--manifest eval_val --out artifacts/reports/eval__${m}__trt__${p}__eval_val__512x1024__\$(git rev-parse --short=7 HEAD).json"
            done
        done
        ;;
    nsys)
        m=segformer_b0
        run run "nsys profile --trace=cuda,nvtx,osrt --output=artifacts/reports/nsys__${m}__fp16 --force-overwrite=true \
edgeinfer perf --model $m --backend trt --artifact $(engine "$m" fp16) --warmup 20 --iterations 100 --repeats 1 \
--out artifacts/reports/perf__${m}__trt__fp16__nsys.json"
        ;;
    *)
        echo "unknown phase: $phase" >&2
        exit 2
        ;;
esac
