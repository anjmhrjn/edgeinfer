#!/usr/bin/env bash
# The C5 runs behind the reports in docs/progress.md, by phase.
#
# On the Mac (from a clean checkout; each report records its git SHA):
#   scripts/c5_runs.sh precheck   INT8+FP16 models of the C3 configurations (full INT8 and
#                                 the deployed exclusions): quantize, validate, ORT CPU
#                                 eval on sens_val100 and golden parity (D-031)
#   scripts/c5_runs.sh placement  ModelOpt's Q/DQ placement vs ours, both min/max-calibrated on
#                                 calib_train200 and FP16 elsewhere (D-015), then validated.
#                                 Skips a model/tool whose report exists (ModelOpt's per-node
#                                 calibration takes over an hour per model)
#   scripts/c5_runs.sh placement-eval  the placement pair on ORT CPU, sens_val100: the SegFormer
#                                 comparison is reported on ORT only, because neither SegFormer
#                                 min/max model keeps the D-034 nodes in FP16 (D-036)
#   scripts/c5_runs.sh candidates every subset of the top-5 ORT sensitivity groups as INT8 and
#                                 INT8+FP16 models, validated, listed in a report (D-032);
#                                 SegFormer keeps the 17 nodes TensorRT doesn't fuse in FP16 (D-034)
#   scripts/c5_runs.sh pareto     latency vs accuracy plots from the session-2 reports (after
#                                 session 2; SegFormer without the placement pair, D-036)
#   scripts/c5_runs.sh dwfp16     SegFormer full INT8+FP16 with the depthwise convs in FP16
#                                 (superseded: not enough, D-034; kept for its reports)
#
# GPU sessions (start with gpu-up.sh, stop with gpu-down.sh; reports come back by themselves):
#   scripts/c5_runs.sh push       the ONNX models every GPU phase needs, with sidecars
#   scripts/c5_runs.sh clean      session 2, first: delete the per-layer report written before
#                                 the NaN fix on the instance (the next sync would bring it back)
#   scripts/c5_runs.sh smoke      container, TensorRT, GPU tests (incl. INT8 build and fusion)
#   scripts/c5_runs.sh gpu PHASE  scripts/c5_gpu.py PHASE in the container; see its docstring.
#                                 Session 1: build, sub, parity, sens. Session 2: build, sub
#                                 (placement pair), trtexec, finalists, perf, parity, eval,
#                                 final, profile, nsys. Latency phases lock the SM clock first
set -euo pipefail
cd "$(dirname "$0")/.."

# MODELS="deeplabv3plus_mbv2" runs a phase for a subset of the models.
read -ra MODELS <<< "${MODELS:-segformer_b0 deeplabv3plus_mbv2}"
QCFG=int8_percentile  # C3's calibration for both models (D-019)

# Clock locking is a host setting (the container has no privileges), as in c4_runs.sh. It
# lasts until a reboot; gpu-down.sh stops the instance, which resets it.
lock_clocks() {
  local mhz
  mhz=$(uv run --no-sync python -c \
    'from edgeinfer.perf.probe import load_perf_config; print(load_perf_config().gpu_clock_mhz)')
  scripts/remote.sh host "sudo nvidia-smi --lock-gpu-clocks=$mhz,$mhz && nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv"
}

# Written before the NaN fix (it reads non-finite layers as exact); the corrected rerun is
# ...__layers__9cb5e07.json. Deleted locally, still on the instance after session 1.
STALE_REPORT=parity__segformer_b0__trt__int8fp16_none__debug__golden_val32__golden__layers__204bef4.json

fp32() { ls artifacts/onnx/"$1"/"$1"__fp32__????????????.onnx; }  # <id[:12]>.onnx only
sha() { git rev-parse --short HEAD; }

# Run a gate command: exit 1 means "checks failed, report written" and is a result to
# read, not a script error; any other non-zero exit stops the run.
gated() {
  local rc=0
  "$@" || rc=$?
  if [[ $rc -eq 1 ]]; then echo "GATE FAILED (report written): $*"; elif [[ $rc -ne 0 ]]; then exit "$rc"; fi
}

# quant MODEL TAG [quantize args...]: quantize with QCFG (override: QCFG=... quant ...), to
# INT8+FP16 (instant when the content-addressed artifact exists), write the report, print
# the artifact path from it.
quant() {
  local m=$1 tag=$2; shift 2
  local rep="artifacts/reports/quantize__${m}__${QCFG}_fp16__${tag}__$(sha).json"
  uv run edgeinfer quantize --model "$m" --artifact "$(fp32 "$m")" --quant-config "$QCFG" \
    --float-dtype fp16 "$@" --out "$rep" >&2
  echo "artifacts/onnx/$m/$(uv run python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["metrics"]["file"])' "$rep")"
}

precheck() {
  for m in "${MODELS[@]}"; do
    for tag in none sens; do
      local args=()
      [[ $tag == sens ]] && args=(--exclusions "configs/quant/exclusions/${m}__sens.yaml")
      a=$(quant "$m" "$tag" "${args[@]+"${args[@]}"}")
      gated uv run edgeinfer validate --model "$m" --artifact "$a" \
        --out "artifacts/reports/validate__${m}__int8_fp16__${tag}__$(sha).json"
      uv run edgeinfer eval --model "$m" --backend ort_cpu --artifact "$a" --manifest sens_val100 \
        --out "artifacts/reports/eval__${m}__ort_cpu__int8_fp16__${tag}__sens_val100__512x1024__$(sha).json"
      gated uv run edgeinfer parity --model "$m" --artifact "$a" --inputs golden \
        --out "artifacts/reports/parity__${m}__ort_cpu__int8_fp16__${tag}__golden_val32__golden__$(sha).json"
    done
  done
}

placement() {
  for m in "${MODELS[@]}"; do
    for tool in ort modelopt; do
      if compgen -G "artifacts/reports/quantize__${m}__int8_minmax_fp16__${tool}__*.json" >/dev/null; then
        echo "exists: $m $tool"
        continue
      fi
      a=$(QCFG=int8_minmax quant "$m" "$tool" --tool "$tool")
      gated uv run edgeinfer validate --model "$m" --artifact "$a" \
        --out "artifacts/reports/validate__${m}__int8_fp16__minmax_${tool}__$(sha).json"
    done
  done
}

placement_eval() {
  for m in "${MODELS[@]}"; do
    for tool in ort modelopt; do
      local rep a
      rep=$(ls artifacts/reports/quantize__"${m}"__int8_minmax_fp16__"${tool}"__*.json)
      a="artifacts/onnx/$m/$(uv run python -c \
        'import json, sys; print(json.load(open(sys.argv[1]))["metrics"]["file"])' "$rep")"
      uv run edgeinfer eval --model "$m" --backend ort_cpu --artifact "$a" --manifest sens_val100 \
        --out "artifacts/reports/eval__${m}__ort_cpu__int8_fp16__minmax_${tool}__sens_val100__512x1024__$(sha).json"
    done
  done
}

# The C3 ORT sensitivity reports whose rankings C5 re-tests on TensorRT.
ort_ranking() {
  case $1 in
    segformer_b0) echo artifacts/reports/sensitivity__segformer_b0__int8_percentile__sens_val100__cb60f28.json ;;
    deeplabv3plus_mbv2) echo artifacts/reports/sensitivity__deeplabv3plus_mbv2__int8_percentile__sens_val100__da584fa.json ;;
  esac
}

# Layers kept in float in every C5 configuration of a model (D-034), if any.
base_exclusions() {
  case $1 in
    segformer_b0) echo configs/quant/exclusions/segformer_b0__unfused_fp16.yaml ;;
  esac
}

candidates() {
  for m in "${MODELS[@]}"; do
    local base args=() tag=""
    base=$(base_exclusions "$m")
    if [[ -n $base ]]; then args=(--base-exclusions "$base"); tag="__$(basename "${base%.yaml}")"; fi
    uv run edgeinfer candidates --model "$m" --artifact "$(fp32 "$m")" --quant-config "$QCFG" \
      --ranking "$(ort_ranking "$m")" --top 5 "${args[@]+"${args[@]}"}" \
      --out "artifacts/reports/candidates__${m}__${QCFG}__top5${tag}__$(sha).json"
  done
}

# SegFormer's full INT8+FP16 model with its depthwise convs in FP16 (D-034), for golden
# parity and the sens_val100 eval on TensorRT.
dwfp16() {
  local m=segformer_b0
  a=$(quant "$m" dwfp16 --exclusions "$(base_exclusions "$m")")
  gated uv run edgeinfer validate --model "$m" --artifact "$a" \
    --out "artifacts/reports/validate__${m}__int8_fp16__dwfp16__$(sha).json"
}

# pareto: every engine scored on sens_val100 on TensorRT, one point each. The sensitivity
# report brings our INT8+FP16 configurations; the rest come from their sub-phase evals.
pareto() {
  local m
  for m in "${MODELS[@]}"; do
    local ev args=() spec label kind
    ev() { ls -t artifacts/reports/eval__"$m"__trt__"$1"__sens_val100__512x1024__*.json | head -1; }
    for spec in FP32:baseline:fp32 FP16:baseline:fp16 "INT8+FP32 full":int8_fp32:int8_c3_none; do
      IFS=: read -r label kind tag <<< "$spec"
      args+=(--point "$label:$kind:$(ev "$tag")")
    done
    if [[ $m == deeplabv3plus_mbv2 ]]; then  # SegFormer's C3 sens model is its full model
      args+=(--point "INT8+FP32 stem float:int8_fp32:$(ev int8_c3_sens)"
             --point "ours min/max:minmax_ours:$(ev int8fp16_minmax_ort)"
             --point "ModelOpt min/max:modelopt:$(ev int8fp16_minmax_modelopt)")
    fi
    uv run --group plots python scripts/pareto.py --model "$m" \
      --sensitivity "$(ls -t artifacts/reports/sensitivity__"$m"__trt__int8_fp16__sens_val100__*.json | head -1)" \
      "${args[@]}"
  done
}

# Every ONNX file a GPU phase reads, from the reports that name it (with its sidecar).
push() {
  uv run edgeinfer export --model tests/fixtures/tiny_fixture.yaml >/dev/null  # no torch there
  local files=() f
  while IFS= read -r f; do files+=("$f"); done < <(uv run python - "${MODELS[@]}" <<'PY'
import json, sys
from pathlib import Path
R = Path("artifacts/reports")
for m in sys.argv[1:]:
    names = []
    for pattern in (f"candidates__{m}__*.json", f"quantize__{m}__int8_percentile__none__cb60f28.json",
                    f"quantize__{m}__int8_percentile__sens__da584fa.json",
                    f"quantize__{m}__int8_percentile_fp16__*.json", f"quantize__{m}__int8_minmax_fp16__*.json"):
        for rep in sorted(R.glob(pattern)):
            r = json.loads(rep.read_text())
            if r["tool"] == "candidates":
                names += [c[p]["file"] for c in r["metrics"]["configs"] for p in ("int8", "int8_fp16")]
            elif r["passed"]:
                names.append(r["metrics"]["file"])
    for n in sorted(set(names)):
        f = Path("artifacts/onnx") / m / n
        print(f)
        print(f.with_suffix(".meta.json"))
PY
)
  echo "pushing ${#files[@]} files"
  scripts/remote.sh push-artifacts artifacts/onnx/tiny_fixture "${files[@]}"
}

phase=${1:?usage: scripts/c5_runs.sh precheck|placement|placement-eval|candidates|pareto|dwfp16|push|clean|smoke|gpu PHASE}
case "$phase" in
  precheck) precheck ;;
  placement) placement ;;
  placement-eval) placement_eval ;;
  candidates) candidates ;;
  pareto) pareto ;;
  dwfp16) dwfp16 ;;
  push) push ;;
  clean)
    # shellcheck source=scripts/gpu-config.sh
    source scripts/gpu-config.sh
    scripts/remote.sh host "cd ~/$GPU_REMOTE_DIR/artifacts/reports && if [ -e $STALE_REPORT ]; then rm -v $STALE_REPORT; else echo 'already gone: $STALE_REPORT'; fi" ;;
  smoke)
    scripts/remote.sh run "python -c 'import tensorrt; print(\"tensorrt\", tensorrt.__version__)' \
&& pytest -q tests/test_trt.py tests/test_trt_meta.py tests/test_trt_layers.py tests/test_perf.py -s" ;;
  gpu)
    case ${2:?gpu PHASE} in trtexec|perf|profile|nsys) lock_clocks ;; esac
    scripts/remote.sh run "MODELS='${MODELS[*]}' python scripts/c5_gpu.py $2" ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
