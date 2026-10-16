#!/usr/bin/env bash
# The C5 runs behind the reports in docs/progress.md, by phase.
#
# On the Mac (from a clean checkout; each report records its git SHA):
#   scripts/c5_runs.sh precheck   INT8+FP16 models of the C3 configurations (full INT8 and
#                                 the deployed exclusions): quantize, validate, ORT CPU
#                                 eval on sens_val100 and golden parity (D-031)
set -euo pipefail
cd "$(dirname "$0")/.."

# MODELS="deeplabv3plus_mbv2" runs a phase for a subset of the models.
read -ra MODELS <<< "${MODELS:-segformer_b0 deeplabv3plus_mbv2}"
QCFG=int8_percentile  # C3's calibration for both models (D-019)

fp32() { ls artifacts/onnx/"$1"/"$1"__fp32__*.onnx; }
sha() { git rev-parse --short HEAD; }

# Run a gate command: exit 1 means "checks failed, report written" and is a result to
# read, not a script error; any other non-zero exit stops the run.
gated() {
  local rc=0
  "$@" || rc=$?
  if [[ $rc -eq 1 ]]; then echo "GATE FAILED (report written): $*"; elif [[ $rc -ne 0 ]]; then exit "$rc"; fi
}

# quant MODEL TAG [quantize args...]: quantize with QCFG (instant when the content-addressed
# artifact exists), write the report, print the artifact path from it.
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

phase=${1:?usage: scripts/c5_runs.sh precheck}
case "$phase" in
  precheck) precheck ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
