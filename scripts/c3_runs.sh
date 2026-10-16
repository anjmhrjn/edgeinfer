#!/usr/bin/env bash
# The C3 runs behind the reports in docs/progress.md, by phase:
#   scripts/c3_runs.sh quantize   INT8 QDQ models, 3 calibration methods x 2 models
#   scripts/c3_runs.sh evals      FP32 subset baselines; INT8 eval (full val, sens_val100,
#                                 val_minus_sens100) and per-layer parity per method
#   scripts/c3_runs.sh calibsize  calibration-size study (needs METHOD_<model> set)
#   scripts/c3_runs.sh sens       sensitivity (needs METHOD_<model> and MARGIN set)
#   scripts/c3_runs.sh final      suggested exclusions end to end (needs METHOD_<model>)
# Run from a clean checkout: each report records the git SHA it was produced at.
set -euo pipefail
cd "$(dirname "$0")/.."

# MODELS="deeplabv3plus_mbv2" runs a phase for a subset of the models.
read -ra MODELS <<< "${MODELS:-segformer_b0 deeplabv3plus_mbv2}"
METHODS=(minmax entropy percentile)

fp32() { ls artifacts/onnx/"$1"/"$1"__fp32__*.onnx; }
sha() { git rev-parse --short HEAD; }
method() { local v="METHOD_$1"; [[ -n "${!v:-}" ]] || { echo "set $v" >&2; exit 2; }; echo "${!v}"; }

# Run a gate command: exit 1 means "checks failed, report written" and is a result to
# read, not a script error; any other non-zero exit stops the run.
gated() {
  local rc=0
  "$@" || rc=$?
  if [[ $rc -eq 1 ]]; then echo "GATE FAILED (report written): $*"; elif [[ $rc -ne 0 ]]; then exit "$rc"; fi
}

# quant MODEL QUANT_CONFIG TAG [--exclusions FILE]: quantize (instant when the content-
# addressed artifact exists), write the report, print the INT8 artifact path from it.
# Paths come from reports, not sidecars: identical configurations share one artifact.
quant() {
  local m=$1 qc=$2 tag=$3; shift 3
  local rep
  rep="artifacts/reports/quantize__${m}__$(basename "${qc%.yaml}")__${tag}__$(sha).json"
  uv run edgeinfer quantize --model "$m" --artifact "$(fp32 "$m")" --quant-config "$qc" "$@" \
    --out "$rep" >&2
  echo "artifacts/onnx/$m/$(uv run python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["metrics"]["file"])' "$rep")"
}

quantize() {
  for m in "${MODELS[@]}"; do for q in "${METHODS[@]}"; do quant "$m" "int8_$q" none; done; done
}

evals() {
  for m in "${MODELS[@]}"; do
    for man in sens_val100 val_minus_sens100; do
      uv run edgeinfer eval --model "$m" --backend ort_cpu --artifact "$(fp32 "$m")" --manifest "$man"
    done
    for q in "${METHODS[@]}"; do
      a=$(quant "$m" "int8_$q" none)
      for man in eval_val sens_val100 val_minus_sens100; do
        uv run edgeinfer eval --model "$m" --backend ort_cpu --artifact "$a" --manifest "$man" \
          --out "artifacts/reports/eval__${m}__ort_cpu__int8_${q}__${man}__512x1024__$(sha).json"
      done
      gated uv run edgeinfer parity --model "$m" --artifact "$a" --per-layer --ref-artifact "$(fp32 "$m")" \
        --out "artifacts/reports/parity__${m}__ort_cpu__int8_${q}__golden_val32__images__layers__$(sha).json"
    done
  done
}

# Calibration-size study: the chosen method on the first N images of calib_train200,
# scored on sens_val100 (N=200 is the full-calibration run from `evals`). The derived
# configs go to a gitignored directory; each quantize report embeds the full config.
calibsize() {
  mkdir -p artifacts/quant_configs
  for m in "${MODELS[@]}"; do
    q=$(method "$m")
    for n in 25 50 100; do
      cfg="artifacts/quant_configs/int8_${q}_calib${n}.yaml"
      sed -e "s/^name: int8_${q}$/name: int8_${q}_calib${n}/" -e "s/^  limit: null$/  limit: ${n}/" \
        "configs/quant/int8_${q}.yaml" > "$cfg"
      grep -q "^  limit: ${n}$" "$cfg" || { echo "limit not set in $cfg"; exit 2; }
      a=$(quant "$m" "$cfg" none)
      uv run edgeinfer eval --model "$m" --backend ort_cpu --artifact "$a" --manifest sens_val100 \
        --out "artifacts/reports/eval__${m}__ort_cpu__int8_${q}_calib${n}__sens_val100__512x1024__$(sha).json"
    done
  done
}

# Sensitivity on sens_val100; writes the suggested exclusion list.
sens() {
  : "${MARGIN:?set MARGIN (points)}"
  for m in "${MODELS[@]}"; do
    q=$(method "$m")
    gated uv run edgeinfer sensitivity --model "$m" --artifact "$(fp32 "$m")" --quant-config "int8_$q" \
      --margin "$MARGIN" --write-exclusions "configs/quant/exclusions/${m}__sens.yaml"
  done
}

# The suggested exclusions end to end: quantize -> validate -> eval -> per-layer parity.
final() {
  for m in "${MODELS[@]}"; do
    q=$(method "$m")
    a=$(quant "$m" "int8_$q" sens --exclusions "configs/quant/exclusions/${m}__sens.yaml")
    gated uv run edgeinfer validate --model "$m" --artifact "$a" \
      --out "artifacts/reports/validate__${m}__int8_${q}__sens__$(sha).json"
    for man in eval_val val_minus_sens100; do
      uv run edgeinfer eval --model "$m" --backend ort_cpu --artifact "$a" --manifest "$man" \
        --out "artifacts/reports/eval__${m}__ort_cpu__int8_${q}__sens__${man}__512x1024__$(sha).json"
    done
    gated uv run edgeinfer parity --model "$m" --artifact "$a" --per-layer --ref-artifact "$(fp32 "$m")" \
      --out "artifacts/reports/parity__${m}__ort_cpu__int8_${q}__sens__golden_val32__images__layers__$(sha).json"
  done
}

"$@"
