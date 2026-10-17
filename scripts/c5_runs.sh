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
#   scripts/c5_runs.sh candidates every subset of the top-5 ORT sensitivity groups as INT8 and
#                                 INT8+FP16 models, validated, listed in a report (D-032);
#                                 SegFormer keeps its depthwise convs in FP16 (D-034)
#   scripts/c5_runs.sh dwfp16     SegFormer full INT8+FP16 with the depthwise convs in FP16
#
# GPU sessions (start with gpu-up.sh, stop with gpu-down.sh; reports come back by themselves):
#   scripts/c5_runs.sh push       the ONNX models every GPU phase needs, with sidecars
#   scripts/c5_runs.sh smoke      container, TensorRT, GPU tests (incl. INT8 build and fusion)
#   scripts/c5_runs.sh gpu PHASE  scripts/c5_gpu.py PHASE in the container (build, sub, parity,
#                                 sens; see its docstring)
set -euo pipefail
cd "$(dirname "$0")/.."

# MODELS="deeplabv3plus_mbv2" runs a phase for a subset of the models.
read -ra MODELS <<< "${MODELS:-segformer_b0 deeplabv3plus_mbv2}"
QCFG=int8_percentile  # C3's calibration for both models (D-019)

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
    segformer_b0) echo configs/quant/exclusions/segformer_b0__dwconv_fp16.yaml ;;
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

phase=${1:?usage: scripts/c5_runs.sh precheck|placement|candidates|dwfp16|push|smoke|gpu PHASE}
case "$phase" in
  precheck) precheck ;;
  placement) placement ;;
  candidates) candidates ;;
  dwfp16) dwfp16 ;;
  push) push ;;
  smoke)
    scripts/remote.sh run "python -c 'import tensorrt; print(\"tensorrt\", tensorrt.__version__)' \
&& pytest -q tests/test_trt.py tests/test_trt_meta.py tests/test_trt_layers.py tests/test_perf.py -s" ;;
  gpu) scripts/remote.sh run "python scripts/c5_gpu.py ${2:?gpu PHASE}" ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
