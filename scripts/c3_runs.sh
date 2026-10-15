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

MODELS=(segformer_b0 deeplabv3plus_mbv2)
METHODS=(minmax entropy percentile)

fp32() { ls artifacts/onnx/"$1"/"$1"__fp32__*.onnx; }

# Run a gate command: exit 1 means "checks failed, report written" and is a result to
# read, not a script error; any other non-zero exit stops the run.
gated() {
  local rc=0
  "$@" || rc=$?
  if [[ $rc -eq 1 ]]; then echo "GATE FAILED (report written): $*"; elif [[ $rc -ne 0 ]]; then exit "$rc"; fi
}

# INT8 artifact of model $1 for quant config $2 and exclusion list $3 (from the sidecars).
int8() {
  uv run python - "$1" "$2" "$3" <<'EOF'
import json, sys
from pathlib import Path
model, qname, exc = sys.argv[1:]
hits = [p for p in Path("artifacts/onnx", model).glob(f"{model}__int8__*.meta.json")
        if (m := json.loads(p.read_text()))["quant_config"]["name"] == qname
        and m["exclusions"]["name"] == exc]
if len(hits) != 1:
    sys.exit(f"expected one {qname}/{exc} artifact for {model}, found {len(hits)}")
print(hits[0].with_name(json.loads(hits[0].read_text())["file"]))
EOF
}

quantize() {
  for m in "${MODELS[@]}"; do for q in "${METHODS[@]}"; do
    uv run edgeinfer quantize --model "$m" --artifact "$(fp32 "$m")" --quant-config "int8_$q"
  done; done
}

evals() {
  for m in "${MODELS[@]}"; do
    for man in sens_val100 val_minus_sens100; do
      uv run edgeinfer eval --model "$m" --backend ort_cpu --artifact "$(fp32 "$m")" --manifest "$man"
    done
    for q in "${METHODS[@]}"; do
      a=$(int8 "$m" "int8_$q" none)
      for man in eval_val sens_val100 val_minus_sens100; do
        uv run edgeinfer eval --model "$m" --backend ort_cpu --artifact "$a" --manifest "$man" \
          --out "artifacts/reports/eval__${m}__ort_cpu__int8_${q}__${man}__512x1024__$(git rev-parse --short HEAD).json"
      done
      gated uv run edgeinfer parity --model "$m" --artifact "$a" --per-layer --ref-artifact "$(fp32 "$m")" \
        --out "artifacts/reports/parity__${m}__ort_cpu__int8_${q}__golden_val32__images__layers__$(git rev-parse --short HEAD).json"
    done
  done
}

"$@"
