"""Convert an INT8 Q/DQ model's float parts to FP16 in this isolated environment (D-031).

Called by `edgeinfer quantize --float-dtype fp16` through a subprocess:
    uv run --frozen --project tools/modelopt python tools/modelopt/qdq_fp16.py '<json>'
The JSON gives the input and output paths and `keep_io_types`. `convert_to_f16` is the step
ModelOpt's own quantizer runs after inserting Q/DQ: everything outside the Q/DQ pairs (and
the Q/DQ scales) becomes FP16; INT8 weights, zero points and placement are unchanged.
ModelOpt logs go to stderr; the last stdout line is a JSON summary (tool versions, seconds).
"""

import json
import sys
import time
from importlib import metadata

import onnx
from modelopt.onnx.autocast.convert import convert_to_f16


def main() -> int:
    args = json.loads(sys.argv[1])
    t0 = time.perf_counter()
    model = convert_to_f16(
        onnx.load(args["onnx_path"]),
        low_precision_type="fp16",
        keep_io_types=args["keep_io_types"],
    )
    onnx.save(model, args["output_path"])
    versions = {
        name: metadata.version(name) for name in ("nvidia-modelopt", "onnx", "onnxruntime", "numpy")
    }
    print(json.dumps({"versions": versions, "seconds": round(time.perf_counter() - t0, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
