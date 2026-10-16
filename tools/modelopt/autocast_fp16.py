"""Run ModelOpt AutoCast in this isolated environment (D-024).

Called by `edgeinfer fp16` through a subprocess:
    uv run --frozen --project tools/modelopt python tools/modelopt/autocast_fp16.py '<json>'
The JSON gives the input and output paths and AutoCast's arguments. AutoCast logs go to
stderr; the last stdout line is a JSON summary (tool versions, seconds).
"""

import json
import sys
import time
from importlib import metadata

import onnx
from modelopt.onnx.autocast import convert_to_mixed_precision


def main() -> int:
    args = json.loads(sys.argv[1])
    t0 = time.perf_counter()
    model = convert_to_mixed_precision(
        onnx_path=args["onnx_path"],
        low_precision_type=args["low_precision_type"],
        nodes_to_exclude=args["nodes_to_exclude"] or None,
        op_types_to_exclude=args["op_types_to_exclude"] or None,
        data_max=args["data_max"],
        init_max=args["init_max"],
        keep_io_types=args["keep_io_types"],
        calibration_data=args["calibration_data"],
        providers=["cpu"],
    )
    onnx.save(model, args["output_path"])
    versions = {
        name: metadata.version(name) for name in ("nvidia-modelopt", "onnx", "onnxruntime", "numpy")
    }
    print(json.dumps({"versions": versions, "seconds": round(time.perf_counter() - t0, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
