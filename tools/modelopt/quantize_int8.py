"""ModelOpt's own INT8 ONNX quantization, for the placement comparison (D-015, D-031).

Called by `edgeinfer quantize --tool modelopt` through a subprocess:
    uv run --frozen --project tools/modelopt python tools/modelopt/quantize_int8.py '<json>'
The JSON gives the input and output paths, a directory of calibration NPZ files (one
preprocessed image each), the calibration method and whether to calibrate node by node
(lower memory). ModelOpt picks the Q/DQ placement; everything outside the Q/DQ pairs is
FP16 (`high_precision_dtype`), I/O stays float32. Logs go to stderr; the last stdout line
is a JSON summary (tool versions, seconds, peak memory).
"""

import json
import resource
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from modelopt.onnx.quantization import quantize


def main() -> int:
    args = json.loads(sys.argv[1])
    files = sorted(Path(args["calibration_data"]).glob("*.npz"))
    data = np.concatenate([np.load(f)[args["input_name"]] for f in files])
    t0 = time.perf_counter()
    quantize(
        args["onnx_path"],
        quantize_mode="int8",
        calibration_data={args["input_name"]: data},
        calibration_method=args["calibration_method"],
        calibration_eps=["cpu"],
        calibrate_per_node=args["calibrate_per_node"],
        high_precision_dtype="fp16",
        output_path=args["output_path"],
        log_level="WARNING",
    )
    versions = {
        name: metadata.version(name) for name in ("nvidia-modelopt", "onnx", "onnxruntime", "numpy")
    }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {
        "versions": versions,
        "seconds": round(time.perf_counter() - t0, 2),
        "n_images": len(files),
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
