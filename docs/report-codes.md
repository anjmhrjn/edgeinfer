# Report finding codes

Stable enum of `findings[].code` values used in JSON reports. Codes are never renamed or reused; deprecated codes stay listed.

| Code | Tool | Severity | Meaning |
|---|---|---|---|
| `EVAL_CLASS_ABSENT` | eval | warn | A class has no pixels in the labels or the predictions, so it's excluded from mIoU. Expected only on small subsets. |
| `VALIDATE_CHECKER_FAILED` | validate | error | `onnx.checker` rejects the model (malformed graph, unsorted nodes, missing inputs). |
| `VALIDATE_SHAPE_INFERENCE_FAILED` | validate | error | Strict ONNX shape inference fails, e.g. a declared shape contradicts the computed one. |
| `VALIDATE_DYNAMIC_DIM` | validate | warn | A graph I/O or intermediate tensor has a non-static dimension; engines are built for one fixed shape (D-013). |
| `VALIDATE_OPSET_UNSUPPORTED` | validate | error | The model's opset is outside the range in `configs/supported_ops.yaml`. |
| `VALIDATE_UNSUPPORTED_OP` | validate | error | An op isn't supported by the TensorRT ONNX parser (`configs/supported_ops.yaml`), or is in a custom domain that needs a plugin. |
| `VALIDATE_OP_RESTRICTION` | validate | error | A supported op violates a parser restriction checked statically: Resize mode/coordinate mode/antialias/aspect policy, Reduce* `axes` not an initializer, non-zero Q/DQ zero point, LayerNormalization extra outputs used. |
| `VALIDATE_IO_MISMATCH` | validate | error | Input/output names, dtypes or shapes differ from what the model config implies, or the embedded contract's model/classes/stride differ from the config. |
| `VALIDATE_INITIALIZER_NONFINITE` | validate | error | A weight tensor contains NaN or Inf. |
| `VALIDATE_INITIALIZER_FP16_RANGE` | validate | warn | A weight's magnitude exceeds 65504, the FP16 maximum; it would overflow in an FP16 engine. |
| `VALIDATE_METADATA_MISSING` | validate | error | The model has no embedded I/O contract (`edgeinfer.contract`), so config consistency can't be checked. |
| `VALIDATE_PREPROC_MISMATCH` | validate | error | The config's input/preprocess/postprocess spec differs from the one embedded at export (e.g. `input.channel_order` RGB vs BGR). `where` names the field. |
| `PARITY_INPUT_MISMATCH` | parity | error | In `images` mode, the input preprocessed with the given config differs from the golden input. The message diagnoses the likely cause from per-channel correlation: reversed channels (`input.channel_order`), a per-channel affine difference (`scale`/`mean`/`std`), or spatial differences (resize, wrong image). |
| `PARITY_OUTPUT_NONFINITE` | parity | error | Candidate logits contain NaN or Inf. |
| `PARITY_OUTPUT_DRIFT` | parity | error | Logits max abs error or cosine similarity vs golden is outside tolerance (`configs/tolerances.yaml`). `where` is the worst image. |
| `PARITY_ARGMAX_DISAGREEMENT` | parity | error | The fraction of native-stride pixels whose argmax class matches golden is below tolerance. |
| `PARITY_MIOU_DELTA` | parity | error | mIoU on the golden images (vs labels) changed by more than the tolerance in points. |
| `PARITY_MIOU_SKIPPED` | parity | info | No labels were available (dataset root not given), so the mIoU delta wasn't computed. |
| `PARITY_GOLDEN_UNVERIFIED` | parity | info | Golden files weren't checked against committed hashes (no hash file, or golden files outside the repo, e.g. a test fixture). |
| `PARITY_LAYER_DRIFT` | parity | error | Per-layer mode: at least one intermediate tensor exceeds the per-layer tolerance. `where` is the first such tensor in graph order. |
| `VALIDATE_QDQ_DTYPE` | validate | error | A Quantize/DequantizeLinear uses a type other than signed INT8 (e.g. UINT8, or the ONNX default UINT8 when no zero point is given). TensorRT explicit quantization needs signed INT8 (D-016). `where` is the first such node. |
| `VALIDATE_QDQ_BIAS` | validate | warn | A Conv/Gemm bias goes through an INT32 DequantizeLinear. Our TensorRT format keeps bias in float (D-016). |
| `VALIDATE_QDQ_WEIGHT_PER_TENSOR` | validate | warn | A Conv/MatMul/Gemm weight is quantized with one scale for the whole tensor instead of per output channel. |
| `QUANT_CALIB_ON_EVAL_SPLIT` | quantize | error | The calibration manifest contains images outside the train split. Calibrating on val would tune the quantizer on the evaluation images. |
| `QUANT_CALIB_TOO_SMALL` | quantize | error | `calibration.limit` asks for more images than the calibration manifest has. |
| `QUANT_CALIB_MISMATCH` | quantize | error | The calibration cache was computed on a different FP32 model. |
| `QUANT_INPUT_NOT_FP32` | quantize | error | The input artifact isn't an FP32 export (e.g. an already-quantized model). |
| `QUANT_GROUP_UNMATCHED` | quantize | error | A quantizable node matches no layer-group rule, a rule matches no node, or the model lacks the exporter's name-scope metadata (`configs/quant/groups/`). |
| `QUANT_UNKNOWN_EXCLUSION` | quantize | error | An exclusion list names a group or node the model doesn't have. A typo must never silently quantize a layer meant to stay in float. |
| `QUANT_PLACEMENT_MISMATCH` | quantize | error | After quantizing, a node meant to be INT8 reads a float input, or an excluded node reads a dequantized one. |
