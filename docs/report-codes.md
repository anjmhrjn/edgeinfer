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
