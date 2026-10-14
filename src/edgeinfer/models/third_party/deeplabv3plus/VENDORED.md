# Vendored: VainF/DeepLabV3Plus-Pytorch

- Source: https://github.com/VainF/DeepLabV3Plus-Pytorch
- Commit: `4e1087de98bc49d55b9239ae92810ef7368660db` (2022-10-11)
- License: MIT (`LICENSE`, copied unmodified)

| File here | Upstream path | Changes |
|---|---|---|
| `_deeplab.py` | `network/_deeplab.py` | none |
| `utils.py` | `network/utils.py` | none |
| `mobilenetv2.py` | `network/backbone/mobilenetv2.py` | replaced a bare `try/except` torchvision import with `from torch.hub import load_state_dict_from_url` |

Why vendored: the repo isn't pip-installable, and only these three files are needed for
DeepLabV3+ MobileNetV2. The model is assembled in `edgeinfer/models/deeplabv3plus.py`,
which mirrors upstream `network/modeling.py::_segm_mobilenet` but skips the final
in-graph upsample (decision D-003). Excluded from ruff.
