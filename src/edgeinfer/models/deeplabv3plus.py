"""DeepLabV3+ MobileNetV2 from the vendored VainF code, without the in-graph upsample.

Mirrors upstream `network/modeling.py::_segm_mobilenet`. Upstream `forward()`
bilinearly upsamples logits to the input size; we return the stride-4 logits and
leave upsampling to postprocess (D-003), the same as SegFormer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from edgeinfer.models.third_party.deeplabv3plus._deeplab import DeepLabHeadV3Plus, DeepLabV3
from edgeinfer.models.third_party.deeplabv3plus.mobilenetv2 import mobilenet_v2
from edgeinfer.models.third_party.deeplabv3plus.utils import IntermediateLayerGetter

# The checkpoint stores best_score as a NumPy scalar. Allowlist exactly that so
# torch.load can stay weights_only=True (no arbitrary pickle code). NumPy 2 moved
# numpy.core -> numpy._core, so the pickled name is mapped explicitly.
_CHECKPOINT_SAFE_GLOBALS = [
    (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
    np.dtype,
    type(np.dtype(np.float64)),
]


def _build_network(num_classes: int, backbone_output_stride: int) -> DeepLabV3:
    aspp_dilate = [12, 24, 36] if backbone_output_stride == 8 else [6, 12, 18]
    backbone = mobilenet_v2(pretrained=False, output_stride=backbone_output_stride)
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None
    backbone = IntermediateLayerGetter(
        backbone, return_layers={"high_level_features": "out", "low_level_features": "low_level"}
    )
    classifier = DeepLabHeadV3Plus(320, 24, num_classes, aspp_dilate)
    return DeepLabV3(backbone, classifier)


class DeepLabV3PlusLogits(nn.Module):
    """NCHW image -> stride-4 logits (no final upsample)."""

    def __init__(self, num_classes: int, backbone_output_stride: int) -> None:
        super().__init__()
        self.net = _build_network(num_classes, backbone_output_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.classifier(self.net.backbone(x))

    def load_checkpoint(self, path: Path) -> None:
        with torch.serialization.safe_globals(_CHECKPOINT_SAFE_GLOBALS):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(ckpt["model_state"], strict=True)
