"""SegFormer via Hugging Face transformers, wrapped to return a plain logits tensor."""

from __future__ import annotations

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation


class SegformerLogits(nn.Module):
    """NCHW image -> stride-4 logits as a plain tensor (`return_dict=False`; ONNX-friendly)."""

    def __init__(self, hf_model: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x, return_dict=False)[0]


def load_segformer(repo: str, revision: str) -> SegformerLogits:
    return SegformerLogits(
        SegformerForSemanticSegmentation.from_pretrained(repo, revision=revision)
    )
