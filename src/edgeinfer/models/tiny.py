"""Tiny seeded test model: exercises the export -> validate -> ORT -> parity path in CI.

No checkpoint and no dataset. It contains the op types the real models depend on:
conv + batchnorm + relu (CNN), LayerNorm + softmax attention (transformer), and a
bilinear upsample + concat (as in both decode heads). Output: stride-4 logits.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv_bn_relu(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


class TinyAttention(nn.Module):
    """Pre-norm single-head self-attention over spatial tokens, with a residual."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B x N x C
        q, k, v = self.qkv(self.norm(tokens)).chunk(3, dim=-1)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=-1)
        tokens = tokens + self.proj(attn @ v)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class TinySegNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(_conv_bn_relu(3, 16, 2), _conv_bn_relu(16, 32, 2))  # stride 4
        self.attn = TinyAttention(32)
        self.down = _conv_bn_relu(32, 32, 2)  # stride 8 context branch
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f4 = self.attn(self.stem(x))
        f8 = F.interpolate(self.down(f4), size=f4.shape[2:], mode="bilinear", align_corners=False)
        return self.classifier(torch.cat([f4, f8], dim=1))


def build_tiny(num_classes: int, seed: int) -> TinySegNet:
    """Deterministic random weights. BatchNorm statistics are randomized too, so
    exporters really have to fold them (the defaults 0/1 would make folding a no-op)."""
    gen = torch.Generator().manual_seed(seed)
    model = TinySegNet(num_classes)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(torch.randn(m.num_features, generator=gen) * 0.1)
                m.running_var.copy_(torch.rand(m.num_features, generator=gen) + 0.5)
    return model
