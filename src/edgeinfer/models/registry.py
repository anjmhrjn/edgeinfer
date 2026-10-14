"""Build an eval-mode PyTorch model (image -> stride-`output_stride` logits) from a config."""

from __future__ import annotations

from torch import nn

from edgeinfer.config import ModelConfig
from edgeinfer.models.weights import fetch_url_weights


def weights_id(cfg: ModelConfig) -> str:
    """Stable identifier of the exact weights a config points to."""
    w = cfg.weights
    if w["source"] == "huggingface":
        return f"hf:{w['repo']}@{w['revision']}"
    if w["source"] == "url":
        return f"sha256:{w['sha256']}"
    raise ValueError(f"{cfg.name}: unknown weights source {w['source']!r}")


def _require(cfg: ModelConfig, *keys: str) -> None:
    missing = [k for k in keys if k not in cfg.weights]
    if missing:
        raise ValueError(f"{cfg.name}: weights config missing {missing}")


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.architecture == "segformer":
        from edgeinfer.models.segformer import load_segformer

        _require(cfg, "repo", "revision")
        model = load_segformer(cfg.weights["repo"], cfg.weights["revision"])
    elif cfg.architecture == "deeplabv3plus_mobilenet":
        from edgeinfer.models.deeplabv3plus import DeepLabV3PlusLogits

        _require(cfg, "url", "filename", "sha256")
        if cfg.backbone_output_stride not in (8, 16):
            raise ValueError(f"{cfg.name}: backbone_output_stride must be 8 or 16")
        model = DeepLabV3PlusLogits(cfg.num_classes, cfg.backbone_output_stride)
        model.load_checkpoint(
            fetch_url_weights(cfg.weights["url"], cfg.weights["filename"], cfg.weights["sha256"])
        )
    else:
        raise ValueError(f"{cfg.name}: unknown architecture {cfg.architecture!r}")
    return model.eval()
