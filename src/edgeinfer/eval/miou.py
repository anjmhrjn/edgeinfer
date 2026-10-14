"""mIoU from one confusion matrix accumulated over the whole dataset.

Per-image IoUs are not averaged: a class covering 10 pixels in one image would
weigh as much as one covering a million pixels in another. Summing counts over
all images first gives every pixel equal weight, which is the Cityscapes protocol.
"""

from __future__ import annotations

import numpy as np

from edgeinfer.data.labels import IGNORE_INDEX


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # rows = ground truth, columns = prediction
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, label: np.ndarray) -> None:
        if pred.shape != label.shape:
            raise ValueError(f"prediction {pred.shape} and label {label.shape} differ")
        valid = label != self.ignore_index
        gt = label[valid].astype(np.int64)
        pr = pred[valid].astype(np.int64)
        if gt.size and (gt.max() >= self.num_classes or pr.max() >= self.num_classes):
            raise ValueError("class index out of range")
        counts = np.bincount(gt * self.num_classes + pr, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def iou(self) -> np.ndarray:
        """Per-class IoU = TP / (TP + FP + FN); NaN if a class is in neither GT nor prediction."""
        tp = np.diag(self.matrix).astype(np.float64)
        union = self.matrix.sum(axis=0) + self.matrix.sum(axis=1) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(union > 0, tp / union, np.nan)

    def miou(self) -> float:
        return float(np.nanmean(self.iou()))

    def pixel_accuracy(self) -> float:
        total = self.matrix.sum()
        return float(np.trace(self.matrix) / total) if total else float("nan")
