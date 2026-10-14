"""Cityscapes labelId -> trainId mapping.

The gtFine `*_labelIds.png` files store 34 label IDs (0-33). Evaluation uses 19 of
them, renumbered 0-18 as "trainIds"; everything else maps to IGNORE_INDEX and is
excluded from the confusion matrix. Mapping copied from cityscapesScripts
`helpers/labels.py` (license plate, id -1, never appears in labelIds PNGs).
"""

from __future__ import annotations

import numpy as np

IGNORE_INDEX = 255
NUM_CLASSES = 19

# (labelId, name, trainId)
LABELS: tuple[tuple[int, str, int], ...] = (
    (0, "unlabeled", IGNORE_INDEX),
    (1, "ego vehicle", IGNORE_INDEX),
    (2, "rectification border", IGNORE_INDEX),
    (3, "out of roi", IGNORE_INDEX),
    (4, "static", IGNORE_INDEX),
    (5, "dynamic", IGNORE_INDEX),
    (6, "ground", IGNORE_INDEX),
    (7, "road", 0),
    (8, "sidewalk", 1),
    (9, "parking", IGNORE_INDEX),
    (10, "rail track", IGNORE_INDEX),
    (11, "building", 2),
    (12, "wall", 3),
    (13, "fence", 4),
    (14, "guard rail", IGNORE_INDEX),
    (15, "bridge", IGNORE_INDEX),
    (16, "tunnel", IGNORE_INDEX),
    (17, "pole", 5),
    (18, "polegroup", IGNORE_INDEX),
    (19, "traffic light", 6),
    (20, "traffic sign", 7),
    (21, "vegetation", 8),
    (22, "terrain", 9),
    (23, "sky", 10),
    (24, "person", 11),
    (25, "rider", 12),
    (26, "car", 13),
    (27, "truck", 14),
    (28, "bus", 15),
    (29, "caravan", IGNORE_INDEX),
    (30, "trailer", IGNORE_INDEX),
    (31, "train", 16),
    (32, "motorcycle", 17),
    (33, "bicycle", 18),
)

CLASS_NAMES: tuple[str, ...] = tuple(
    name for _, name, train_id in sorted(LABELS, key=lambda r: r[2]) if train_id != IGNORE_INDEX
)


def _build_lut() -> np.ndarray:
    # 256 entries so any uint8 value is a valid index; unknown IDs are ignored.
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for label_id, _, train_id in LABELS:
        lut[label_id] = train_id
    return lut


LABEL_ID_TO_TRAIN_ID = _build_lut()


def to_train_ids(label_ids: np.ndarray) -> np.ndarray:
    """Map a uint8 array of Cityscapes labelIds to trainIds (ignore = 255)."""
    if label_ids.dtype != np.uint8:
        raise TypeError(f"expected uint8 labelIds, got {label_ids.dtype}")
    return LABEL_ID_TO_TRAIN_ID[label_ids]
