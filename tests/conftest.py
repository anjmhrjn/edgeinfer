from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edgeinfer.data.cityscapes import IMAGE_SUFFIX, LABEL_SUFFIX


@pytest.fixture
def fake_cityscapes(tmp_path: Path) -> Path:
    """A tiny dataset in the official layout: 2 cities x 3 images per split, 4x6 pixels."""
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for city in ("aachen", "zurich"):
            img_dir = tmp_path / "leftImg8bit" / split / city
            lbl_dir = tmp_path / "gtFine" / split / city
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            for i in range(3):
                image_id = f"{city}_{i:06d}_000019"
                image = rng.integers(0, 256, (4, 6, 3), dtype=np.uint8)
                label = rng.integers(0, 34, (4, 6), dtype=np.uint8)
                Image.fromarray(image, "RGB").save(img_dir / f"{image_id}{IMAGE_SUFFIX}")
                Image.fromarray(label, "L").save(lbl_dir / f"{image_id}{LABEL_SUFFIX}")
    return tmp_path


FIXTURES = Path(__file__).parent / "fixtures"
TINY_CONFIG = FIXTURES / "tiny_fixture.yaml"


@pytest.fixture(scope="session")
def tiny_onnx(tmp_path_factory) -> Path:
    """The tiny fixture model exported to ONNX once per test session."""
    from edgeinfer.config import load_model_config
    from edgeinfer.export.onnx_export import export_model

    path, _ = export_model(load_model_config(TINY_CONFIG), tmp_path_factory.mktemp("onnx"))
    return path
