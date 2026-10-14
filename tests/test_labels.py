import numpy as np
import pytest

from edgeinfer.data.labels import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, to_train_ids

# Hand-checked against cityscapesScripts helpers/labels.py: labelId -> trainId for the 19 classes.
EXPECTED = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}  # fmt: skip


def test_evaluated_label_ids_map_to_train_ids():
    ids = np.array(list(EXPECTED), dtype=np.uint8)
    assert to_train_ids(ids).tolist() == list(EXPECTED.values())


def test_every_other_value_is_ignored():
    others = np.array([v for v in range(256) if v not in EXPECTED], dtype=np.uint8)
    assert (to_train_ids(others) == IGNORE_INDEX).all()


def test_class_names_in_train_id_order():
    assert len(CLASS_NAMES) == NUM_CLASSES
    assert CLASS_NAMES[0] == "road"
    assert CLASS_NAMES[13] == "car"
    assert CLASS_NAMES[-1] == "bicycle"


def test_rejects_non_uint8():
    with pytest.raises(TypeError):
        to_train_ids(np.zeros(3, dtype=np.int64))
