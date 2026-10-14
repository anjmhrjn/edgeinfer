import numpy as np
import pytest

from edgeinfer.eval.miou import ConfusionMatrix


def test_hand_computed_two_class_example():
    # GT:   0 0 1 1      Pred: 0 1 1 1
    # class 0: TP=1, FN=1, FP=0 -> IoU 1/2
    # class 1: TP=2, FN=0, FP=1 -> IoU 2/3
    cm = ConfusionMatrix(2)
    cm.update(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(cm.iou(), [1 / 2, 2 / 3])
    assert cm.miou() == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert cm.pixel_accuracy() == pytest.approx(3 / 4)


def test_ignore_pixels_do_not_count():
    cm = ConfusionMatrix(2)
    cm.update(np.array([1, 1, 0]), np.array([255, 255, 0]))
    np.testing.assert_array_equal(cm.matrix, [[1, 0], [0, 0]])


def test_dataset_level_accumulation_is_not_per_image_mean():
    # Image A: class 0 tiny and wrong; image B: class 0 large and right.
    a_pred, a_gt = np.array([1, 1]), np.array([0, 1])
    b_pred, b_gt = np.zeros(98, dtype=int), np.zeros(98, dtype=int)
    cm = ConfusionMatrix(2)
    cm.update(a_pred, a_gt)
    cm.update(b_pred, b_gt)
    whole = ConfusionMatrix(2)
    whole.update(np.concatenate([a_pred, b_pred]), np.concatenate([a_gt, b_gt]))
    np.testing.assert_array_equal(cm.matrix, whole.matrix)
    assert cm.iou()[0] == pytest.approx(98 / 99)  # per-image mean would give (0 + 1) / 2


def test_absent_class_is_nan_and_excluded():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 1]), np.array([0, 1]))
    assert np.isnan(cm.iou()[2])
    assert cm.miou() == 1.0


def test_predicted_but_absent_class_scores_zero():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 2]), np.array([0, 1]))
    assert cm.iou()[2] == 0.0
    assert cm.iou()[1] == 0.0


def test_out_of_range_and_shape_errors():
    cm = ConfusionMatrix(2)
    with pytest.raises(ValueError, match="out of range"):
        cm.update(np.array([5]), np.array([0]))
    with pytest.raises(ValueError, match="differ"):
        cm.update(np.array([0, 1]), np.array([0]))
