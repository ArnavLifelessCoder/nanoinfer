import numpy as np
import pytest

from scripts.hidden_error_stats import hidden_error_stats


def test_stats_masks_counts_and_worst_coordinates():
    target = np.array([[0.5, 1, 1023], [1024, -2048, 0]], dtype=np.float64)
    actual = target + np.array([[1 / 16384, 0.25, 0.5], [2, 4, 0]])
    stats = hidden_error_stats(actual, target)
    assert stats["max_abs_diff"] == 4
    assert stats["max_abs_diff_torch_abs_lt_1024"] == 0.5
    assert stats["max_relative_diff_torch_abs_ge_1"] == 0.25
    assert stats["count_abs_diff_gt_1e_4"] == 4
    assert stats["worst_element"] == {"position": 1, "channel": 1, "torch_abs": 2048.0}


def test_empty_mask_is_null_and_worst_tie_uses_first_element():
    stats = hidden_error_stats(np.zeros((1, 2)), np.zeros((1, 2)))
    assert stats["max_relative_diff_torch_abs_ge_1"] is None
    assert stats["worst_element"] == {"position": 0, "channel": 0, "torch_abs": 0}
    stats = hidden_error_stats(np.full((1, 2), 1024), np.full((1, 2), 1024))
    assert stats["max_abs_diff_torch_abs_lt_1024"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_stats_are_rejected(bad):
    with pytest.raises(AssertionError):
        hidden_error_stats(np.array([[bad]]), np.zeros((1, 1)))
    with pytest.raises(AssertionError):
        hidden_error_stats(np.zeros((1, 1)), np.array([[bad]]))


def test_wrong_shape_is_rejected():
    with pytest.raises(AssertionError):
        hidden_error_stats(np.zeros((2, 1)), np.zeros((1, 2)))
