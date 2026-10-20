"""Protect the numerical acceptance gate independently of model execution."""

import numpy as np
import pytest

from tests.golden_forward import compare_array


def test_hidden_tolerance_is_elementwise():
    target = np.array([3000, 0], dtype=np.float32)
    actual = np.array([3000.0005, 0.00009], dtype=np.float32)
    assert compare_array(actual, target, hidden=True)[2]
    actual[1] = 0.0002
    assert not compare_array(actual, target, hidden=True)[2]


def test_hidden_boundary_is_inclusive():
    target = np.zeros(1, dtype=np.float64)
    assert compare_array(np.array([1e-4]), target, hidden=True)[2]
    assert not compare_array(np.array([1.00001e-4]), target, hidden=True)[2]


def test_logits_use_strict_absolute_bound():
    target = np.zeros(1, dtype=np.float64)
    assert compare_array(np.array([0.000999]), target, hidden=False)[2]
    assert not compare_array(np.array([0.001]), target, hidden=False)[2]
    assert not compare_array(np.array([3000.002]), np.array([3000.0]), hidden=False)[2]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("hidden", [False, True])
def test_nonfinite_values_fail(value, hidden):
    for actual, target in [(value, 0), (0, value)]:
        with pytest.raises(AssertionError):
            compare_array(np.array([actual]), np.array([target]), hidden=hidden)


def test_shape_and_empty_arrays_fail():
    with pytest.raises(AssertionError):
        compare_array(np.zeros((1, 2)), np.zeros((2,)), hidden=True)
    with pytest.raises(AssertionError):
        compare_array(np.zeros(0), np.zeros(0), hidden=False)
