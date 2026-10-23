import numpy as np
import pytest

from python.quantize import GROUP, dequantize, quantize


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shape", [(32, 1), (64, 7), (768, 96), (96, 768)])
def test_round_trip_error_is_at_most_half_a_step(axis, shape):
    if shape[axis] % GROUP:
        pytest.skip("input dimension must be a multiple of the group")
    rng = np.random.default_rng(42)
    w = (rng.standard_normal(shape) * rng.uniform(0.01, 3, size=shape)).astype(np.float32)
    q, scales = quantize(w, axis)
    assert q.dtype == np.int8 and q.shape == w.shape
    assert scales.dtype == np.float32
    assert scales.shape[axis] == shape[axis] // GROUP
    assert np.abs(q).max() <= 127
    error = np.abs(dequantize(q, scales, axis) - w)
    bound = np.repeat(scales, GROUP, axis=axis) / 2
    # Half a quantization step, plus float32 rounding of q * scale.
    assert np.all(error <= bound * (1 + 1e-6) + 1e-12)


def test_group_maximum_maps_to_127_and_sign_is_kept():
    w = np.zeros((32, 2), dtype=np.float32)
    w[5, 0], w[9, 0] = 2.0, -1.0
    w[3, 1] = -4.0
    q, scales = quantize(w, 0)
    assert q[5, 0] == 127 and q[9, 0] == -64 and q[3, 1] == -127
    np.testing.assert_allclose(scales, [[2 / 127, 4 / 127]], rtol=1e-7)


def test_all_zero_group_has_zero_scale_and_values():
    w = np.zeros((64, 3), dtype=np.float32)
    w[40, 1] = 1.0
    q, scales = quantize(w, 0)
    assert scales[0].tolist() == [0, 0, 0] and not q[:32].any()
    assert dequantize(q, scales, 0)[40, 1] == pytest.approx(1.0)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        quantize(np.zeros((33, 2), dtype=np.float32), 0)
    with pytest.raises(ValueError):
        quantize(np.zeros(32, dtype=np.float32), 0)
