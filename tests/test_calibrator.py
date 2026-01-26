import numpy as np
import pytest
from fixed_income_ad_playground.calibration.calibrator import _raise_with_block_debug


def test_raise_with_block_debug_no_error():
    vec = np.array([0.1, -0.2, 0.0])
    slices = {"swaps": (0, 3)}
    # should not raise
    _raise_with_block_debug(vec, slices)


def test_raise_with_block_debug_raises():
    vec = np.array([0.1, np.inf, 0.0])
    slices = {"swaps": (0, 2), "futures": (2, 3)}
    with pytest.raises(ValueError) as exc:
        _raise_with_block_debug(vec, slices)
    assert "Non-finite residuals detected" in str(exc.value)
