"""Calibration changes confidence while preserving classes."""

import numpy as np
import pytest

from mortgage.calibration import apply_temperature, fit_temperature, multiclass_brier


def test_temperature_fits_confidence_without_changing_predictions():
    probabilities = np.tile(np.array([[0.99, 0.01]], dtype=np.float32), (100, 1))
    labels = np.array([0] * 75 + [1] * 25, dtype=np.int64)
    temperature = fit_temperature(probabilities, labels)
    calibrated = apply_temperature(probabilities, temperature)
    assert temperature > 1
    np.testing.assert_allclose(calibrated[:, 0], 0.75, atol=1e-4)
    np.testing.assert_array_equal(calibrated.argmax(1), probabilities.argmax(1))
    assert multiclass_brier(calibrated, labels) < multiclass_brier(
        probabilities, labels
    )


def test_temperature_handles_zero_probabilities_and_rejects_invalid_inputs():
    probabilities = np.eye(3, dtype=np.float32)
    result = apply_temperature(probabilities, 1)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(1), 1)
    np.testing.assert_array_equal(result.argmax(1), np.arange(3))
    with pytest.raises(ValueError):
        apply_temperature(probabilities, 0)
    with pytest.raises(ValueError):
        fit_temperature(probabilities, np.array([], dtype=np.int64))
