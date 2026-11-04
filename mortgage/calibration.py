"""Scalar temperature calibration from held-out targets."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .data import FloatArray, IntArray

PROBABILITY_FLOOR = 1e-15


def apply_temperature(probabilities: FloatArray, temperature: float) -> FloatArray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    scores = (
        np.log(np.clip(probabilities.astype(np.float64), PROBABILITY_FLOOR, 1))
        / temperature
    )
    return np.exp(scores - logsumexp(scores, axis=1, keepdims=True)).astype(np.float32)


def fit_temperature(probabilities: FloatArray, labels: IntArray) -> float:
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError(
            "Calibration requires matching nonempty probabilities and labels"
        )
    scores = np.log(np.clip(probabilities.astype(np.float64), PROBABILITY_FLOOR, 1))
    actual_scores = scores[np.arange(len(labels)), labels]

    def loss(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        return float(
            (
                logsumexp(scores / temperature, axis=1) - actual_scores / temperature
            ).mean()
        )

    result = minimize_scalar(
        loss,
        bounds=(-np.log(10), np.log(10)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x)) if result.fun < loss(0.0) else 1.0


def multiclass_brier(probabilities: FloatArray, labels: IntArray) -> float:
    actual = np.eye(probabilities.shape[1], dtype=np.float32)[labels]
    return float(
        np.square(probabilities.astype(np.float64) - actual).sum(axis=1).mean()
    )
