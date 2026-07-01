from __future__ import annotations

import pytest
from cascadia_mlx.train import _calibration_metrics


def test_calibration_metrics_recover_exact_linear_relation() -> None:
    # Predictions [1, 2, 3], targets [3, 5, 7] => target = 2 * prediction + 1.
    metrics = _calibration_metrics(
        3,
        [
            6.0,
            15.0,
            14.0,
            83.0,
            34.0,
        ],
    )

    assert metrics["predicted_total_mean"] == pytest.approx(2.0)
    assert metrics["target_total_mean"] == pytest.approx(5.0)
    assert metrics["total_bias"] == pytest.approx(-3.0)
    assert metrics["total_correlation"] == pytest.approx(1.0)
    assert metrics["calibration_slope"] == pytest.approx(2.0)
    assert metrics["calibration_intercept"] == pytest.approx(1.0)


def test_calibration_metrics_handle_constant_predictions() -> None:
    metrics = _calibration_metrics(
        2,
        [
            8.0,
            10.0,
            32.0,
            52.0,
            40.0,
        ],
    )

    assert metrics["total_correlation"] == 0.0
    assert metrics["calibration_slope"] == 0.0
    assert metrics["calibration_intercept"] == pytest.approx(5.0)
