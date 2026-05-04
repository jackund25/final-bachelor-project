import numpy as np
import pytest

from src.utils.metrics import calculate_all_metrics


def test_calculate_all_metrics_keys_and_values():
	y_true = np.array([100, 120, 150, 180, 200], dtype=float)
	y_pred = np.array([105, 115, 155, 175, 195], dtype=float)

	metrics = calculate_all_metrics(y_true, y_pred)

	expected_keys = {
		"RMSE",
		"MAE",
		"MAPE",
		"Clarke_A",
		"Clarke_B",
		"Clarke_C",
		"Clarke_D",
		"Clarke_E",
		"Clarke_A+B",
	}
	assert expected_keys.issubset(metrics.keys())

	assert metrics["RMSE"] == pytest.approx(5.0, rel=1e-6)
	assert metrics["MAE"] == pytest.approx(5.0, rel=1e-6)
	assert metrics["MAPE"] == pytest.approx(3.5555555556, rel=1e-6)

	assert 0.0 <= metrics["Clarke_A"] <= 100.0
	assert 0.0 <= metrics["Clarke_B"] <= 100.0
	assert 0.0 <= metrics["Clarke_A+B"] <= 100.0
	assert metrics["Clarke_A+B"] >= 90.0