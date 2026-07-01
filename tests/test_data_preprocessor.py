import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor


def _build_config():
	return {
		"model": {
			"features": ["glucose", "carbs", "insulin", "activity", "stress"],
		},
	}


def test_create_sequences_requires_enough_rows():
	preprocessor = DataPreprocessor(_build_config())
	df = pd.DataFrame(
		{
			"patient_id": ["P001"] * 4,
			"timestamp": pd.date_range("2024-01-01", periods=4, freq="5min"),
			"glucose": [100.0, 101.0, 102.0, 103.0],
			"carbs": [0.0, 0.0, 0.0, 0.0],
			"insulin": [0.0, 0.0, 0.0, 0.0],
			"activity": [0, 0, 0, 0],
			"stress": [5, 5, 5, 5],
		}
	)

	with pytest.raises(ValueError, match=r"Need at least sequence_length"):
		preprocessor.create_sequences(df, sequence_length=4)


def test_create_sequences_does_not_cross_patient_boundaries():
	preprocessor = DataPreprocessor(_build_config())
	df = pd.DataFrame(
		{
			"patient_id": ["P001"] * 3 + ["P002"] * 3,
			"timestamp": pd.to_datetime([
				"2024-01-01 00:00",
				"2024-01-01 00:05",
				"2024-01-01 00:10",
				"2024-01-02 00:00",
				"2024-01-02 00:05",
				"2024-01-02 00:10",
			]),
			"glucose": [100.0, 101.0, 102.0, 110.0, 111.0, 112.0],
			"carbs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			"insulin": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			"activity": [0, 0, 0, 0, 0, 0],
			"stress": [5, 5, 5, 6, 6, 6],
		}
	)

	X, y = preprocessor.create_sequences(df, sequence_length=2)

	assert X.shape == (2, 2, 5)
	assert y.tolist() == [102.0, 112.0]


def test_handle_missing_values_sorts_and_fills():
	preprocessor = DataPreprocessor(_build_config())
	df = pd.DataFrame(
		{
			"patient_id": ["P001", "P001", "P001"],
			"timestamp": pd.to_datetime(["2024-01-01 00:10", "2024-01-01 00:00", "2024-01-01 00:05"]),
			"glucose": [102.0, np.nan, 101.0],
			"carbs": [0.0, 15.0, np.nan],
			"insulin": [0.0, 1.5, np.nan],
			"activity": [0, np.nan, 5],
			"stress": [5, 6, np.nan],
		}
	)

	cleaned = preprocessor.handle_missing_values(df)

	assert cleaned["timestamp"].is_monotonic_increasing
	assert cleaned.isnull().sum().sum() == 0