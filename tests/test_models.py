import json
from pathlib import Path

import numpy as np
import yaml

from src.models.rf_model import RandomForestGlucoseModel, train_random_forest_from_config


def test_rf_model_train_predict_and_reload(tmp_path):
	rng = np.random.default_rng(42)
	X_train = rng.normal(size=(120, 12, 5))
	y_train = rng.normal(loc=120, scale=20, size=(120,))
	X_test = rng.normal(size=(24, 12, 5))

	model = RandomForestGlucoseModel(
		{
			"data": {"seed": 42},
			"model": {"random_forest": {"n_estimators": 50, "max_depth": 8, "min_samples_split": 2}},
		}
	)
	history = model.train(X_train, y_train)
	preds = model.predict(X_test)

	assert "train_RMSE" in history
	assert len(preds) == len(X_test)

	model_path = tmp_path / "rf.pkl"
	model.save(str(model_path))
	assert model_path.exists()

	reloaded = RandomForestGlucoseModel()
	reloaded.load(str(model_path))
	preds_reload = reloaded.predict(X_test)

	assert np.allclose(preds, preds_reload)


def test_train_random_forest_from_config_runs(tmp_path):
	data_dir = tmp_path / "raw"
	data_dir.mkdir(parents=True, exist_ok=True)

	timestamps = np.arange("2024-01-01T00:00", "2024-01-02T00:00", dtype="datetime64[5m]")
	rows = []
	for idx, patient_id in enumerate(["P001", "P002", "P003", "P004"]):
		for t in timestamps:
			glucose = 95 + idx * 5 + np.sin(len(rows) / 20.0) * 10
			rows.append(
				{
					"patient_id": patient_id,
					"timestamp": str(t),
					"glucose": float(glucose),
					"carbs": float((len(rows) % 6 == 0) * 30),
					"insulin": float((len(rows) % 6 == 0) * 3),
					"activity": int(len(rows) % 4 == 0) * 15,
					"stress": int(4 + (len(rows) % 3)),
				}
			)

	import pandas as pd

	pd.DataFrame(rows).to_csv(data_dir / "training_data_complete.csv", index=False)

	config = {
		"data": {"output_dir": str(data_dir), "seed": 42},
		"model": {
			"sequence_length": 12,
			"features": ["glucose", "carbs", "insulin", "activity", "stress"],
			"random_forest": {"n_estimators": 30, "max_depth": 8, "min_samples_split": 2},
		},
	}

	cfg_path = tmp_path / "config.yaml"
	with open(cfg_path, "w", encoding="utf-8") as f:
		yaml.safe_dump(config, f)

	cwd = Path.cwd()
	try:
		import os

		os.chdir(tmp_path)
		metrics = train_random_forest_from_config(str(cfg_path))
	finally:
		os.chdir(cwd)

	assert "RMSE" in metrics
	assert (tmp_path / "models" / "rf_baseline.pkl").exists()

	with open(tmp_path / "models" / "rf_baseline_metrics.json", "r", encoding="utf-8") as f:
		metrics_json = json.load(f)

	assert "Clarke_A+B" in metrics_json
