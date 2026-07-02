"""Random Forest baseline model for glucose prediction."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.base_model import BaseGlucoseModel
from src.utils.metrics import calculate_all_metrics


class RandomForestGlucoseModel(BaseGlucoseModel):
	"""Random Forest model using flattened time windows."""

	def __init__(self, config: Optional[Dict] = None):
		super().__init__(config)
		rf_cfg = (config or {}).get("model", {}).get("random_forest", {})

		self.model = RandomForestRegressor(
			n_estimators=rf_cfg.get("n_estimators", 200),
			max_depth=rf_cfg.get("max_depth", 20),
			min_samples_split=rf_cfg.get("min_samples_split", 5),
			random_state=(config or {}).get("data", {}).get("seed", 42),
			n_jobs=-1,
		)

	def _flatten(self, X: np.ndarray) -> np.ndarray:
		if X.ndim != 3:
			raise ValueError("Expected 3D input (samples, sequence_length, n_features)")
		n_samples, seq_len, n_features = X.shape
		return X.reshape(n_samples, seq_len * n_features)

	def train(
		self,
		X_train: np.ndarray,
		y_train: np.ndarray,
		X_val: Optional[np.ndarray] = None,
		y_val: Optional[np.ndarray] = None,
	) -> Dict:
		X_train, y_train = self._validate_training_data(X_train, y_train)
		if X_val is not None and y_val is not None:
			X_val, y_val = self._validate_training_data(X_val, y_val)

		X_train_flat = self._flatten(X_train)
		self.model.fit(X_train_flat, y_train)
		self.is_trained = True

		history: Dict[str, float] = {"train_samples": float(len(y_train))}
		train_pred = self.model.predict(X_train_flat)
		train_metrics = calculate_all_metrics(y_train, train_pred)
		history.update({f"train_{k}": float(v) for k, v in train_metrics.items()})

		if X_val is not None and y_val is not None and len(y_val) > 0:
			val_pred = self.predict(X_val)
			val_metrics = calculate_all_metrics(y_val, val_pred)
			history.update({f"val_{k}": float(v) for k, v in val_metrics.items()})

		return history

	def predict(self, X: np.ndarray) -> np.ndarray:
		if not self.is_trained:
			raise RuntimeError("Model is not trained yet")
		X = self._validate_prediction_data(X)
		X_flat = self._flatten(X)
		return self.model.predict(X_flat)

	def save(self, filepath: str) -> None:
		model_path = Path(filepath)
		model_path.parent.mkdir(parents=True, exist_ok=True)

		with open(model_path, "wb") as f:
			pickle.dump(self.model, f)

	def load(self, filepath: str) -> None:
		with open(filepath, "rb") as f:
			self.model = pickle.load(f)

		self.is_trained = True


def train_random_forest_from_config(
	config_path: str = "config.yaml",
	data_source: str = "auto",
	horizon: Optional[int] = None,
) -> Dict:
	"""Train RF baseline pada horizon tertentu, lalu simpan model + metrik.

	Jika horizon None, pakai config.model.default_horizon. Bundle untuk horizon
	default juga disimpan sebagai `rf_inference_bundle.pkl` (dipakai aplikasi).
	"""
	with open(config_path, "r", encoding="utf-8") as f:
		config = yaml.safe_load(f)

	loader = DiabetesDataLoader(config["data"]["output_dir"])
	if data_source == "auto":
		primary_source = config.get("data", {}).get("primary_source", "ohio_t1dm")
		fallback_source = config.get("data", {}).get("fallback_source", "latest_generated")
		df, used_source = loader.load_preferred_dataset(primary_source, fallback_source)
	elif data_source == "ohio_t1dm":
		df = loader.load_csv("ohio_t1dm_merged.csv")
		used_source = "ohio_t1dm"
	else:
		df = loader.load_latest_dataset()
		used_source = "latest_generated"
	df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
	print(f"Data source  : {used_source}")

	preprocessor = DataPreprocessor(config)
	df = preprocessor.handle_missing_values(df)

	sequence_length = config["model"].get("sequence_length", 12)
	default_horizon = config["model"].get("default_horizon", 1)
	if horizon is None:
		horizon = default_horizon

	# Fitur engineered (IOB/COB/tren/waktu) + target delta — opsional dari config.
	use_engineered = config["model"].get("use_engineered", False)
	predict_delta = config["model"].get("predict_delta", False)
	fe_params = config["model"].get("feature_engineering", {})
	if use_engineered:
		df = preprocessor.engineer_features(df, **fe_params)
		feature_list = config["model"]["engineered_features"]
	else:
		feature_list = config["model"]["features"]
	preprocessor.feature_columns = list(feature_list)

	patient_ids = sorted(df["patient_id"].unique().tolist())
	if len(patient_ids) < 2:
		raise ValueError("Need at least 2 patients for train/test split")

	test_patients = patient_ids[-2:]
	train_df, test_df = preprocessor.split_by_patient(df, test_patients)

	X_train, y_train, anc_train = preprocessor.create_sequences(train_df, sequence_length, horizon, return_anchor=True)
	X_test, y_test, anc_test = preprocessor.create_sequences(test_df, sequence_length, horizon, return_anchor=True)

	X_train_scaled, X_test_scaled = preprocessor.normalize_data(X_train, X_test)

	# Target: absolut, atau delta (selisih dari glukosa terakhir window) lalu direkonstruksi.
	y_train_fit = (y_train - anc_train) if predict_delta else y_train

	model = RandomForestGlucoseModel(config)
	model.train(X_train_scaled, y_train_fit)
	y_pred = model.predict(X_test_scaled)
	if predict_delta:
		y_pred = y_pred + anc_test

	metrics = calculate_all_metrics(y_test, y_pred)

	models_dir = Path("models")
	models_dir.mkdir(parents=True, exist_ok=True)

	model.save(str(models_dir / f"rf_baseline_h{horizon}.pkl"))

	bundle = {
		"model": model.model,
		"scaler": preprocessor.scaler,
		"features": feature_list,
		"sequence_length": sequence_length,
		"prediction_horizon": horizon,
		"use_engineered": use_engineered,
		"predict_delta": predict_delta,
		"feature_engineering": fe_params,
	}
	with open(models_dir / f"rf_inference_bundle_h{horizon}.pkl", "wb") as f:
		pickle.dump(bundle, f)
	with open(models_dir / f"rf_metrics_h{horizon}.json", "w", encoding="utf-8") as f:
		json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

	# Horizon default → simpan juga dengan nama kanonik untuk aplikasi
	if horizon == default_horizon:
		model.save(str(models_dir / "rf_baseline.pkl"))
		with open(models_dir / "rf_inference_bundle.pkl", "wb") as f:
			pickle.dump(bundle, f)
		with open(models_dir / "rf_baseline_metrics.json", "w", encoding="utf-8") as f:
			json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

	print("=" * 60)
	print(f"RANDOM FOREST — horizon {horizon} langkah (+{horizon * 5} menit)")
	print("=" * 60)
	print(f"Train patients: {train_df['patient_id'].nunique()}")
	print(f"Test patients : {test_df['patient_id'].nunique()}")
	print(f"Train samples : {len(y_train)}")
	print(f"Test samples  : {len(y_test)}")
	print("-" * 60)
	for key, value in metrics.items():
		print(f"{key:12s}: {value:.4f}")
	print("-" * 60)

	return metrics


def main() -> None:
	parser = argparse.ArgumentParser(description="Train Random Forest baseline model")
	parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
	parser.add_argument(
		"--data_source",
		default="auto",
		choices=["auto", "latest", "ohio_t1dm"],
		help="Data source to train on (auto: config primary/fallback, latest, or ohio_t1dm)",
	)
	parser.add_argument(
		"--horizon",
		type=int,
		default=None,
		help="Horizon prediksi (langkah). Default: latih semua config.model.prediction_horizons",
	)
	args = parser.parse_args()

	if args.horizon is not None:
		train_random_forest_from_config(args.config, args.data_source, args.horizon)
	else:
		with open(args.config, "r", encoding="utf-8") as f:
			cfg = yaml.safe_load(f)
		horizons = cfg["model"].get("prediction_horizons", [cfg["model"].get("default_horizon", 1)])
		for h in horizons:
			train_random_forest_from_config(args.config, args.data_source, h)


if __name__ == "__main__":
	main()
