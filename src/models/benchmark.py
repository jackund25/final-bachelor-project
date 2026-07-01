"""Benchmarking utilities: trains RF baseline and a gradient-boosting fallback.

This module saves model artifacts and a metrics_experiments.json file containing
results and provenance (config, seed, data source).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import yaml
import logging
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)

# Optional accelerators
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.rf_model import RandomForestGlucoseModel
from src.models.base_model import BaseGlucoseModel
from src.utils.metrics import calculate_all_metrics


class SklearnGBMGlucoseModel(BaseGlucoseModel):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        gb_cfg = (config or {}).get("model", {}).get("gbm", {})
        self.model = GradientBoostingRegressor(
            n_estimators=gb_cfg.get("n_estimators", 100),
            learning_rate=gb_cfg.get("learning_rate", 0.1),
            max_depth=gb_cfg.get("max_depth", 3),
            random_state=(config or {}).get("data", {}).get("seed", 42),
        )

    def _flatten(self, X):
        if X.ndim != 3:
            raise ValueError("Expected 3D input (samples, sequence_length, n_features)")
        n_samples, seq_len, n_features = X.shape
        return X.reshape(n_samples, seq_len * n_features)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X_train, y_train = self._validate_training_data(X_train, y_train)
        X_train_flat = self._flatten(X_train)
        self.model.fit(X_train_flat, y_train)
        self.is_trained = True
        preds = self.model.predict(X_train_flat)
        return {"train_samples": float(len(y_train)), **calculate_all_metrics(y_train, preds)}

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        X = self._validate_prediction_data(X)
        return self.model.predict(self._flatten(X))

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True


def run_benchmark_from_config(config_path: str = "config.yaml", data_source: str = "auto") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DiabetesDataLoader(config["data"]["output_dir"]) if config.get("data") else DiabetesDataLoader("data/raw")
    if data_source == "auto":
        df, used_source = loader.load_preferred_dataset(config.get("data", {}).get("primary_source", "ohio_t1dm"),
                                                       config.get("data", {}).get("fallback_source", "latest_generated"))
    elif data_source == "ohio_t1dm":
        df = loader.load_csv("ohio_t1dm_merged.csv")
        used_source = "ohio_t1dm"
    else:
        df = loader.load_latest_dataset()
        used_source = "latest_generated"

    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    preprocessor = DataPreprocessor(config)
    df = preprocessor.handle_missing_values(df)

    sequence_length = config["model"].get("sequence_length", 12)

    patient_ids = sorted(df["patient_id"].unique().tolist())
    test_patients = patient_ids[-2:]
    train_df, test_df = preprocessor.split_by_patient(df, test_patients)

    X_train, y_train = preprocessor.create_sequences(train_df, sequence_length=sequence_length)
    X_test, y_test = preprocessor.create_sequences(test_df, sequence_length=sequence_length)

    X_train_scaled, X_test_scaled = preprocessor.normalize_data(X_train, X_test)

    results = {}
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Random Forest
    rf = RandomForestGlucoseModel(config)
    rf.train(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    rf_metrics = calculate_all_metrics(y_test, rf_preds)
    rf.save(str(models_dir / "rf_baseline.pkl"))
    rf_bundle = {"model": rf.model, "scaler": preprocessor.scaler, "features": config["model"]["features"], "sequence_length": sequence_length}
    with open(models_dir / "rf_inference_bundle.pkl", "wb") as f:
        pickle.dump(rf_bundle, f)
    results["rf"] = rf_metrics

    # Gradient boosting fallback
    gb = SklearnGBMGlucoseModel(config)
    gb.train(X_train_scaled, y_train)
    gb_preds = gb.predict(X_test_scaled)
    gb_metrics = calculate_all_metrics(y_test, gb_preds)
    gb.save(str(models_dir / "gb_baseline.pkl"))
    gb_bundle = {"model": gb.model, "scaler": preprocessor.scaler, "features": config["model"]["features"], "sequence_length": sequence_length}
    with open(models_dir / "gb_inference_bundle.pkl", "wb") as f:
        pickle.dump(gb_bundle, f)
    results["gb"] = gb_metrics

    # Try XGBoost or LightGBM if available
    if _HAS_XGB:
        logger.info("XGBoost available — training XGBRegressor as benchmark")
        xgb_model = XGBRegressor(n_estimators=config.get("model", {}).get("xgb", {}).get("n_estimators", 100),
                                 max_depth=config.get("model", {}).get("xgb", {}).get("max_depth", 6),
                                 random_state=(config or {}).get("data", {}).get("seed", 42))
        xgb_model.fit(rf._flatten(X_train_scaled), y_train)
        xgb_preds = xgb_model.predict(rf._flatten(X_test_scaled))
        xgb_metrics = calculate_all_metrics(y_test, xgb_preds)
        with open(models_dir / "xgb_baseline.pkl", "wb") as f:
            pickle.dump(xgb_model, f)
        results["xgb"] = xgb_metrics
    elif _HAS_LGB:
        logger.info("LightGBM available — training LGBMRegressor as benchmark")
        lgbm_model = LGBMRegressor(n_estimators=config.get("model", {}).get("lgb", {}).get("n_estimators", 100),
                                   max_depth=config.get("model", {}).get("lgb", {}).get("max_depth", 6),
                                   random_state=(config or {}).get("data", {}).get("seed", 42))
        lgbm_model.fit(rf._flatten(X_train_scaled), y_train)
        lgbm_preds = lgbm_model.predict(rf._flatten(X_test_scaled))
        lgbm_metrics = calculate_all_metrics(y_test, lgbm_preds)
        with open(models_dir / "lgb_baseline.pkl", "wb") as f:
            pickle.dump(lgbm_model, f)
        results["lgb"] = lgbm_metrics
    else:
        logger.info("Neither XGBoost nor LightGBM available; skipping faster boosters.")

    # Save experiments metrics with provenance
    metrics_path = models_dir / "metrics_experiments.json"
    experiments = {
        "data_source": used_source,
        "config": config,
        "results": {k: {kk: float(v) for kk, v in mv.items()} for k, mv in results.items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)

    return experiments
