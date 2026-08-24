"""
Gradient Boosting model for modality-aware glucose prediction.

Final M5.4 design
-----------------
CGM:
    +30 min and +60 min fixed horizons.

FINGER_STICK / SMBG:
    8 valid historical observations
    target approximately +4 hours
    accepted target window: 180–300 minutes
    history interval <5 minutes excluded

Official OhioT1DM train/test split is respected.

The model remains HistGradientBoostingRegressor with quantile models
for uncertainty estimation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.base_model import BaseGlucoseModel
from src.utils.metrics import calculate_all_metrics


QUANTILE_LOW = 0.025
QUANTILE_HIGH = 0.975
GAUSS_95_WIDTH = 3.92

STD_METHOD_QUANTILE = "quantile_spread"


class GBMGlucoseModel(BaseGlucoseModel):
    """HistGradientBoostingRegressor over flattened temporal windows."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        with_uncertainty: bool = True,
    ):
        super().__init__(config)

        cfg = config or {}

        gbm_cfg = cfg.get(
            "model",
            {}
        ).get(
            "gradient_boosting",
            {},
        )

        seed = gbm_cfg.get(
            "random_state",
            cfg.get(
                "data",
                {}
            ).get(
                "seed",
                42,
            ),
        )

        self.model = (
            HistGradientBoostingRegressor(
                random_state=seed
            )
        )

        self.with_uncertainty = (
            with_uncertainty
        )

        if with_uncertainty:
            self.model_q_low = (
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=QUANTILE_LOW,
                    random_state=seed,
                )
            )

            self.model_q_high = (
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=QUANTILE_HIGH,
                    random_state=seed,
                )
            )
        else:
            self.model_q_low = None
            self.model_q_high = None

    @staticmethod
    def _flatten(
        X: np.ndarray,
    ) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(
                "Expected 3D input "
                "(samples, sequence_length, n_features)"
            )

        n_samples, seq_len, n_features = (
            X.shape
        )

        return X.reshape(
            n_samples,
            seq_len * n_features,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        X_train, y_train = (
            self._validate_training_data(
                X_train,
                y_train,
            )
        )

        X_train_flat = self._flatten(
            X_train
        )

        self.model.fit(
            X_train_flat,
            y_train,
        )

        if self.with_uncertainty:
            self.model_q_low.fit(
                X_train_flat,
                y_train,
            )

            self.model_q_high.fit(
                X_train_flat,
                y_train,
            )

        self.is_trained = True

        history = {
            "train_samples": float(
                len(y_train)
            )
        }

        train_pred = self.model.predict(
            X_train_flat
        )

        train_metrics = (
            calculate_all_metrics(
                y_train,
                train_pred,
            )
        )

        history.update(
            {
                f"train_{k}": float(v)
                for k, v in train_metrics.items()
            }
        )

        if (
            X_val is not None
            and y_val is not None
            and len(y_val) > 0
        ):
            val_pred = self.predict(
                X_val
            )

            val_metrics = (
                calculate_all_metrics(
                    y_val,
                    val_pred,
                )
            )

            history.update(
                {
                    f"val_{k}": float(v)
                    for k, v in val_metrics.items()
                }
            )

        return history

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError(
                "Model is not trained yet"
            )

        X = self._validate_prediction_data(
            X
        )

        return self.model.predict(
            self._flatten(X)
        )

    def predict_std(
        self,
        X: np.ndarray,
    ) -> Optional[np.ndarray]:
        if (
            not self.is_trained
            or self.model_q_low is None
            or self.model_q_high is None
        ):
            return None

        X = self._validate_prediction_data(
            X
        )

        X_flat = self._flatten(
            X
        )

        lo = self.model_q_low.predict(
            X_flat
        )

        hi = self.model_q_high.predict(
            X_flat
        )

        return (
            np.maximum(
                hi - lo,
                0.0,
            )
            / GAUSS_95_WIDTH
        )

    def save(
        self,
        filepath: str,
    ) -> None:
        model_path = Path(
            filepath
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            model_path,
            "wb",
        ) as f:
            pickle.dump(
                self.model,
                f,
            )

    def load(
        self,
        filepath: str,
    ) -> None:
        with open(
            filepath,
            "rb",
        ) as f:
            self.model = pickle.load(
                f
            )

        self.is_trained = True


def _resolve_source_config(
    config: Dict,
    source: str,
) -> Dict:
    return (
        config
        .get("model", {})
        .get("source_profiles", {})
        .get(source, {})
    )


def train_gbm_from_config(
    config_path: str = "config.yaml",
    data_source: str = "auto",
    source: str = "CGM",
    horizon_min: Optional[float] = None,
) -> Dict:
    """
    Train one modality-specific GBM.

    CGM:
        fixed horizon using target_tolerance_min.

    FINGER_STICK:
        window target using
        min_target_horizon_min / max_target_horizon_min.
    """
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    source = source.upper()

    if source not in {
        "CGM",
        "FINGER_STICK",
    }:
        raise ValueError(
            "source must be CGM or FINGER_STICK"
        )

    loader = DiabetesDataLoader(
        config["data"]["output_dir"]
    )

    if data_source in (
        "auto",
        "ohio_t1dm",
    ):
        df = loader.load_csv(
            config["data"].get(
                "unified_dataset",
                "ohio_t1dm.csv",
            )
        )

        used_source = (
            "ohio_t1dm_unified"
        )
    else:
        df = loader.load_latest_dataset()
        used_source = (
            "latest_generated"
        )

    print(
        f"Data source : {used_source}"
    )

    required_metadata = [
        "glucose_source",
        "dataset_split",
    ]

    missing = [
        c
        for c in required_metadata
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Unified OhioT1DM dataset is missing metadata: "
            f"{missing}"
        )

    df = df[
        df["glucose_source"]
        == source
    ].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for glucose_source={source}"
        )

    df = df.sort_values(
        [
            "patient_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    preprocessor = DataPreprocessor(
        config
    )

    max_interp = config[
        "model"
    ].get(
        "max_interpolate_steps"
    )

    df = preprocessor.handle_missing_values(
        df,
        max_interpolate_steps=max_interp,
    )

    fe_params = config[
        "model"
    ].get(
        "feature_engineering",
        {},
    )

    df = preprocessor.engineer_features(
        df,
        **fe_params,
    )

    feature_list = config[
        "model"
    ].get(
        "engineered_features",
        config["model"]["features"],
    )

    preprocessor.feature_columns = list(
        feature_list
    )

    source_cfg = _resolve_source_config(
        config,
        source,
    )

    if horizon_min is None:
        horizons = source_cfg.get(
            "prediction_horizons_min",
            [30],
        )
    else:
        horizons = [
            float(horizon_min)
        ]

    sequence_length = int(
        source_cfg.get(
            "sequence_length",
            config["model"].get(
                "sequence_length",
                12,
            ),
        )
    )

    tolerance = float(
        source_cfg.get(
            "target_tolerance_min",
            2.5,
        )
    )

    max_history_gap = source_cfg.get(
        "max_history_gap_min"
    )

    min_history_interval = float(
        source_cfg.get(
            "min_history_interval_min",
            0.0,
        )
    )

    min_target_horizon = source_cfg.get(
        "min_target_horizon_min"
    )

    max_target_horizon = source_cfg.get(
        "max_target_horizon_min"
    )

    train_df = df[
        df["dataset_split"]
        == "train"
    ].copy()

    test_df = df[
        df["dataset_split"]
        == "test"
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Both official train and test splits "
            "must contain rows."
        )

    all_metrics = {}

    for current_horizon in horizons:
        current_horizon = float(
            current_horizon
        )

        X_train, y_train, anc_train = (
            preprocessor.create_time_horizon_sequences(
                train_df,
                sequence_length=sequence_length,
                horizon_min=current_horizon,
                target_tolerance_min=tolerance,
                max_history_gap_min=max_history_gap,
                return_anchor=True,
                min_target_horizon_min=min_target_horizon,
                max_target_horizon_min=max_target_horizon,
                min_history_interval_min=min_history_interval,
            )
        )

        X_test, y_test, anc_test = (
            preprocessor.create_time_horizon_sequences(
                test_df,
                sequence_length=sequence_length,
                horizon_min=current_horizon,
                target_tolerance_min=tolerance,
                max_history_gap_min=max_history_gap,
                return_anchor=True,
                min_target_horizon_min=min_target_horizon,
                max_target_horizon_min=max_target_horizon,
                min_history_interval_min=min_history_interval,
            )
        )

        X_train_scaled, X_test_scaled = (
            preprocessor.normalize_data(
                X_train,
                X_test,
            )
        )

        predict_delta = bool(
            config["model"].get(
                "predict_delta",
                True,
            )
        )

        y_train_fit = (
            y_train - anc_train
            if predict_delta
            else y_train
        )

        model = GBMGlucoseModel(
            config
        )

        history = model.train(
            X_train_scaled,
            y_train_fit,
        )

        pred_fit = model.predict(
            X_test_scaled
        )

        y_pred = (
            pred_fit + anc_test
            if predict_delta
            else pred_fit
        )

        metrics = calculate_all_metrics(
            y_test,
            y_pred,
        )

        # Vektor prediksi disimpan supaya Clarke Error Grid dapat digambar tanpa
        # melatih ulang, dan dijamin berasal dari model yang SAMA dengan tabelnya.
        # Impor lokal agar modul ini tidak bergantung siklik pada persiapan_data.
        from src.models.persiapan_data import simpan_prediksi

        simpan_prediksi(
            "gbm",
            source,
            current_horizon,
            y_test,
            y_pred,
            anc_test,
        )

        label = (
            f"h{int(current_horizon)}m"
        )

        all_metrics[label] = {
            "source": source,
            "horizon_min": current_horizon,
            "sequence_length": sequence_length,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "target_tolerance_min": tolerance,
            "min_target_horizon_min": min_target_horizon,
            "max_target_horizon_min": max_target_horizon,
            "min_history_interval_min": min_history_interval,
            "max_history_gap_min": max_history_gap,
            **{
                k: float(v)
                for k, v in metrics.items()
            },
        }

        models_dir = Path(
            "models"
        )
        models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if source == "FINGER_STICK":
            stem = (
                "gbm_finger_stick_"
                "h4h_seq8"
            )
        else:
            stem = (
                f"gbm_{source.lower()}_"
                f"h{int(current_horizon)}m"
            )

        model.save(
            str(
                models_dir
                / f"{stem}.pkl"
            )
        )

        bundle = {
            "model": model.model,
            "scaler": preprocessor.scaler,
            "features": feature_list,
            "sequence_length": sequence_length,
            "prediction_horizon_min": current_horizon,
            "target_tolerance_min": tolerance,
            "min_target_horizon_min": min_target_horizon,
            "max_target_horizon_min": max_target_horizon,
            "min_history_interval_min": min_history_interval,
            "max_history_gap_min": max_history_gap,
            "glucose_source": source,
            "dataset_split": "official_train_test",
            "use_engineered": True,
            "predict_delta": predict_delta,
            "feature_engineering": fe_params,
            "std_method": STD_METHOD_QUANTILE,
            "std_models": {
                "low": model.model_q_low,
                "high": model.model_q_high,
            },
            "std_quantiles": [
                QUANTILE_LOW,
                QUANTILE_HIGH,
            ],
            "model_family": (
                "HistGradientBoostingRegressor"
            ),
            "training_design": (
                "SMBG_8_observations_to_"
                "approximately_4h"
                if source == "FINGER_STICK"
                else "CGM_fixed_horizon"
            ),
        }

        with open(
            models_dir
            / f"{stem}_inference_bundle.pkl",
            "wb",
        ) as f:
            pickle.dump(
                bundle,
                f,
            )

        with open(
            models_dir
            / f"{stem}_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                all_metrics[label],
                f,
                indent=2,
            )

        print("=" * 70)
        print(
            f"GBM {source} — target ~"
            f"{current_horizon:g} menit"
        )
        print("=" * 70)
        print(
            f"Train rows        : {len(train_df):,}"
        )
        print(
            f"Test rows         : {len(test_df):,}"
        )
        print(
            f"Train samples     : {len(y_train):,}"
        )
        print(
            f"Test samples      : {len(y_test):,}"
        )
        print(
            f"History length    : {sequence_length}"
        )

        if min_target_horizon is not None:
            print(
                "Target window     : "
                f"{min_target_horizon:g}–"
                f"{max_target_horizon:g} menit"
            )
        else:
            print(
                f"Target tolerance  : ±{tolerance:g} menit"
            )

        print(
            f"Min history gap   : "
            f"{min_history_interval:g} menit"
        )

        for key, value in metrics.items():
            print(
                f"{key:14s}: {value:.4f}"
            )

        print("-" * 70)

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train modality-specific, "
            "elapsed-time-aware GBM"
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--data_source",
        default="auto",
        choices=[
            "auto",
            "latest",
            "ohio_t1dm",
        ],
    )

    parser.add_argument(
        "--source",
        default="CGM",
        choices=[
            "CGM",
            "FINGER_STICK",
        ],
    )

    parser.add_argument(
        "--horizon_min",
        type=float,
        default=None,
        help=(
            "Override prediction horizon. "
            "Otherwise source profile is used."
        ),
    )

    args = parser.parse_args()

    train_gbm_from_config(
        config_path=args.config,
        data_source=args.data_source,
        source=args.source,
        horizon_min=args.horizon_min,
    )


if __name__ == "__main__":
    main()