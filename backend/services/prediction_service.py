from pathlib import Path
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.constants import risk_from_condition_class
from src.conformal import prediction_interval, coverage_achieved


MODEL_FAMILIES = [
    (
        "GBM",
        [
            "models/gbm_inference_bundle_h6.pkl",
            "models/gbm_inference_bundle_h12.pkl",
        ],
    ),
    (
        "RF",
        [
            "models/rf_inference_bundle_h6.pkl",
            "models/rf_inference_bundle_h12.pkl",
        ],
    ),
]


class PredictionService:

    def __init__(self):
        self.horizons = self._load_horizons()

        if not self.horizons:
            raise RuntimeError("Prediction model belum tersedia.")

        self.condition_classifier = self._load_condition_classifier()

    def _load_artifacts(self, path: str):
        bf = Path(path)

        if not bf.exists():
            return None

        with open(bf, "rb") as f:
            b = pickle.load(f)

        return {
            "model": b["model"],
            "scaler": b.get("scaler"),
            "features": b.get(
                "features",
                ["glucose", "carbs", "insulin", "activity"],
            ),
            "sequence_length": int(
                b.get("sequence_length", 12)
            ),
            "horizon": int(
                b.get("prediction_horizon", 6)
            ),
            "use_engineered": bool(
                b.get("use_engineered", False)
            ),
            "predict_delta": bool(
                b.get("predict_delta", False)
            ),
            "feature_engineering": dict(
                b.get("feature_engineering", {})
            ),
            "std_method": str(
                b.get("std_method", "tree_variance")
            ),
            "std_models": b.get("std_models"),
            "model_family": str(
                b.get(
                    "model_family",
                    type(b["model"]).__name__,
                )
            ),
        }

    def _load_horizons(self):
        for _, files in MODEL_FAMILIES:
            arts = [
                self._load_artifacts(path)
                for path in files
            ]

            arts = [
                art for art in arts
                if art is not None
            ]

            if arts:
                return sorted(
                    arts,
                    key=lambda x: x["horizon"],
                )

        generic = self._load_artifacts(
            "models/rf_inference_bundle.pkl"
        )

        return [generic] if generic else []

    def _load_condition_classifier(self):
        path = Path("models") / "gbm_condition_classifier_h6.pkl"

        if not path.exists():
            return None

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        features = list(bundle.get("features", []))
        scaler = bundle.get("scaler")
        expected = getattr(scaler, "n_features_in_", None)

        if len(features) != 9 or expected != 9:
            raise RuntimeError(
                "Artifact condition classifier harus menggunakan 9 features. "
                f"Ditemukan features={len(features)}, scaler={expected}."
            )

        if bundle.get("glucose_source", bundle.get("source")) != "CGM":
            raise RuntimeError(
                "Artifact condition classifier harus berasal dari source CGM."
            )

        return bundle

    def _build_window(
        self,
        patient_df: pd.DataFrame,
        art: Dict[str, Any],
    ):
        seq = art["sequence_length"]

        if art["use_engineered"]:
            from src.data.preprocessor import DataPreprocessor

            feat_df = DataPreprocessor(
                {}
            ).engineer_features(
                patient_df,
                **art["feature_engineering"],
            )
        else:
            feat_df = patient_df

        return feat_df.tail(seq).reset_index(
            drop=True
        )

    def _predict_next(
        self,
        window_df: pd.DataFrame,
        art: Dict[str, Any],
    ) -> float:

        X = window_df[
            art["features"]
        ].values.astype(float)

        if art["scaler"] is not None:
            X = art["scaler"].transform(X)

        out = float(
            art["model"].predict(
                X.reshape(1, -1)
            )[0]
        )

        if art["predict_delta"]:
            out += float(
                window_df["glucose"].iloc[-1]
            )

        return out

    def _predict_uncertainty(
        self,
        window_df: pd.DataFrame,
        art: Dict[str, Any],
    ):

        X = window_df[
            art["features"]
        ].values.astype(float)

        if art["scaler"] is not None:
            X = art["scaler"].transform(X)

        Xf = X.reshape(1, -1)

        if art.get("std_method") == "quantile_spread":

            qm = art.get("std_models") or {}

            lo_model = qm.get("low")
            hi_model = qm.get("high")

            if lo_model is None or hi_model is None:
                return None

            from src.models.gbm_model import GAUSS_95_WIDTH

            width = (
                float(hi_model.predict(Xf)[0])
                - float(lo_model.predict(Xf)[0])
            )

            return max(width, 0.0) / GAUSS_95_WIDTH

        estimators = getattr(
            art["model"],
            "estimators_",
            None,
        )

        if not estimators:
            return None

        preds = np.array([
            tree.predict(Xf)[0]
            for tree in estimators
        ])

        return float(preds.std())

    def _predict_condition(
        self,
        window_df: pd.DataFrame,
        art: Dict[str, Any],
    ):
        """Predict future clinical condition using the classifier's own schema.

        The regression model and the clinical-condition classifier are separate
        artifacts. The classifier always owns its feature schema and scaler.
        """

        clf = self.condition_classifier

        if clf is None:
            return None

        scaler = clf.get("scaler")

        classifier_features = list(clf["features"])

        missing = [
            feature
            for feature in classifier_features
            if feature not in window_df.columns
        ]

        if missing:
            raise ValueError(
                "Feature clinical condition classifier tidak tersedia: "
                + ", ".join(missing)
            )

        X = window_df[
            classifier_features
        ].values.astype(float)

        if scaler is not None:
            expected = getattr(
                scaler,
                "n_features_in_",
                None,
            )

            if expected is not None and X.shape[1] != int(expected):
                raise RuntimeError(
                    "Clinical condition classifier menerima jumlah fitur yang salah: "
                    f"{X.shape[1]} vs scaler expects {int(expected)}."
                )

            X = scaler.transform(X)

        label = str(
            clf["model"].predict(
                X.reshape(1, -1)
            )[0]
        )

        return risk_from_condition_class(label)

    def predict(
        self,
        patient_df: pd.DataFrame,
        glucose_source: str = "CGM",
    ) -> Dict[str, Any]:

        glucose_source = str(glucose_source).upper()

        if "glucose_source" not in patient_df.columns:
            patient_df = patient_df.copy()
            patient_df["glucose_source"] = glucose_source
        else:
            patient_df = patient_df[
                patient_df["glucose_source"]
                .astype(str)
                .str.upper()
                .eq(glucose_source)
            ].copy()

        if patient_df.empty:
            raise ValueError(
                f"Tidak ada observation untuk source {glucose_source}."
            )

        required = self.horizons[0]["sequence_length"]

        if glucose_source != "CGM":
            current_glucose = float(patient_df["glucose"].iloc[-1])
            return {
                "status": "MODEL_UNAVAILABLE",
                "mode": "current_state",
                "prediction_available": False,
                "current_glucose": current_glucose,
                "prediction": None,
                "prediction_30m": None,
                "prediction_60m": None,
                "condition": None,
                "horizons": [],
                "history": {
                    "available": True,
                    "history_observations": len(patient_df),
                    "history_required": required,
                },
                "reason": (
                    f"No valid prediction artifact is configured for "
                    f"{glucose_source}."
                ),
                "prediction_artifact_available": False,
                "minimum_required": required,
                "raw_reading_count": len(patient_df),
                "valid_reading_count": len(patient_df),
                "has_sufficient_history": len(patient_df) >= required,
            }

        if len(patient_df) < required:
            current_glucose = float(patient_df["glucose"].iloc[-1])
            return {
                "status": "INSUFFICIENT_HISTORY",
                "mode": "current_state",
                "prediction_available": False,
                "current_glucose": current_glucose,
                "prediction": None,
                "prediction_30m": None,
                "prediction_60m": None,
                "condition": None,
                "horizons": [],
                "history": {
                    "available": True,
                    "history_observations": len(patient_df),
                    "history_required": required,
                },
                "reason": (
                    f"Insufficient historical {glucose_source} observations: "
                    f"{len(patient_df)}/{required}."
                ),
                "prediction_artifact_available": True,
                "minimum_required": required,
                "raw_reading_count": len(patient_df),
                "valid_reading_count": len(patient_df),
                "has_sufficient_history": False,
            }

        results: List[Dict[str, Any]] = []

        for art in self.horizons:

            window = self._build_window(
                patient_df,
                art,
            )

            prediction = self._predict_next(
                window,
                art,
            )

            std = self._predict_uncertainty(
                window,
                art,
            )

            interval = prediction_interval(
                prediction,
                std,
                art["horizon"],
                level=95,
            )

            results.append({
                "horizon_minutes": art["horizon"] * 5,
                "prediction": prediction,
                "std": std,
                "interval": interval,
                "coverage": coverage_achieved(
                    art["horizon"],
                    level=95,
                ),
                "model_family": art["model_family"],
            })

        primary = results[0]

        condition = None

        classifier_source = self.condition_classifier.get(
            "glucose_source",
            self.condition_classifier.get("source"),
        ) if self.condition_classifier else None

        if classifier_source == glucose_source:
            classifier_art = {
                "sequence_length": int(
                    self.condition_classifier["sequence_length"]
                ),
                "use_engineered": True,
                "feature_engineering": dict(
                    self.condition_classifier["feature_engineering"]
                ),
            }
            condition = self._predict_condition(
                self._build_window(
                    patient_df,
                    classifier_art,
                ),
                self.horizons[0],
            )

        current_glucose = float(
            patient_df["glucose"].iloc[-1]
        )

        return {
            "mode": "prediction",
            "prediction_available": True,
            "prediction_artifact_available": True,
            "minimum_required": required,
            "raw_reading_count": len(patient_df),
            "valid_reading_count": len(patient_df),
            "has_sufficient_history": True,
            "current_glucose": current_glucose,
            "prediction": primary["prediction"],
            "prediction_30m": primary["prediction"],
            "condition": condition,
            "horizons": results,
        }