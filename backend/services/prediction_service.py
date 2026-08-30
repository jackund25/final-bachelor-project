from pathlib import Path
import pickle
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.constants import risk_from_condition_class
from src.conformal import prediction_interval, coverage_achieved


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


# MODEL_FAMILIES DICABUT 25 Agustus 2026. Konstanta itu tidak pernah dirujuk satu
# baris pun sejak _load_horizons memilih artefak menurut glucose_source, dan dua
# dari empat lintasan yang didaftarkannya (rf_inference_bundle_h6/h12.pkl) bahkan
# tidak ada di models/. Artefak produksi yang benar-benar dimuat ada di
# _load_horizons dan _load_condition_classifier.

SOURCE_DEFAULT_SEQUENCE_LENGTH = {
    "CGM": 12,
    "FINGER_STICK": 8,
}


class PredictionService:

    def __init__(self):
        self.horizons = self._load_horizons("CGM")
        self.condition_classifier = self._load_condition_classifier()

    def _load_artifacts(self, path: str):
        bf = ROOT / path

        if not bf.exists():
            return None

        try:
            with open(bf, "rb") as f:
                b = pickle.load(f)
        except Exception as exc:
            logger.exception("Failed to load prediction artifact %s", bf)
            return None

        horizon_min = b.get("prediction_horizon_min")
        horizon_steps = b.get("prediction_horizon")
        if horizon_min is not None:
            horizon_min = float(horizon_min)
            horizon_steps = int(horizon_min / 5)

        return {
            "artifact_path": str(bf),
            "model": b["model"],
            "scaler": b.get("scaler"),
            "features": b.get(
                "features",
                ["glucose", "carbs", "insulin", "activity"],
            ),
            "sequence_length": int(
                b.get("sequence_length", 12)
            ),
            "horizon": int(horizon_steps or 6),
            "horizon_min": float(horizon_min or (int(horizon_steps or 6) * 5)),
            # Batas temporal yang dipakai SAAT PELATIHAN, dibawa ikut di dalam bundle.
            # Penjaga kelayakan memakai angka yang sama persis, sehingga jendela yang
            # ditolak pelatihan juga ditolak saat penyajian.
            "max_history_gap_min": b.get("max_history_gap_min"),
            "min_history_interval_min": float(
                b.get("min_history_interval_min", 0.0) or 0.0
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

    def _load_horizons(self, glucose_source: str):
        if glucose_source == "FINGER_STICK":
            files = [
                "models/gbm_finger_stick_h4h_seq8_inference_bundle.pkl",
            ]
        else:
            files = [
                "models/gbm_cgm_h30m_inference_bundle.pkl",
                "models/gbm_cgm_h60m_inference_bundle.pkl",
            ]

        arts = [self._load_artifacts(path) for path in files]
        arts = [
            art for art in arts
            if art is not None
            and art.get("glucose_source", glucose_source)
            == glucose_source
        ]
        return sorted(arts, key=lambda item: item["horizon"])

    def _load_condition_classifier(self):
        path = ROOT / "models" / "gbm_condition_classifier_h6.pkl"

        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
        except Exception as exc:
            logger.exception("Failed to load condition classifier %s", path)
            return None

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

    def _periksa_kelayakan(
        self,
        patient_df: pd.DataFrame,
        art: Dict[str, Any],
    ):
        """Apakah jendela terakhir sepadan dengan jendela pelatihan model ini?

        Aturannya TIDAK ditulis ulang di sini melainkan didelegasikan ke
        ``src.logbook.periksa_kelayakan_menit`` — satu-satunya tempat kriteria
        kelayakan hidup, supaya sisi pelatihan dan sisi penyajian tidak dapat
        menyimpang diam-diam.

        Batasnya diambil dari bundle model, bukan dari config: bundle merekam nilai
        yang benar-benar dipakai ketika model itu dilatih, sedangkan config dapat
        sudah berubah sesudahnya.
        """
        from src.logbook import periksa_kelayakan_menit

        return periksa_kelayakan_menit(
            patient_df,
            sequence_length=art["sequence_length"],
            max_gap_min=art.get("max_history_gap_min"),
            min_interval_min=art.get("min_history_interval_min", 0.0),
        )

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

        horizons = self._load_horizons(glucose_source)
        required = (
            horizons[0]["sequence_length"]
            if horizons
            else SOURCE_DEFAULT_SEQUENCE_LENGTH.get(glucose_source, 12)
        )

        if not horizons:
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

        # -----------------------------------------------------------------
        # PENJAGA KELAYAKAN TEMPORAL (M5.7 clinical readiness rules)
        #
        # Jumlah baris yang cukup TIDAK cukup. Pelatihan juga membuang jendela
        # yang memuat jeda terlalu lebar atau observasi terlalu rapat. Tanpa
        # penjaga yang sama di sini, model diberi jendela yang tidak pernah ia
        # lihat dan tetap mengeluarkan angka — kegagalan senyap yang paling
        # berbahaya pada alat bantu klinis.
        #
        # Batasnya dibaca dari BUNDLE, bukan dari config, supaya selalu sama
        # persis dengan yang dipakai saat model itu dilatih.
        # -----------------------------------------------------------------
        kelayakan = self._periksa_kelayakan(patient_df, horizons[0])

        if not kelayakan.boleh_diprediksi:
            current_glucose = float(patient_df["glucose"].iloc[-1])
            return {
                "status": "WINDOW_NOT_ELIGIBLE",
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
                "reason": kelayakan.alasan,
                "eligibility": {
                    "verdict": kelayakan.verdict,
                    "alasan": kelayakan.alasan,
                    "jeda_maks_menit": kelayakan.jeda_maks_menit,
                    "batas_jeda_menit": kelayakan.batas_jeda_menit,
                    "interval_min_menit": kelayakan.interval_min_menit,
                    "batas_interval_menit": kelayakan.batas_interval_menit,
                },
                "prediction_artifact_available": True,
                "minimum_required": required,
                "raw_reading_count": len(patient_df),
                "valid_reading_count": len(patient_df),
                "has_sufficient_history": True,
            }

        results: List[Dict[str, Any]] = []

        # Jendela terekayasa horizon UTAMA disimpan untuk dikembalikan ke pemanggil.
        # Ia memuat iob/cob yang sudah MELURUH menurut waktu (preprocessor.py), bukan
        # dosis mentah pada satu observasi. Sebelumnya jendela ini hanya hidup di
        # dalam metode ini lalu dibuang, sehingga clinical.py terpaksa memakai
        # `last.insulin` dan mengirimkannya ke LLM dengan label "insulin on board" —
        # dosis 5 unit enam jam lalu terbaca sebagai 5,00 unit insulin aktif.
        window_utama = None

        for art in horizons:

            window = self._build_window(
                patient_df,
                art,
            )

            if window_utama is None:
                window_utama = window

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
                "horizon_minutes": art["horizon_min"],
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
                    horizons[0],
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
            "prediction_30m": (
                primary["prediction"]
                if glucose_source == "CGM"
                else None
            ),
            "prediction_60m": (
                next(
                    (
                        item["prediction"]
                        for item in results
                        if item["horizon_minutes"] == 60
                    ),
                    None,
                )
                if glucose_source == "CGM"
                else None
            ),
            "condition": condition,
            "horizons": results,
            # Baris TERAKHIR jendela terekayasa horizon utama. Inilah vektor fitur
            # yang benar-benar dilihat model, jadi inilah pula yang harus dilihat
            # model bahasa — bukan kolom mentah dari baris logbook terakhir.
            "engineered_last": self._ringkas_fitur(window_utama),
        }

    @staticmethod
    def _ringkas_fitur(window) -> Dict[str, Any]:
        """Ambil fitur turunan yang bermakna klinis dari baris terakhir jendela.

        Hanya medan yang benar-benar ada yang dikembalikan; medan yang tidak dihitung
        preprocessor SENGAJA tidak diisi nilai bawaan. Mengisi 0.0 untuk fitur yang
        tak pernah dihitung persis mekanisme kegagalan T4: nilainya masuk ke konteks
        LLM sebagai fakta pasien, tanpa satu pun galat yang menandainya.
        """
        if window is None or len(window) == 0:
            return {}

        baris = window.iloc[-1]
        medan = ("iob", "cob", "glucose_rate", "glucose_delta", "time_since_prev_glucose")

        ringkas: Dict[str, Any] = {}
        for nama in medan:
            if nama in window.columns:
                nilai = baris[nama]
                if pd.notna(nilai):
                    ringkas[nama] = float(nilai)

        return ringkas