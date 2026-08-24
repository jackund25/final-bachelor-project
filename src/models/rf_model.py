"""Random Forest sebagai lengan pembanding prediksi glukosa.

POSISI DALAM PENELITIAN
-----------------------
RF BUKAN model produksi. Jalur produksi memakai GBM (lihat catatan pemilihan pada
config.yaml). RF dipertahankan sebagai lengan pembanding karena pembahasan menuntut
penjelasan mengapa tiga keluarga model diuji dan apa kelebihan-kekurangan masing-masing.

Karena itu modul ini SECARA BAWAAN tidak menyimpan artefak modelnya. Satu berkas
RandomForest terlatih pada konfigurasi ini berukuran ratusan megabita, dan yang
dibutuhkan pembahasan hanyalah metriknya. Gunakan ``--simpan-model`` bila artefaknya
memang diperlukan.

PERLAKUAN YANG IDENTIK
----------------------
Pemuatan, pembersihan, rekayasa fitur, pembagian resmi, dan pembentukan jendela
seluruhnya didelegasikan ke ``src.models.persiapan_data`` — modul yang sama yang
dipakai jalur GBM. Lengan pembanding yang menerima perlakuan berbeda tidak dapat
dibandingkan, dan perbedaan semacam itu tidak akan terlihat dari angka mana pun.

Ekuivalensinya diuji, bukan diasumsikan: lihat
``tests/test_persiapan_data.py::test_persiapan_bersama_mereproduksi_jendela_gbm``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import BaseGlucoseModel
from src.models.persiapan_data import bentuk_jendela, siapkan_data, simpan_prediksi
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
    source: str = "CGM",
    horizon_min: Optional[float] = None,
    simpan_model: bool = False,
) -> Dict:
    """Latih RF pada satu modalitas, untuk seluruh horizon profilnya.

    Argumen ``horizon_min`` dinyatakan dalam MENIT, bukan langkah. API berbasis
    langkah dicabut ketika parser OhioT1DM disatukan: jendela kini dibentuk dari
    waktu nyata sehingga modalitas dengan cadence tak teratur (finger-stick) tidak
    lagi diperlakukan seolah-olah CGM 5-menit.

    Mengembalikan peta ``{"h30m": {...metrik...}, ...}``.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    siap = siapkan_data(
        config,
        sumber=source,
        data_source=data_source,
        horizon_min=horizon_min,
    )

    print(f"Data source : {siap.sumber_data}")
    print(f"Modalitas   : {source}")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    seluruh_metrik: Dict[str, Dict] = {}

    for horizon in siap.profil.horizons_min:
        X_train, y_train, anc_train, X_test, y_test, anc_test = bentuk_jendela(
            siap, horizon
        )

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError(
                f"Tidak ada jendela sah pada horizon {horizon:g} menit untuk {source}."
            )

        X_train_scaled, X_test_scaled = siap.preprocessor.normalize_data(
            X_train, X_test
        )

        y_train_fit = (y_train - anc_train) if siap.predict_delta else y_train

        model = RandomForestGlucoseModel(config)
        model.train(X_train_scaled, y_train_fit)

        pred_fit = model.predict(X_test_scaled)
        y_pred = (pred_fit + anc_test) if siap.predict_delta else pred_fit

        metrics = calculate_all_metrics(y_test, y_pred)

        # Vektor prediksi disimpan supaya Clarke Error Grid dapat digambar tanpa
        # melatih ulang, dan dijamin berasal dari model yang SAMA dengan tabelnya.
        jalur_pred = simpan_prediksi("rf", source, horizon, y_test, y_pred, anc_test)

        label = f"h{int(horizon)}m"
        prefix = f"rf_{source.lower()}_{label}"

        catatan = {
            "source": source,
            "horizon_min": float(horizon),
            "sequence_length": siap.profil.sequence_length,
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "target_tolerance_min": siap.profil.target_tolerance_min,
            "max_history_gap_min": siap.profil.max_history_gap_min,
            "min_history_interval_min": siap.profil.min_history_interval_min,
            "min_target_horizon_min": siap.profil.min_target_horizon_min,
            "max_target_horizon_min": siap.profil.max_target_horizon_min,
            # Pembagian resmi OhioT1DM bersifat TEMPORAL DALAM-PASIEN. Dicatat pada
            # artefaknya supaya angka ini tidak pernah terbaca sebagai hasil
            # pembagian lintas-pasien.
            "split": "official_dataset_split (temporal within-patient)",
            **{k: float(v) for k, v in metrics.items()},
        }

        with open(models_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(catatan, f, indent=2)

        if simpan_model:
            model.save(str(models_dir / f"{prefix}.pkl"))

            bundle = {
                "model": model.model,
                "scaler": siap.preprocessor.scaler,
                "features": siap.feature_list,
                "sequence_length": siap.profil.sequence_length,
                "prediction_horizon_min": float(horizon),
                "predict_delta": siap.predict_delta,
                "feature_engineering": siap.feature_engineering,
                "glucose_source": source,
            }
            with open(models_dir / f"{prefix}_inference_bundle.pkl", "wb") as f:
                pickle.dump(bundle, f)

        seluruh_metrik[label] = catatan

        print("=" * 60)
        print(f"RANDOM FOREST — {source} +{horizon:g} menit")
        print("=" * 60)
        print(f"Sampel latih : {len(y_train)}")
        print(f"Sampel uji   : {len(y_test)}")
        print("-" * 60)
        for key, value in metrics.items():
            print(f"{key:12s}: {value:.4f}")
        print("-" * 60)
        print(f"Prediksi     : {jalur_pred}")

    if not simpan_model:
        print(
            "Artefak model TIDAK disimpan (bawaan). RF adalah lengan pembanding; "
            "metriknya sudah cukup. Pakai --simpan-model bila diperlukan."
        )

    return seluruh_metrik


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latih Random Forest sebagai lengan pembanding"
    )
    parser.add_argument("--config", default="config.yaml", help="Path ke YAML config")
    parser.add_argument(
        "--data_source",
        default="auto",
        choices=["auto", "latest", "ohio_t1dm"],
        help="Sumber data (auto: dataset gabungan OhioT1DM)",
    )
    parser.add_argument(
        "--source",
        default="CGM",
        choices=["CGM", "FINGER_STICK"],
        help="Modalitas observasi glukosa",
    )
    parser.add_argument(
        "--horizon-min",
        type=float,
        default=None,
        help=(
            "Horizon prediksi dalam MENIT. Bawaan: seluruh horizon pada "
            "config.model.source_profiles[<source>].prediction_horizons_min"
        ),
    )
    parser.add_argument(
        "--simpan-model",
        action="store_true",
        help="Simpan artefak .pkl (ratusan MB). Bawaan: hanya metrik.",
    )
    args = parser.parse_args()

    train_random_forest_from_config(
        args.config,
        args.data_source,
        args.source,
        args.horizon_min,
        args.simpan_model,
    )


if __name__ == "__main__":
    main()
