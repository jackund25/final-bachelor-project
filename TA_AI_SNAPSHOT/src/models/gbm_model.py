"""Gradient Boosting model untuk prediksi glukosa — prediktor produksi.

MENGAPA GBM, BUKAN RANDOM FOREST
--------------------------------
Rumusan Masalah 1 menuntut modul prediksi yang dapat dijalankan pada perangkat
kelas konsumen, dan Tujuan 1 menuntutnya efisien secara komputasi. Random Forest
adalah yang TERBURUK dari ketiga kandidat pada kriteria itu (T4.1, 6 fold):

    model   RMSE h6    ukuran       waktu latih
    RF      21,079     322,53 MB    708,1 dtk
    GBM     20,721       0,485 MB     3,4 dtk     <- 665x lebih kecil, 208x lebih cepat
    LSTM    20,602       0,124 MB    477   dtk

Keunggulan akurasi GBM atas RF konsisten antar-fold tetapi TIDAK bermakna secara
klinis (+0,358 mg/dL = 2,39% ambang ISO 15197; +0,976 mg/dL = 6,51% pada h12).
Yang menjadi alasan penggantian adalah ukuran dan kecepatannya, yang besarnya
tidak dapat diperdebatkan oleh definisi simpangan baku mana pun.

BATAS YANG WAJIB DINYATAKAN. Pada h12 GBM KALAH dari RF pada deteksi hipoglikemia
(sensitivitas 2,42% lawan 4,36%; hipoglikemia berat terlewat 957 lawan 933).
Klaim "GBM unggul di setiap dimensi" hanya berlaku pada h6. Pada produksi
kelemahan itu sebagian tertutup karena kondisi yang ditampilkan kepada dokter
berasal dari pengklasifikasi kondisi, bukan dari pengambangan nilai regresi
(lihat src/patient_state.py: predicted_condition menggantikan risk_level).

HIPERPARAMETER SENGAJA DIBIARKAN BAWAAN
---------------------------------------
T4.1 mengukur GBM dengan hiperparameter BAWAAN scikit-learn agar perbandingannya
terhadap RF yang juga tidak disetel tidak berat sebelah. Menyetel GBM di produksi
akan membuat model produksi BERBEDA dari model yang angkanya dilaporkan Bab VI.
Karena itu blok `model.gradient_boosting` di config.yaml hanya memuat `random_state`.

MENGAPA HistGradientBoostingRegressor, BUKAN GradientBoostingRegressor
----------------------------------------------------------------------
GradientBoostingRegressor membangun pohon persis dan tidak sepadan anggaran
komputasinya pada 100+ ribu jendela. Ini pilihan implementasi, bukan pilihan
keluarga model — sama seperti pada T4.1.

KETIDAKPASTIAN PER SAMPEL: MENGAPA MODEL KUANTIL
------------------------------------------------
Random Forest memberi ketidakpastian per sampel secara gratis lewat sebaran
antar-pohon (`estimators_`). HistGradientBoostingRegressor TIDAK punya
`estimators_`, sehingga jalur itu tidak tersedia dan interval prediksi akan
hilang sama sekali — padahal Tujuan 3 menyebut kalibrasi ketidakpastian secara
eksplisit.

Penggantinya: dua model kuantil tambahan (0,025 dan 0,975) yang melebar pada
wilayah masukan yang sebarannya lebar, lalu

    sigma(x) = (q975(x) - q025(x)) / 3,92

yaitu simpangan baku setara-Gauss. Nilai 3,92 = 2 x 1,96.

Yang PENTING untuk keabsahan: jaminan cakupan konformal TIDAK bergantung pada
bagaimana sigma dipilih. Sigma hanya penskala heuristik; kuantil konformal q yang
dihitung pada himpunan kalibrasi menyerap skalanya, dan cakupannya tetap dijamin
secara distribution-free (Angelopoulos & Bates, 2021). Sebaran antar-pohon pada RF
juga heuristik dengan status yang persis sama — tidak ada yang diturunkan mutunya
di sini, hanya sumber heuristiknya yang berganti.

Biaya: dua pelatihan tambahan, masing-masing beberapa detik.
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

# Kuantil untuk penaksir sigma. Simetris terhadap median dan sepadan dengan
# tingkat 95% yang dipakai UI, sehingga sigma-nya terbaca sebagai "setara std".
QUANTILE_LOW = 0.025
QUANTILE_HIGH = 0.975
# 3,92 = 2 x 1,96 — pengubah lebar interval 95% Gauss menjadi satu simpangan baku.
GAUSS_95_WIDTH = 3.92

# Penanda cara sigma dihitung, disimpan di bundle agar aplikasi tidak perlu
# menebak dari jenis model. Bundle RF lama tidak punya medan ini dan ditangani
# lewat nilai bawaan pada pemuat.
STD_METHOD_QUANTILE = "quantile_spread"
STD_METHOD_TREE = "tree_variance"


class GBMGlucoseModel(BaseGlucoseModel):
    """Histogram Gradient Boosting di atas jendela waktu yang diratakan.

    Menyediakan ``predict`` untuk nilai titik dan ``predict_std`` untuk
    ketidakpastian per sampel dari sepasang model kuantil.
    """

    def __init__(self, config: Optional[Dict] = None, with_uncertainty: bool = True):
        super().__init__(config)
        cfg = config or {}
        gbm_cfg = cfg.get("model", {}).get("gradient_boosting", {})
        seed = gbm_cfg.get("random_state", cfg.get("data", {}).get("seed", 42))

        # Bawaan scikit-learn kecuali random_state — lihat catatan modul.
        self.model = HistGradientBoostingRegressor(random_state=seed)

        self.with_uncertainty = with_uncertainty
        if with_uncertainty:
            self.model_q_low = HistGradientBoostingRegressor(
                loss="quantile", quantile=QUANTILE_LOW, random_state=seed
            )
            self.model_q_high = HistGradientBoostingRegressor(
                loss="quantile", quantile=QUANTILE_HIGH, random_state=seed
            )
        else:
            self.model_q_low = None
            self.model_q_high = None

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
        if self.with_uncertainty:
            self.model_q_low.fit(X_train_flat, y_train)
            self.model_q_high.fit(X_train_flat, y_train)
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
        return self.model.predict(self._flatten(X))

    def predict_std(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Sigma per sampel dari rentang kuantil. ``None`` bila model kuantil tak ada.

        Dipangkas pada nol: kuantil yang dilatih terpisah tidak dijamin berurutan
        (*quantile crossing*), sehingga selisihnya dapat negatif pada sebagian kecil
        masukan. Membiarkannya negatif akan menghasilkan interval terbalik.
        """
        if not self.is_trained or self.model_q_low is None or self.model_q_high is None:
            return None
        X = self._validate_prediction_data(X)
        Xf = self._flatten(X)
        lo = self.model_q_low.predict(Xf)
        hi = self.model_q_high.predict(Xf)
        return np.maximum(hi - lo, 0.0) / GAUSS_95_WIDTH

    def save(self, filepath: str) -> None:
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True


def train_gbm_from_config(
    config_path: str = "config.yaml",
    data_source: str = "auto",
    horizon: Optional[int] = None,
) -> Dict:
    """Latih GBM pada satu horizon, lalu simpan bundle inferensi + metrik.

    Alurnya sengaja identik dengan ``train_random_forest_from_config``: sumber data,
    praproses, segmentasi jeda, pembagian pasien, penskalaan, dan target delta sama
    persis. Yang berbeda hanya keluarga modelnya, supaya angka yang dihasilkan
    sebanding dengan angka RF yang sudah tersimpan.
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
    max_gap_steps = config["model"].get("max_gap_steps")
    max_interp = config["model"].get("max_interpolate_steps")
    cadence_min = config.get("data", {}).get("sampling_interval_min", 5)
    df = preprocessor.handle_missing_values(df, max_interpolate_steps=max_interp)

    sequence_length = config["model"].get("sequence_length", 12)
    default_horizon = config["model"].get("default_horizon", 1)
    if horizon is None:
        horizon = default_horizon

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

    seq_kwargs = {"max_gap_steps": max_gap_steps, "source_interval_min": cadence_min}
    X_train, y_train, anc_train = preprocessor.create_sequences(
        train_df, sequence_length, horizon, return_anchor=True, **seq_kwargs)
    X_test, y_test, anc_test = preprocessor.create_sequences(
        test_df, sequence_length, horizon, return_anchor=True, **seq_kwargs)

    X_train_scaled, X_test_scaled = preprocessor.normalize_data(X_train, X_test)

    y_train_fit = (y_train - anc_train) if predict_delta else y_train

    model = GBMGlucoseModel(config)
    model.train(X_train_scaled, y_train_fit)
    y_pred = model.predict(X_test_scaled)
    if predict_delta:
        y_pred = y_pred + anc_test

    metrics = calculate_all_metrics(y_test, y_pred)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(models_dir / f"gbm_baseline_h{horizon}.pkl"))

    # Skema bundle SAMA dengan bundle RF, ditambah dua medan ketidakpastian.
    # Kesamaan itu disengaja: aplikasi memuat keduanya lewat pemuat yang sama.
    bundle = {
        "model": model.model,
        "scaler": preprocessor.scaler,
        "features": feature_list,
        "sequence_length": sequence_length,
        "prediction_horizon": horizon,
        "use_engineered": use_engineered,
        "predict_delta": predict_delta,
        "feature_engineering": fe_params,
        # Ketidakpastian per sampel — lihat catatan modul.
        "std_method": STD_METHOD_QUANTILE,
        "std_models": {"low": model.model_q_low, "high": model.model_q_high},
        "std_quantiles": [QUANTILE_LOW, QUANTILE_HIGH],
        "model_family": "HistGradientBoostingRegressor",
    }
    with open(models_dir / f"gbm_inference_bundle_h{horizon}.pkl", "wb") as f:
        pickle.dump(bundle, f)
    with open(models_dir / f"gbm_metrics_h{horizon}.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    ukuran_mb = (models_dir / f"gbm_inference_bundle_h{horizon}.pkl").stat().st_size / 1e6

    print("=" * 60)
    print(f"GRADIENT BOOSTING — horizon {horizon} langkah (+{horizon * 5} menit)")
    print("=" * 60)
    print(f"Train patients: {train_df['patient_id'].nunique()}")
    print(f"Test patients : {test_df['patient_id'].nunique()}")
    print(f"Train samples : {len(y_train)}")
    print(f"Test samples  : {len(y_test)}")
    print(f"Ukuran bundle : {ukuran_mb:.3f} MB")
    print("-" * 60)
    for key, value in metrics.items():
        print(f"{key:12s}: {value:.4f}")
    print("-" * 60)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih prediktor Gradient Boosting")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_source", default="auto",
                        choices=["auto", "latest", "ohio_t1dm"])
    parser.add_argument("--horizon", type=int, default=None,
                        help="Horizon (langkah). Default: semua config.model.prediction_horizons")
    args = parser.parse_args()

    if args.horizon is not None:
        train_gbm_from_config(args.config, args.data_source, args.horizon)
    else:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        horizons = cfg["model"].get("prediction_horizons",
                                    [cfg["model"].get("default_horizon", 1)])
        for h in horizons:
            train_gbm_from_config(args.config, args.data_source, h)


if __name__ == "__main__":
    main()
