"""LSTM minimal sebagai lengan pembanding prediksi glukosa.

Arsitektur: Stacked LSTM (2 lapis) → Dense(1). Tanpa mekanisme atensi — kesederhanaan
ini disengaja supaya selisih akurasi terhadap RF dan GBM dapat diatribusikan ke
perbedaan arsitektur, bukan ke komponen tambahan.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from src.models.base_model import BaseGlucoseModel
from src.utils.metrics import calculate_all_metrics


class LSTMGlucoseModel(BaseGlucoseModel):
    """Stacked LSTM (no attention) for next-step glucose prediction."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        lstm_cfg = (config or {}).get("model", {}).get("lstm", {})
        self.units_1: int = lstm_cfg.get("units_1", 64)
        self.units_2: int = lstm_cfg.get("units_2", 32)
        self.dropout: float = lstm_cfg.get("dropout", 0.2)
        self.learning_rate: float = lstm_cfg.get("learning_rate", 0.001)
        self.epochs: int = lstm_cfg.get("epochs", 100)
        self.patience: int = lstm_cfg.get("patience", 10)
        # verbose: 0=diam, 1=progress bar (live), 2=satu baris/epoch
        self.verbose: int = lstm_cfg.get("verbose", 1)
        # keras model stored in self.model (set after build)

    def _build_keras_model(self, input_shape: tuple):
        """Construct and compile the Keras LSTM graph."""
        from tensorflow import keras

        inp = keras.Input(shape=input_shape, name="glucose_window")
        x = keras.layers.LSTM(self.units_1, return_sequences=True, name="lstm_1")(inp)
        x = keras.layers.Dropout(self.dropout, name="drop_1")(x)
        x = keras.layers.LSTM(self.units_2, return_sequences=False, name="lstm_2")(x)
        x = keras.layers.Dropout(self.dropout, name="drop_2")(x)
        out = keras.layers.Dense(1, name="glucose_pred")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        from tensorflow import keras

        X_train, y_train = self._validate_training_data(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val, y_val = self._validate_training_data(X_val, y_val)

        _, seq_len, n_features = X_train.shape
        self.model = self._build_keras_model((seq_len, n_features))

        # Adaptive batch size: avoids empty batches on very sparse datasets
        batch_size = max(8, min(32, len(X_train) // 10))

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=self.verbose,
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        hist = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.verbose,
        )

        self.is_trained = True

        # Collect training history
        history: Dict[str, float] = {
            "train_samples": float(len(y_train)),
            "epochs_run": float(len(hist.history["loss"])),
        }

        train_pred = self.model.predict(X_train, verbose=0).flatten()
        train_metrics = calculate_all_metrics(y_train, train_pred)
        history.update({f"train_{k}": float(v) for k, v in train_metrics.items()})

        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_pred = self.model.predict(X_val, verbose=0).flatten()
            val_metrics = calculate_all_metrics(y_val, val_pred)
            history.update({f"val_{k}": float(v) for k, v in val_metrics.items()})

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet")
        X = self._validate_prediction_data(X)
        return self.model.predict(X, verbose=0).flatten()

    def save(self, filepath: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save — train first")
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as Keras SavedModel format (directory)
        self.model.save(str(model_path))

    def load(self, filepath: str) -> None:
        from tensorflow import keras

        self.model = keras.models.load_model(filepath)
        self.is_trained = True


def train_lstm_from_config(
    config_path: str = "config.yaml",
    data_source: str = "auto",
    source: str = "CGM",
    horizon_min: Optional[float] = None,
    simpan_model: bool = False,
) -> Dict:
    """Latih LSTM sebagai lengan pembanding, lalu simpan metriknya.

    PERLAKUAN YANG IDENTIK. Pemuatan, pembersihan, rekayasa fitur, pembagian resmi,
    dan pembentukan jendela didelegasikan ke ``src.models.persiapan_data`` — modul
    yang sama yang dipakai jalur GBM dan RF. Sebelumnya jalur LSTM berbeda dari
    keduanya dalam empat hal sekaligus, sehingga selisih angkanya tidak dapat
    diatribusikan ke arsitektur:

    * memakai fitur dasar (4 kolom), bukan fitur hasil rekayasa (9 kolom);
    * tidak memakai target delta, sedangkan RF dan GBM memakainya;
    * membagi dengan menyisihkan dua pasien terakhir, bukan pembagian resmi;
    * memanggil ``create_sequences`` tanpa horizon, sehingga sebenarnya memprediksi
      satu langkah (+5 menit), bukan +30 menit seperti lengan lain.

    Argumen ``horizon_min`` dinyatakan dalam MENIT.

    JALUR SMBG WARISAN DICABUT. Parameter ``smbg_downsample`` menurunkan cadence CGM
    secara buatan untuk menirukan SMBG. Ia sudah tidak dapat dijalankan sejak parser
    disatukan, dan tidak pernah dipakai hasil mana pun di laporan. Skenario SMBG yang
    sah memakai modalitas FINGER_STICK sungguhan lewat ``--source FINGER_STICK``,
    yang parameter temporalnya diambil dari ``config.model.source_profiles``.
    """
    from src.models.persiapan_data import (
        bentuk_jendela,
        siapkan_data,
        simpan_prediksi,
    )

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

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    seluruh_metrik: Dict[str, Dict] = {}

    for horizon in siap.profil.horizons_min:
        X_train, y_train, anc_train, X_test, y_test, anc_test = bentuk_jendela(
            siap, horizon
        )

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError(
                f"Tidak ada jendela sah pada horizon {horizon:g} menit untuk {source}."
            )

        X_train_s, X_test_s = siap.preprocessor.normalize_data(X_train, X_test)

        y_train_fit = (y_train - anc_train) if siap.predict_delta else y_train

        model = LSTMGlucoseModel(config)
        model.train(X_train_s, y_train_fit)

        pred_fit = model.predict(X_test_s)
        y_pred = (pred_fit + anc_test) if siap.predict_delta else pred_fit

        metrics = calculate_all_metrics(y_test, y_pred)

        # Vektor prediksi disimpan supaya Clarke Error Grid dapat digambar tanpa
        # melatih ulang, dan dijamin berasal dari model yang SAMA dengan tabelnya.
        jalur_pred = simpan_prediksi("lstm", source, horizon, y_test, y_pred, anc_test)

        label = f"h{int(horizon)}m"
        prefix = f"lstm_{source.lower()}_{label}"

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
            # Pembagian resmi OhioT1DM bersifat TEMPORAL DALAM-PASIEN.
            "split": "official_dataset_split (temporal within-patient)",
            **{k: float(v) for k, v in metrics.items()},
        }

        metrics_path = results_dir / f"{prefix}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(catatan, f, indent=2)

        if simpan_model:
            bundle_dir = Path("models") / f"{prefix}_bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(bundle_dir / "keras_model"))

            with open(bundle_dir / "meta.pkl", "wb") as f:
                pickle.dump(
                    {
                        "scaler": siap.preprocessor.scaler,
                        "features": siap.feature_list,
                        "sequence_length": siap.profil.sequence_length,
                        "prediction_horizon_min": float(horizon),
                        "predict_delta": siap.predict_delta,
                        "glucose_source": source,
                    },
                    f,
                )

        seluruh_metrik[label] = catatan

        print("=" * 60)
        print(f"LSTM — {source} +{horizon:g} menit")
        print("=" * 60)
        print(f"Sampel latih : {len(y_train)}")
        print(f"Sampel uji   : {len(y_test)}")
        print("-" * 60)
        for key, value in metrics.items():
            print(f"{key:12s}: {value:.4f}")
        print("-" * 60)
        print(f"Metrik       : {metrics_path}")
        print(f"Prediksi     : {jalur_pred}")

    return seluruh_metrik


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latih LSTM sebagai lengan pembanding"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--data_source",
        default="auto",
        choices=["auto", "latest", "ohio_t1dm"],
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
        help="Simpan bundle Keras. Bawaan: hanya metrik.",
    )
    args = parser.parse_args()

    train_lstm_from_config(
        args.config,
        args.data_source,
        args.source,
        args.horizon_min,
        args.simpan_model,
    )


if __name__ == "__main__":
    main()
