"""Minimal LSTM model for glucose prediction (comparison baseline against RF).

Architecture: Stacked LSTM (2 layers) → Dense(1).
No attention mechanism — this is intentionally minimal so that any accuracy
gap vs the RF model is attributable to architecture differences, not auxiliary
components.
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
    smbg_downsample: bool = False,
) -> Dict:
    """Train the LSTM model using dataset and config, then save bundle + metrics.

    Args:
        config_path: Path to config.yaml.
        data_source: 'auto' | 'ohio_t1dm' | 'latest'.
        smbg_downsample: Legacy experiment path. If True, downsample CGM data to an
            SMBG-like cadence (config.data.smbg_interval_min). Not used by any result
            reported in the thesis: the SMBG scenario is evaluated on the REAL
            finger_stick timeline (data/raw/ohio_t1dm_smbg.csv), not on downsampled CGM.

    Returns:
        Evaluation metrics dict.
    """
    from src.data.loader import DiabetesDataLoader
    from src.data.preprocessor import DataPreprocessor

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DiabetesDataLoader(config["data"]["output_dir"])
    if data_source == "auto":
        primary = config.get("data", {}).get("primary_source", "ohio_t1dm")
        fallback = config.get("data", {}).get("fallback_source", "latest_generated")
        df, used_source = loader.load_preferred_dataset(primary, fallback)
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

    if smbg_downsample:
        interval = config.get("data", {}).get("smbg_interval_min", 240)
        df = preprocessor.downsample_smbg(df, interval_minutes=interval)
        # With 240-min cadence: seq_len=6 covers ~1 day of SMBG history
        sequence_length = config["model"].get("smbg_sequence_length", 6)
        mode_tag = "smbg"
    else:
        sequence_length = config["model"].get("sequence_length", 12)
        mode_tag = "cgm"

    patient_ids = sorted(df["patient_id"].unique().tolist())
    if len(patient_ids) < 2:
        raise ValueError("Need at least 2 patients for train/test split")

    test_patients = patient_ids[-2:]
    train_df, test_df = preprocessor.split_by_patient(df, test_patients)

    X_train, y_train = preprocessor.create_sequences(train_df, sequence_length=sequence_length)
    X_test, y_test = preprocessor.create_sequences(test_df, sequence_length=sequence_length)
    X_train_s, X_test_s = preprocessor.normalize_data(X_train, X_test)

    model = LSTMGlucoseModel(config)
    model.train(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    metrics = calculate_all_metrics(y_test, y_pred)

    # --- Save bundle (Keras model dir + scaler + metadata) ---
    bundle_dir = Path("models") / "lstm_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    keras_dir = bundle_dir / "keras_model"
    model.save(str(keras_dir))

    meta_bundle = {
        "scaler": preprocessor.scaler,
        "features": config["model"]["features"],
        "sequence_length": sequence_length,
        "mode": mode_tag,
    }
    with open(bundle_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta_bundle, f)

    # --- Save metrics ---
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"lstm_{mode_tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print("=" * 60)
    print(f"LSTM TRAINING ({mode_tag.upper()} mode, seq_len={sequence_length})")
    print("=" * 60)
    print(f"Train patients: {train_df['patient_id'].nunique()}")
    print(f"Test patients : {test_df['patient_id'].nunique()}")
    print(f"Train samples : {len(y_train)}")
    print(f"Test samples  : {len(y_test)}")
    print("-" * 60)
    for key, value in metrics.items():
        print(f"{key:12s}: {value:.4f}")
    print("-" * 60)
    print(f"Bundle saved  : {bundle_dir}")
    print(f"Metrics saved : {metrics_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal LSTM glucose model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--data_source",
        default="auto",
        choices=["auto", "latest", "ohio_t1dm"],
    )
    parser.add_argument(
        "--smbg",
        action="store_true",
        help="Downsample to SMBG cadence before training",
    )
    args = parser.parse_args()
    train_lstm_from_config(args.config, args.data_source, args.smbg)


if __name__ == "__main__":
    main()
