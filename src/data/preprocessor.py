"""
Data Preprocessing utilities
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.contracts import assert_feature_set, validate_data_contract

logger = logging.getLogger(__name__)


def _decay_accumulate(values: np.ndarray, decay: float) -> np.ndarray:
    """Akumulasi peluruhan eksponensial first-order: out[t] = values[t] + decay*out[t-1].

    Dipakai untuk Insulin-on-Board (IOB) dan Carbs-on-Board (COB): kejadian masa lalu
    masih "aktif" namun meluruh seiring waktu (model fisiologis sederhana).
    """
    out = np.zeros(len(values), dtype=float)
    acc = 0.0
    for i, v in enumerate(values):
        acc = float(v) + decay * acc
        out[i] = acc
    return out


class DataPreprocessor:
    """
    Preprocess raw data for model training
    """
    
    def __init__(self, config):
        self.config = config or {}
        self.scaler = StandardScaler()
        model_config = self.config.get('model', {})
        self.feature_columns = list(model_config.get('features', ['glucose', 'carbs', 'insulin', 'activity', 'stress']))

    def _require_columns(self, df: pd.DataFrame, columns: List[str]) -> None:
        assert_feature_set(df, columns)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values via interpolation
        
        Args:
            df: Input DataFrame
            
        Returns:
            df_clean: Cleaned DataFrame
        """
        logger.info("Handling missing values...")

        df = validate_data_contract(df)
        self._require_columns(df, ['patient_id', 'timestamp', *self.feature_columns])

        df = df.copy()
        df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
        
        missing_before = df.isnull().sum().sum()
        
        # Interpolate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        # Fill remaining NaNs with forward fill
        df = df.ffill().bfill()
        
        missing_after = df.isnull().sum().sum()
        
        logger.info(f"Missing values: {missing_before} → {missing_after}")
        
        return df
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        insulin_tau_min: float = 240.0,
        carbs_tau_min: float = 180.0,
        trend_steps: int = 3,
        source_interval_min: float = 5.0,
    ) -> pd.DataFrame:
        """Tambah fitur turunan berbasis fisiologi (dihitung per pasien, tanpa leakage).

        - iob  : Insulin-on-Board, peluruhan ~insulin_tau_min (default 4 jam) dari bolus.
        - cob  : Carbs-on-Board, peluruhan ~carbs_tau_min (default 3 jam) dari makanan.
        - glucose_delta : tren glukosa (selisih `trend_steps` langkah terakhir).
        - hour_sin/hour_cos : waktu dalam hari (menangkap pola diurnal / dawn phenomenon).

        Referensi: model glukosa-insulin-karbohidrat (Bergman minimal model; UVA/Padova).
        """
        df = validate_data_contract(df)
        df = df.copy().sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

        insulin_decay = float(np.exp(-source_interval_min / insulin_tau_min))
        carbs_decay = float(np.exp(-source_interval_min / carbs_tau_min))

        parts = []
        for _, g in df.groupby("patient_id", sort=False):
            g = g.copy()
            insulin_src = g["bolus_dose"] if "bolus_dose" in g.columns else g["insulin"]
            g["iob"] = _decay_accumulate(insulin_src.to_numpy(dtype=float), insulin_decay)
            g["cob"] = _decay_accumulate(g["carbs"].to_numpy(dtype=float), carbs_decay)
            g["glucose_delta"] = g["glucose"].diff(trend_steps).fillna(0.0)
            hour = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
            g["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            g["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            parts.append(g)

        result = pd.concat(parts, ignore_index=True)
        logger.info(
            f"Engineered features ditambahkan: iob, cob, glucose_delta, hour_sin, hour_cos "
            f"(insulin_tau={insulin_tau_min}min, carbs_tau={carbs_tau_min}min, trend={trend_steps} langkah)"
        )
        return result

    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 12,
                         prediction_horizon: int = 1, return_anchor: bool = False
                        ) -> Tuple[np.ndarray, ...]:
        """
        Create sequences for time-series prediction

        Args:
            df: Input DataFrame with features
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of steps ahead to predict (1 = next step).
                Pada cadence CGM 5-menit: 6 = +30 menit, 12 = +60 menit.

        Returns:
            X: Input sequences (n_samples, sequence_length, n_features)
            y: Target values (n_samples,)
        """
        logger.info(
            f"Creating sequences with length {sequence_length}, horizon {prediction_horizon}..."
        )

        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")

        df = validate_data_contract(df)
        self._require_columns(df, ['patient_id', 'timestamp', *self.feature_columns])

        X, y, anchors = [], [], []
        span = sequence_length + prediction_horizon  # baris minimum dibutuhkan

        for _, patient_df in df.sort_values(['patient_id', 'timestamp']).groupby('patient_id', sort=False):
            patient_df = patient_df.reset_index(drop=True)
            data = patient_df[self.feature_columns].values

            if len(data) < span:
                continue

            for i in range(len(data) - span + 1):
                X.append(data[i:i + sequence_length])
                # target = glukosa pada (akhir window + horizon)
                y.append(data[i + sequence_length + prediction_horizon - 1, 0])
                # anchor = glukosa terakhir di window (untuk target delta), fitur index 0 = glucose
                anchors.append(data[i + sequence_length - 1, 0])

        if not X:
            raise ValueError(
                f"Need at least sequence_length + prediction_horizon rows per patient; "
                f"no valid windows for sequence_length={sequence_length}, "
                f"prediction_horizon={prediction_horizon}"
            )

        X = np.array(X)
        y = np.array(y)
        anchors = np.array(anchors)

        logger.info(f"Created {len(X)} sequences")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

        if return_anchor:
            return X, y, anchors
        return X, y
    
    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize data using StandardScaler
        
        Args:
            X_train: Training data
            X_test: Test data (optional)
            
        Returns:
            X_train_scaled, X_test_scaled
        """
        logger.info("Normalizing data...")

        if X_train.ndim != 3:
            raise ValueError("X_train must have shape (samples, sequence_length, n_features)")
        if X_train.shape[0] == 0:
            raise ValueError("X_train must contain at least one sample")
        
        # Reshape for scaling
        n_samples, seq_len, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        # Fit scaler on training data
        self.scaler.fit(X_train_reshaped)
        
        # Transform
        X_train_scaled = self.scaler.transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, seq_len, n_features)
        
        if X_test is not None:
            if X_test.ndim != 3:
                raise ValueError("X_test must have shape (samples, sequence_length, n_features)")
            n_test, seq_len_test, _ = X_test.shape
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(n_test, seq_len_test, n_features)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def downsample_smbg(
        self,
        df: pd.DataFrame,
        interval_minutes: int = 240,
        source_interval_minutes: int = 5,
    ) -> pd.DataFrame:
        """Downsample dense CGM data to simulate SMBG (finger-prick) cadence.

        OhioT1DM records glucose every 5 minutes (CGM).  Real SMBG devices are
        used 3–6 times per day (~240-min gaps).  This function retains every
        N-th row per patient so downstream experiments can simulate a patient
        who self-monitors rather than wearing a CGM sensor.

        Args:
            df: DataFrame with 'patient_id' and 'timestamp' columns.
            interval_minutes: Target gap between retained readings (default 240 = 4 h ≈ 6/day).
            source_interval_minutes: Cadence of the source data in minutes (default 5).

        Returns:
            Downsampled DataFrame with the same schema.
        """
        step = max(1, round(interval_minutes / source_interval_minutes))
        logger.info(
            f"Downsampling CGM→SMBG: keeping 1 of every {step} rows "
            f"({interval_minutes} min cadence, ~{1440 // interval_minutes} readings/day)"
        )
        parts = []
        for _, patient_df in df.sort_values(["patient_id", "timestamp"]).groupby(
            "patient_id", sort=False
        ):
            parts.append(patient_df.iloc[::step].copy())
        result = pd.concat(parts, ignore_index=True)
        logger.info(
            f"Downsampled: {len(df)} → {len(result)} rows "
            f"({result['patient_id'].nunique()} patients)"
        )
        return result

    def split_by_patient(self, df: pd.DataFrame, test_patients: List[str]
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by patient ID (important for avoiding data leakage)
        
        Args:
            df: Full DataFrame
            test_patients: List of patient IDs for test set
            
        Returns:
            train_df, test_df
        """
        df = validate_data_contract(df)
        self._require_columns(df, ['patient_id'])

        df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)

        train_df = df[~df['patient_id'].isin(test_patients)].copy()
        test_df = df[df['patient_id'].isin(test_patients)].copy()
        
        logger.info(f"Train patients: {train_df['patient_id'].nunique()}")
        logger.info(f"Test patients: {test_df['patient_id'].nunique()}")
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        return train_df, test_df