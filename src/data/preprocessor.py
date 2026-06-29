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
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 12
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series prediction
        
        Args:
            df: Input DataFrame with features
            sequence_length: Number of time steps to look back
            
        Returns:
            X: Input sequences (n_samples, sequence_length, n_features)
            y: Target values (n_samples,)
        """
        logger.info(f"Creating sequences with length {sequence_length}...")

        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")

        df = validate_data_contract(df)
        self._require_columns(df, ['patient_id', 'timestamp', *self.feature_columns])

        X, y = [], []

        for _, patient_df in df.sort_values(['patient_id', 'timestamp']).groupby('patient_id', sort=False):
            patient_df = patient_df.reset_index(drop=True)
            data = patient_df[self.feature_columns].values

            if len(data) <= sequence_length:
                continue

            for i in range(len(data) - sequence_length):
                X.append(data[i:i + sequence_length])
                y.append(data[i + sequence_length, 0])  # Next glucose value

        if not X:
            raise ValueError(
                f"Need at least sequence_length + 1 rows per patient to create sequences; got no valid windows for sequence_length={sequence_length}"
            )

        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
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