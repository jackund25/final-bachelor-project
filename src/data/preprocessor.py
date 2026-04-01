"""
Data Preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocess raw data for model training
    """
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns = config['model']['features']
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values via interpolation
        
        Args:
            df: Input DataFrame
            
        Returns:
            df_clean: Cleaned DataFrame
        """
        logger.info("Handling missing values...")
        
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
        
        # Extract feature columns
        data = df[self.feature_columns].values
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length, 0])  # Next glucose value
        
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
        
        # Reshape for scaling
        n_samples, seq_len, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        # Fit scaler on training data
        self.scaler.fit(X_train_reshaped)
        
        # Transform
        X_train_scaled = self.scaler.transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, seq_len, n_features)
        
        if X_test is not None:
            n_test, seq_len_test, _ = X_test.shape
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(n_test, seq_len_test, n_features)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
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
        train_df = df[~df['patient_id'].isin(test_patients)].copy()
        test_df = df[df['patient_id'].isin(test_patients)].copy()
        
        logger.info(f"Train patients: {train_df['patient_id'].nunique()}")
        logger.info(f"Test patients: {test_df['patient_id'].nunique()}")
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        return train_df, test_df