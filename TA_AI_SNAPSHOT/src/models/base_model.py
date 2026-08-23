"""
Base model class.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class BaseGlucoseModel(ABC):
    """
    Abstract base class untuk semua glucose prediction models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.is_trained = False

    def _validate_training_data(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate model training inputs and normalize them to numpy arrays."""
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).reshape(-1)

        if X_train.ndim != 3:
            raise ValueError("Expected 3D input for training data: (samples, sequence_length, n_features)")
        if X_train.shape[0] == 0:
            raise ValueError("Training data must contain at least one sample")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must contain the same number of samples")
        if not np.isfinite(X_train).all():
            raise ValueError("X_train contains NaN or infinite values")
        if not np.isfinite(y_train).all():
            raise ValueError("y_train contains NaN or infinite values")

        return X_train, y_train

    def _validate_prediction_data(self, X: np.ndarray) -> np.ndarray:
        """Validate prediction inputs and normalize them to numpy arrays."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 3:
            raise ValueError("Expected 3D input for prediction data: (samples, sequence_length, n_features)")
        if X.shape[0] == 0:
            raise ValueError("Prediction data must contain at least one sample")
        if not np.isfinite(X).all():
            raise ValueError("Prediction data contains NaN or infinite values")

        return X
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train the model
        
        Returns:
            training_history: Dict with metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            predictions: Predicted glucose values
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load model from disk"""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Returns:
            metrics: Dict with RMSE, MAE, etc.
        """
        from src.utils.metrics import calculate_all_metrics
        
        X_test = self._validate_prediction_data(X_test)
        y_test = np.asarray(y_test, dtype=float).reshape(-1)
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must contain the same number of samples")

        predictions = self.predict(X_test)
        metrics = calculate_all_metrics(y_test, predictions)
        
        return metrics