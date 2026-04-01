"""
Base model class
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Optional
import pickle
from pathlib import Path


class BaseGlucoseModel(ABC):
    """
    Abstract base class untuk semua glucose prediction models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.is_trained = False
        
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
        
        predictions = self.predict(X_test)
        metrics = calculate_all_metrics(y_test, predictions)
        
        return metrics