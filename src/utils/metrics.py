"""
Evaluation metrics untuk glucose prediction
"""
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def clarke_error_grid(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Clarke Error Grid Analysis
    
    Zones:
    A: Clinically accurate (safe)
    B: Benign errors (acceptable)
    C: Overcorrection errors
    D: Failure to detect
    E: Erroneous treatment
    
    Returns:
        Dict dengan percentage di tiap zone
    """
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    
    for true_val, pred_val in zip(y_true, y_pred):
        # Zone A (clinically accurate)
        if (true_val < 70 and pred_val < 70) or \
           (abs(true_val - pred_val) <= 0.2 * true_val):
            zones['A'] += 1
            
        # Zone B (benign errors)
        elif (true_val >= 70 and true_val <= 180 and pred_val >= 70 and pred_val <= 180):
            zones['B'] += 1
            
        # Zone C (overcorrection)
        elif (true_val < 70 and pred_val > 180) or (true_val > 180 and pred_val < 70):
            zones['C'] += 1
            
        # Zone D (failure to detect)
        elif (true_val < 70 and pred_val >= 70 and pred_val <= 180) or \
             (true_val > 180 and pred_val >= 70 and pred_val <= 180):
            zones['D'] += 1
            
        # Zone E (erroneous treatment)
        else:
            zones['E'] += 1
    
    total = len(y_true)
    return {zone: (count / total) * 100 for zone, count in zones.items()}


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate semua metrics sekaligus
    
    Returns:
        Dict dengan semua metrics
    """
    clarke = clarke_error_grid(y_true, y_pred)
    
    return {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'Clarke_A': clarke['A'],
        'Clarke_B': clarke['B'],
        'Clarke_C': clarke['C'],
        'Clarke_D': clarke['D'],
        'Clarke_E': clarke['E'],
        'Clarke_A+B': clarke['A'] + clarke['B']  # Safe zone
    }