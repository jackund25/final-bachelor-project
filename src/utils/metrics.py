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

    CATATAN AMBANG: memakai CLARKE_LOW/CLARKE_HIGH, BUKAN GLUCOSE_LOW/GLUCOSE_HIGH.
    Angkanya kebetulan sama (70/180), tetapi batas zona Clarke adalah bagian dari
    metrik terbitan yang baku. Bila kebijakan ambang klinis proyek ini berubah,
    metrik ini TIDAK boleh ikut berubah — kalau ikut, ia berhenti menjadi Clarke
    Error Grid dan seluruh perbandingan dengan literatur menjadi batal.
    """
    from src.constants import CLARKE_HIGH, CLARKE_LOW

    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for true_val, pred_val in zip(y_true, y_pred):
        # Zone A (clinically accurate)
        if (true_val < CLARKE_LOW and pred_val < CLARKE_LOW) or \
           (abs(true_val - pred_val) <= 0.2 * true_val):
            zones['A'] += 1

        # Zone B (benign errors)
        elif (CLARKE_LOW <= true_val <= CLARKE_HIGH and CLARKE_LOW <= pred_val <= CLARKE_HIGH):
            zones['B'] += 1

        # Zone C (overcorrection)
        elif (true_val < CLARKE_LOW and pred_val > CLARKE_HIGH) or \
             (true_val > CLARKE_HIGH and pred_val < CLARKE_LOW):
            zones['C'] += 1

        # Zone D (failure to detect)
        elif (true_val < CLARKE_LOW and CLARKE_LOW <= pred_val <= CLARKE_HIGH) or \
             (true_val > CLARKE_HIGH and CLARKE_LOW <= pred_val <= CLARKE_HIGH):
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