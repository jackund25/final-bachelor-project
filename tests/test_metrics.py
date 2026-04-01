# test_metrics.py
from src.utils.metrics import calculate_all_metrics
import numpy as np

y_true = np.array([100, 120, 150, 180, 200])
y_pred = np.array([105, 115, 155, 175, 195])

metrics = calculate_all_metrics(y_true, y_pred)
print(metrics)
# Expected: RMSE ~7, MAE ~6, Clarke A+B >90%