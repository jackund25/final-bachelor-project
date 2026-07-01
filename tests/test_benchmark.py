import json
from pathlib import Path

import numpy as np
import yaml

from src.models.benchmark import run_benchmark_from_config


def test_run_benchmark_creates_artifacts(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamps = np.arange("2024-01-01T00:00", "2024-01-02T00:00", dtype="datetime64[5m]")
    rows = []
    for idx, patient_id in enumerate(["P001", "P002", "P003", "P004"]):
        for t in timestamps:
            glucose = 95 + idx * 5 + np.sin(len(rows) / 20.0) * 10
            rows.append(
                {
                    "patient_id": patient_id,
                    "timestamp": str(t),
                    "glucose": float(glucose),
                    "carbs": float((len(rows) % 6 == 0) * 30),
                    "insulin": float((len(rows) % 6 == 0) * 3),
                    "activity": int(len(rows) % 4 == 0) * 15,
                    "stress": int(4 + (len(rows) % 3)),
                }
            )

    import pandas as pd

    pd.DataFrame(rows).to_csv(data_dir / "training_data_complete.csv", index=False)

    config = {
        "data": {"output_dir": str(data_dir), "seed": 42},
        "model": {
            "sequence_length": 12,
            "features": ["glucose", "carbs", "insulin", "activity", "stress"],
            "random_forest": {"n_estimators": 10, "max_depth": 6},
            "gbm": {"n_estimators": 50, "max_depth": 3},
        },
    }

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        experiments = run_benchmark_from_config(str(cfg_path))
    finally:
        os.chdir(cwd)

    models_dir = tmp_path / "models"
    assert (models_dir / "metrics_experiments.json").exists()
    with open(models_dir / "metrics_experiments.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "rf" in data["results"]
    assert "gb" in data["results"]
