from pathlib import Path

import pandas as pd

from src.data import DiabetesDataGenerator, DiabetesDataLoader


CONFIG_TEMPLATE = """
data:
  num_patients: 2
  days_per_patient: 1
  sampling_interval_min: 5
  output_dir: '{output_dir}'
  seed: 42
  generation_mode: "dummy"

patient:
  baseline_glucose: 100
  glucose_threshold_low: 70
  glucose_threshold_high: 180
  insulin_sensitivity: 50
  carb_ratio: 10

model:
  features: ["glucose", "carbs", "insulin", "activity", "stress"]
"""


def test_dummy_generator_creates_realistic_dataset(tmp_path):
    output_dir = tmp_path / "raw"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEMPLATE.format(output_dir=output_dir.as_posix()).strip(), encoding="utf-8")

    generator = DiabetesDataGenerator(str(config_path))
    df = generator.generate_dataset(save=False)

    assert not df.empty
    assert {"patient_id", "timestamp", "glucose", "carbs", "insulin", "activity", "stress"}.issubset(df.columns)
    assert {"sleep", "work", "illness", "meal_type", "glucose_change"}.issubset(df.columns)
    assert df["glucose"].between(40, 400).all()
    assert df["patient_id"].nunique() == 2


def test_loader_reads_generated_csv(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "patient_id": ["P001"],
            "timestamp": ["2026-04-01 08:00:00"],
            "glucose": [120.0],
            "carbs": [30.0],
            "insulin": [3.0],
            "activity": [15],
            "stress": [5],
        }
    )
    csv_path = data_dir / "training_data_complete.csv"
    df.to_csv(csv_path, index=False)

    loader = DiabetesDataLoader(str(data_dir))
    loaded = loader.load_latest_dataset()

    assert not loaded.empty
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])
    assert {"sleep", "work", "illness", "meal_type"}.issubset(loaded.columns)
