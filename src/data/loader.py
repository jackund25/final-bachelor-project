"""Data loading utilities for the diabetes digital twin project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class DatasetInfo:
    """Metadata about an available dataset file."""

    path: Path
    rows: int
    columns: List[str]


class DiabetesDataLoader:
    """Load raw or processed data produced by the generator."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load a single CSV file from the data directory."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        return self._standardize_columns(df)

    def load_latest_dataset(self) -> pd.DataFrame:
        """Load the default generated dataset if present."""
        candidates = [
            self.data_dir / "training_data_complete.csv",
            self.data_dir / "dummy_diabetes_data.csv",
        ]

        for filepath in candidates:
            if filepath.exists():
                df = pd.read_csv(filepath, parse_dates=["timestamp"])
                return self._standardize_columns(df)

        csv_files = sorted(self.data_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV datasets found in {self.data_dir}")

        df = pd.read_csv(csv_files[0], parse_dates=["timestamp"])
        return self._standardize_columns(df)

    def list_datasets(self) -> List[DatasetInfo]:
        """Return simple metadata for all CSV files in the directory."""
        datasets: List[DatasetInfo] = []

        for filepath in sorted(self.data_dir.glob("*.csv")):
            try:
                preview = pd.read_csv(filepath, nrows=5)
                rows = sum(1 for _ in open(filepath, "r", encoding="utf-8")) - 1
                datasets.append(
                    DatasetInfo(
                        path=filepath,
                        rows=max(rows, 0),
                        columns=list(preview.columns),
                    )
                )
            except Exception:
                continue

        return datasets

    def load_patient_data(self, patient_id: str, filename: str = "training_data_complete.csv") -> pd.DataFrame:
        """Load data filtered for a single patient."""
        df = self.load_csv(filename)
        return df[df["patient_id"] == patient_id].copy()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the canonical columns exist for the rest of the pipeline."""
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        column_defaults = {
            "sleep": 0,
            "work": 0,
            "illness": 0,
            "meal_type": "none",
            "glucose_change": 0.0,
        }

        for column, default_value in column_defaults.items():
            if column not in df.columns:
                df[column] = default_value

        numeric_columns = ["glucose", "carbs", "insulin", "activity", "stress", "sleep", "work", "illness", "glucose_change"]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        return df