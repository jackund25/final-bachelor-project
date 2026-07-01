"""Data loading utilities for the diabetes digital twin project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.data.contracts import validate_data_contract


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
        filepath = Path(filename)
        if not filepath.is_absolute():
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

    def load_preferred_dataset(
        self,
        primary_source: str = "ohio_t1dm",
        fallback_source: str = "latest_generated",
    ) -> Tuple[pd.DataFrame, str]:
        """Load dataset using primary source and automatic fallback source."""
        errors = []

        for source in [primary_source, fallback_source]:
            try:
                df = self._load_by_source(source)
                return df, source
            except FileNotFoundError as exc:
                errors.append(f"{source}: {exc}")

        raise FileNotFoundError("No available dataset found. " + " | ".join(errors))

    def _load_by_source(self, source: str) -> pd.DataFrame:
        """Resolve known source names to concrete dataset files."""
        if source == "ohio_t1dm":
            return self.load_csv("ohio_t1dm_merged.csv")
        if source == "latest_generated":
            return self.load_latest_dataset()
        if source == "manual_logbook":
            return self.load_csv("manual_logbook.csv")
        return self.load_csv(source)

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
        return validate_data_contract(df)