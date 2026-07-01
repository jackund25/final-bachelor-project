"""Canonical data contract for diabetes datasets used by the project."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "patient_id",
    "timestamp",
    "glucose",
    "carbs",
    "insulin",
    "activity",
]

# stress dipindah ke opsional: nyaris tak ada data di OhioT1DM (7 event), sehingga
# bukan fitur model. Tetap dipertahankan sebagai variabel Digital Twin / logbook.
OPTIONAL_DEFAULTS = {
    "stress": 0,
    "sleep": 0,
    "work": 0,
    "illness": 0,
    "meal_type": "none",
    "glucose_change": 0.0,
    "source": "unknown",
}

NUMERIC_COLUMNS = [
    "glucose",
    "carbs",
    "insulin",
    "activity",
    "stress",
    "sleep",
    "work",
    "illness",
    "glucose_change",
]


def validate_data_contract(df: pd.DataFrame, *, require_required: bool = True) -> pd.DataFrame:
    """Validate and coerce a dataframe into the project contract.

    The contract is intentionally small and stable so all pipeline stages use the
    same patient/time/glucose feature names.
    """

    if df.empty:
        raise ValueError("Input dataframe must not be empty")

    frame = df.copy()

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if require_required and missing:
        raise ValueError(f"Missing required data contract columns: {', '.join(missing)}")

    for column, default_value in OPTIONAL_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default_value

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "patient_id" in frame.columns:
        frame["patient_id"] = frame["patient_id"].astype(str)

    if "meal_type" in frame.columns:
        frame["meal_type"] = frame["meal_type"].fillna("none").astype(str)

    frame = frame.dropna(subset=[column for column in ["patient_id", "timestamp"] if column in frame.columns])

    return pd.DataFrame(frame.to_dict("list"), index=frame.index)


def assert_feature_set(frame: pd.DataFrame, required_columns: Iterable[str] | None = None) -> None:
    """Raise a clear error when expected columns are missing."""

    columns = list(required_columns or REQUIRED_COLUMNS)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")