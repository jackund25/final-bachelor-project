"""
Data preprocessing utilities for modality-aware glucose prediction.

M5.4 final design:
    CGM:
        - history: configurable (default 12 observations)
        - horizons: +30m / +60m
        - target tolerance: configurable (default 2.5m)

    FINGER_STICK / SMBG:
        - history: exactly 8 valid observations by default
        - near-duplicate interval <5m excluded
        - target window: 180–300m (~4h)
        - maximum target gap: 12h
        - irregular observation gaps are preserved as features

Important:
    - Official dataset_split is never recreated here.
    - glucose_source is filtered before sequence construction.
    - No artificial CGM→SMBG downsampling is used.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.contracts import assert_feature_set, validate_data_contract

logger = logging.getLogger(__name__)


def _decay_accumulate_time_aware(
    values: np.ndarray,
    timestamps: pd.Series,
    tau_min: float,
) -> np.ndarray:
    """Exponential accumulation using actual elapsed minutes."""
    values = np.asarray(values, dtype=float)
    ts = pd.to_datetime(timestamps).reset_index(drop=True)

    out = np.zeros(len(values), dtype=float)
    acc = 0.0

    for i, value in enumerate(values):
        if i == 0:
            decay = 0.0
        else:
            dt_min = max(
                0.0,
                (ts.iloc[i] - ts.iloc[i - 1]).total_seconds() / 60.0,
            )
            decay = float(np.exp(-dt_min / float(tau_min)))

        acc = float(value) + decay * acc
        out[i] = acc

    return out


class DataPreprocessor:
    """Preprocess raw data for modality-aware GBM training."""

    def __init__(self, config):
        self.config = config or {}
        self.scaler = StandardScaler()

        model_config = self.config.get("model", {})
        self.feature_columns = list(
            model_config.get(
                "engineered_features",
                model_config.get(
                    "features",
                    [
                        "glucose",
                        "carbs",
                        "insulin",
                        "activity",
                        "stress",
                    ],
                ),
            )
        )

    def _require_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> None:
        assert_feature_set(df, columns)

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        max_interpolate_steps: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values without fabricating glucose observations.

        Glucose is never interpolated. For sparse SMBG this is essential:
        an artificial glucose value would create a fake temporal observation.
        """
        logger.info("Handling missing values...")

        df = validate_data_contract(df)
        df = df.copy()
        df = df.sort_values(
            ["patient_id", "timestamp"]
        ).reset_index(drop=True)

        if "glucose" not in df.columns:
            raise ValueError("Dataset requires 'glucose'.")

        missing_before = int(df.isnull().sum().sum())

        # Glucose must be observed, not fabricated.
        before = len(df)
        df = df.dropna(subset=["glucose"])
        dropped_glucose = before - len(df)

        if dropped_glucose:
            logger.info(
                "Dropped %d rows with missing glucose; "
                "glucose is never interpolated.",
                dropped_glucose,
            )

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        numeric_non_glucose = [
            c for c in numeric_cols
            if c != "glucose"
        ]

        if numeric_non_glucose:
            if max_interpolate_steps is not None:
                df[numeric_non_glucose] = (
                    df[numeric_non_glucose]
                    .interpolate(
                        method="linear",
                        limit=int(max_interpolate_steps),
                        limit_direction="both",
                    )
                )
            else:
                df[numeric_non_glucose] = (
                    df[numeric_non_glucose]
                    .interpolate(
                        method="linear",
                        limit_direction="both",
                    )
                    .ffill()
                    .bfill()
                )

        missing_after = int(df.isnull().sum().sum())

        logger.info(
            "Missing values: %d → %d",
            missing_before,
            missing_after,
        )

        return df.reset_index(drop=True)

    def engineer_features(
        self,
        df: pd.DataFrame,
        insulin_tau_min: float = 240.0,
        carbs_tau_min: float = 180.0,
        trend_steps: int = 3,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Create elapsed-time-aware physiological and temporal features.

        IOB/COB use actual elapsed minutes.
        glucose_rate uses actual elapsed time.
        time_since_prev_glucose exposes irregular SMBG cadence.
        """
        df = validate_data_contract(df)
        df = df.copy()

        required = [
            "patient_id",
            "timestamp",
            "glucose",
            "glucose_source",
        ]

        self._require_columns(df, required)

        df = df.sort_values(
            ["patient_id", "timestamp"]
        ).reset_index(drop=True)

        parts = []

        for _, g in df.groupby(
            ["patient_id", "glucose_source"],
            sort=False,
        ):
            g = g.copy().reset_index(drop=True)

            if "bolus_dose" in g.columns:
                insulin_src = g["bolus_dose"].fillna(
                    g.get("insulin", 0.0)
                )
            else:
                insulin_src = g.get(
                    "insulin",
                    pd.Series(
                        0.0,
                        index=g.index,
                    ),
                )

            carbs_src = g.get(
                "carbs",
                pd.Series(
                    0.0,
                    index=g.index,
                ),
            )

            g["iob"] = _decay_accumulate_time_aware(
                insulin_src.fillna(0.0).to_numpy(dtype=float),
                g["timestamp"],
                insulin_tau_min,
            )

            g["cob"] = _decay_accumulate_time_aware(
                carbs_src.fillna(0.0).to_numpy(dtype=float),
                g["timestamp"],
                carbs_tau_min,
            )

            prev_ts = g["timestamp"].shift(1)

            dt_min = (
                g["timestamp"] - prev_ts
            ).dt.total_seconds().div(60.0)

            g["time_since_prev_glucose"] = (
                dt_min.fillna(0.0).clip(lower=0.0)
            )

            shift = max(1, int(trend_steps))

            g["glucose_delta"] = (
                g["glucose"]
                - g["glucose"].shift(shift)
            ).fillna(0.0)

            trend_dt = (
                g["timestamp"]
                - g["timestamp"].shift(shift)
            ).dt.total_seconds().div(60.0)

            g["glucose_rate"] = (
                g["glucose_delta"]
                .div(
                    trend_dt.replace(
                        0,
                        np.nan,
                    )
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0.0)
            )

            hour = (
                g["timestamp"].dt.hour
                + g["timestamp"].dt.minute / 60.0
            )

            g["hour_sin"] = np.sin(
                2 * np.pi * hour / 24.0
            )

            g["hour_cos"] = np.cos(
                2 * np.pi * hour / 24.0
            )

            parts.append(g)

        result = (
            pd.concat(parts, ignore_index=True)
            .sort_values(
                ["patient_id", "timestamp"]
            )
            .reset_index(drop=True)
        )

        logger.info(
            "Added time-aware features: "
            "iob, cob, glucose_delta, glucose_rate, "
            "time_since_prev_glucose, hour_sin, hour_cos"
        )

        return result

    def create_time_horizon_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        horizon_min: float,
        target_tolerance_min: float = 0.0,
        max_history_gap_min: Optional[float] = None,
        return_anchor: bool = False,
        min_target_horizon_min: Optional[float] = None,
        max_target_horizon_min: Optional[float] = None,
        min_history_interval_min: float = 0.0,
    ) -> Tuple[np.ndarray, ...]:
        """
        Create modality-aware temporal sequences.

        Fixed horizon:
            min/max target horizon omitted
            -> nearest future target around horizon_min with tolerance.

        Window horizon:
            min_target_horizon_min/max_target_horizon_min supplied
            -> choose the future observation closest to horizon_min
               inside the allowed elapsed-time window.

        SMBG quality rule:
            consecutive history intervals < min_history_interval_min
            are excluded. This removes near-duplicate finger-stick events.
        """
        if sequence_length < 1:
            raise ValueError(
                "sequence_length must be >= 1"
            )

        if horizon_min <= 0:
            raise ValueError(
                "horizon_min must be > 0"
            )

        if target_tolerance_min < 0:
            raise ValueError(
                "target_tolerance_min must be >= 0"
            )

        if (
            min_target_horizon_min is not None
            and max_target_horizon_min is None
        ):
            raise ValueError(
                "max_target_horizon_min is required "
                "when min_target_horizon_min is used."
            )

        if (
            max_target_horizon_min is not None
            and min_target_horizon_min is None
        ):
            raise ValueError(
                "min_target_horizon_min is required "
                "when max_target_horizon_min is used."
            )

        if (
            min_target_horizon_min is not None
            and min_target_horizon_min > max_target_horizon_min
        ):
            raise ValueError(
                "Invalid target horizon window."
            )

        df = validate_data_contract(df).copy()

        required = [
            "patient_id",
            "timestamp",
            "glucose",
            "glucose_source",
            *self.feature_columns,
        ]

        self._require_columns(
            df,
            list(dict.fromkeys(required)),
        )

        X = []
        y = []
        anchors = []
        target_elapsed = []

        group_cols = [
            "patient_id",
            "glucose_source",
        ]

        for _, g in (
            df.sort_values(
                [
                    "patient_id",
                    "glucose_source",
                    "timestamp",
                ]
            )
            .groupby(
                group_cols,
                sort=False,
            )
        ):
            g = g.reset_index(drop=True)

            if len(g) < sequence_length + 1:
                continue

            times = pd.to_datetime(
                g["timestamp"]
            )

            values = g[
                self.feature_columns
            ].to_numpy(dtype=float)

            glucose = g[
                "glucose"
            ].to_numpy(dtype=float)

            gaps = (
                times.diff()
                .dt.total_seconds()
                .div(60.0)
                .to_numpy()
            )

            for anchor_idx in range(
                sequence_length - 1,
                len(g),
            ):
                start_idx = (
                    anchor_idx
                    - sequence_length
                    + 1
                )

                history_gaps = gaps[
                    start_idx + 1:
                    anchor_idx + 1
                ]

                # No artificial/near-duplicate observations
                # inside the historical window.
                if (
                    min_history_interval_min > 0
                    and len(history_gaps)
                    and np.any(
                        history_gaps
                        < min_history_interval_min
                    )
                ):
                    continue

                # Optional maximum gap between historical observations.
                if (
                    max_history_gap_min is not None
                    and len(history_gaps)
                    and np.any(
                        history_gaps
                        > max_history_gap_min
                    )
                ):
                    continue

                target_time = (
                    times.iloc[anchor_idx]
                    + pd.Timedelta(
                        minutes=float(
                            horizon_min
                        )
                    )
                )

                future_start = anchor_idx + 1

                if future_start >= len(times):
                    continue

                future_times = times.iloc[
                    future_start:
                ]

                elapsed = (
                    future_times
                    - times.iloc[anchor_idx]
                ).dt.total_seconds().div(60.0)

                if (
                    min_target_horizon_min
                    is not None
                ):
                    candidate_mask = (
                        (elapsed >= min_target_horizon_min)
                        & (
                            elapsed
                            <= max_target_horizon_min
                        )
                    )
                else:
                    candidate_mask = (
                        np.abs(
                            elapsed
                            - horizon_min
                        )
                        <= target_tolerance_min
                    )

                candidate_positions = np.where(
                    candidate_mask.to_numpy()
                )[0]

                if len(candidate_positions) == 0:
                    continue

                nearest_pos = candidate_positions[
                    np.argmin(
                        np.abs(
                            elapsed.iloc[
                                candidate_positions
                            ].to_numpy()
                            - horizon_min
                        )
                    )
                ]

                target_idx = (
                    future_start
                    + int(nearest_pos)
                )

                actual_elapsed = float(
                    (
                        times.iloc[target_idx]
                        - times.iloc[anchor_idx]
                    ).total_seconds()
                    / 60.0
                )

                X.append(
                    values[
                        start_idx:
                        anchor_idx + 1
                    ]
                )

                y.append(
                    glucose[target_idx]
                )

                anchors.append(
                    glucose[anchor_idx]
                )

                target_elapsed.append(
                    actual_elapsed
                )

        if not X:
            raise ValueError(
                "No valid time-horizon sequences found: "
                f"sequence={sequence_length}, "
                f"horizon={horizon_min}min"
            )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        anchors = np.asarray(
            anchors,
            dtype=float,
        )

        logger.info(
            "Created %d temporal sequences "
            "(history=%d, target≈%g min, "
            "median actual target=%0.2f min)",
            len(y),
            sequence_length,
            horizon_min,
            float(
                np.median(target_elapsed)
            ),
        )

        if return_anchor:
            return X, y, anchors

        return X, y

    def normalize_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit StandardScaler on train only and transform train/test."""
        if X_train.ndim != 3:
            raise ValueError(
                "X_train must have shape "
                "(samples, sequence_length, n_features)"
            )

        if X_train.shape[0] == 0:
            raise ValueError(
                "X_train must contain samples"
            )

        n_samples, seq_len, n_features = (
            X_train.shape
        )

        train_flat = X_train.reshape(
            -1,
            n_features,
        )

        self.scaler.fit(
            train_flat
        )

        X_train_scaled = (
            self.scaler.transform(
                train_flat
            )
            .reshape(
                n_samples,
                seq_len,
                n_features,
            )
        )

        if X_test is None:
            return X_train_scaled, None

        if X_test.ndim != 3:
            raise ValueError(
                "X_test must have shape "
                "(samples, sequence_length, n_features)"
            )

        n_test = X_test.shape[0]

        X_test_scaled = (
            self.scaler.transform(
                X_test.reshape(
                    -1,
                    n_features,
                )
            )
            .reshape(
                n_test,
                X_test.shape[1],
                n_features,
            )
        )

        return (
            X_train_scaled,
            X_test_scaled,
        )

    def downsample_smbg(
        self,
        df: pd.DataFrame,
        interval_minutes: int = 240,
        source_interval_minutes: int = 5,
    ) -> pd.DataFrame:
        """
        Legacy simulation helper.

        NOT used by the OhioT1DM real-SMBG training pipeline.
        """
        step = max(
            1,
            round(
                interval_minutes
                / source_interval_minutes
            ),
        )

        parts = []

        for _, patient_df in (
            df.sort_values(
                ["patient_id", "timestamp"]
            )
            .groupby(
                "patient_id",
                sort=False,
            )
        ):
            parts.append(
                patient_df.iloc[
                    ::step
                ].copy()
            )

        return pd.concat(
            parts,
            ignore_index=True,
        )

    def split_by_patient(
        self,
        df: pd.DataFrame,
        test_patients: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Legacy helper.

        Production OhioT1DM training uses the official dataset_split
        instead of creating a new patient split.
        """
        df = validate_data_contract(
            df
        ).copy()

        test_patients = set(
            test_patients
        )

        train_df = df[
            ~df["patient_id"].isin(
                test_patients
            )
        ].copy()

        test_df = df[
            df["patient_id"].isin(
                test_patients
            )
        ].copy()

        return train_df, test_df