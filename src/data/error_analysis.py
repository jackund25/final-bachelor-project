"""
M5.4-D — Final GBM Error Analysis

Runs post-training error analysis for:
    CGM +30 min
    CGM +60 min
    SMBG ~4h (180–300 min)

Outputs:
    data/processed/error_analysis_predictions.csv
    data/processed/error_analysis_summary.csv
    data/processed/error_analysis_patient.csv
    data/processed/error_analysis_glucose_range.csv
    data/processed/error_analysis_horizon_bins.csv

Requirements:
    - Run GBM training first.
    - Model inference bundles must exist in models/.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils.metrics import calculate_all_metrics


CONFIG_PATH = "config.yaml"
DATASET_NAME = "ohio_t1dm.csv"
OUTPUT_DIR = Path("data/processed")


MODEL_SPECS = [
    {
        "name": "CGM_30m",
        "source": "CGM",
        "horizon": 30.0,
        "bundle": "models/gbm_cgm_h30m_inference_bundle.pkl",
        "window": None,
    },
    {
        "name": "CGM_60m",
        "source": "CGM",
        "horizon": 60.0,
        "bundle": "models/gbm_cgm_h60m_inference_bundle.pkl",
        "window": None,
    },
    {
        "name": "SMBG_4h",
        "source": "FINGER_STICK",
        "horizon": 240.0,
        "bundle": "models/gbm_finger_stick_h4h_seq8_inference_bundle.pkl",
        "window": (180.0, 300.0),
    },
]


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def clarke_zone(actual, predicted):
    """
    Clarke Error Grid classification.

    This implementation follows the standard five-zone logic
    used by the project's calculate_all_metrics implementation.
    """
    a = float(actual)
    p = float(predicted)

    # Zone E
    if (a < 70 and p > 180) or (a > 180 and p < 70):
        return "E"

    # Zone D
    if a >= 70 and a <= 180:
        if p < 70 or p > 180:
            return "D"

    if a < 70:
        if p >= 70 and p <= 180:
            return "D"

    # Zone C
    if a < 70 and p >= 70 and p < 180:
        return "D"

    # Standard approximation for remaining zones.
    # Zone A: <=20% deviation, or both in hypoglycemic region.
    if a < 70 and p < 70:
        return "A"

    if a != 0 and abs(p - a) <= 0.20 * abs(a):
        return "A"

    # Zone B is clinically benign deviation.
    return "B"


def glucose_range(value):
    if value < 70:
        return "hypoglycemia"
    if value <= 180:
        return "target_range"
    return "hyperglycemia"


def horizon_bin(actual_elapsed):
    if actual_elapsed < 180:
        return "<180m"
    if actual_elapsed < 210:
        return "180–210m"
    if actual_elapsed < 240:
        return "210–240m"
    if actual_elapsed < 270:
        return "240–270m"
    if actual_elapsed <= 300:
        return "270–300m"
    return ">300m"


def build_sequences(preprocessor, df, source, bundle):
    sequence_length = int(bundle["sequence_length"])
    horizon = float(bundle["prediction_horizon_min"])
    tolerance = float(bundle.get("target_tolerance_min", 0.0))
    min_target = bundle.get("min_target_horizon_min")
    max_target = bundle.get("max_target_horizon_min")
    min_interval = float(
        bundle.get("min_history_interval_min", 0.0)
    )
    max_history_gap = bundle.get("max_history_gap_min")

    X, y, anchors = preprocessor.create_time_horizon_sequences(
        df,
        sequence_length=sequence_length,
        horizon_min=horizon,
        target_tolerance_min=tolerance,
        max_history_gap_min=max_history_gap,
        return_anchor=True,
        min_target_horizon_min=min_target,
        max_target_horizon_min=max_target,
        min_history_interval_min=min_interval,
    )

    return X, y, anchors


def reconstruct_metadata(
    preprocessor,
    df,
    bundle,
):
    """
    Reconstruct sequence-level metadata using the same target-selection
    logic as create_time_horizon_sequences.

    The returned rows are ordered exactly as the sequence builder emits them.
    """
    source = bundle["glucose_source"]
    sequence_length = int(bundle["sequence_length"])
    horizon = float(bundle["prediction_horizon_min"])
    tolerance = float(bundle.get("target_tolerance_min", 0.0))
    min_target = bundle.get("min_target_horizon_min")
    max_target = bundle.get("max_target_horizon_min")
    min_interval = float(
        bundle.get("min_history_interval_min", 0.0)
    )
    max_history_gap = bundle.get("max_history_gap_min")

    rows = []

    for (patient_id, glucose_source), g in (
        df.sort_values(
            ["patient_id", "glucose_source", "timestamp"]
        )
        .groupby(
            ["patient_id", "glucose_source"],
            sort=False,
        )
    ):
        if glucose_source != source:
            continue

        g = g.reset_index(drop=True)
        times = pd.to_datetime(g["timestamp"])
        glucose = g["glucose"].to_numpy(dtype=float)

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
            start_idx = anchor_idx - sequence_length + 1

            history_gaps = gaps[
                start_idx + 1 : anchor_idx + 1
            ]

            if (
                min_interval > 0
                and len(history_gaps)
                and np.any(
                    history_gaps < min_interval
                )
            ):
                continue

            if (
                max_history_gap is not None
                and len(history_gaps)
                and np.any(
                    history_gaps > max_history_gap
                )
            ):
                continue

            future_start = anchor_idx + 1
            if future_start >= len(g):
                continue

            future_times = times.iloc[future_start:]
            elapsed = (
                future_times - times.iloc[anchor_idx]
            ).dt.total_seconds().div(60.0)

            if min_target is not None:
                mask = (
                    (elapsed >= float(min_target))
                    & (elapsed <= float(max_target))
                )
            else:
                mask = (
                    np.abs(elapsed - horizon)
                    <= tolerance
                )

            positions = np.where(mask.to_numpy())[0]

            if len(positions) == 0:
                continue

            nearest = positions[
                np.argmin(
                    np.abs(
                        elapsed.iloc[positions].to_numpy()
                        - horizon
                    )
                )
            ]

            target_idx = future_start + int(nearest)

            rows.append(
                {
                    "patient_id": patient_id,
                    "dataset_split": g["dataset_split"].iloc[0],
                    "anchor_timestamp": times.iloc[anchor_idx],
                    "target_timestamp": times.iloc[target_idx],
                    "actual_elapsed_min": float(
                        elapsed.iloc[nearest]
                    ),
                    "anchor_glucose": float(
                        glucose[anchor_idx]
                    ),
                    "actual_glucose": float(
                        glucose[target_idx]
                    ),
                }
            )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    loader = DiabetesDataLoader(
        config["data"]["output_dir"]
    )

    df = loader.load_csv(
        config["data"].get(
            "unified_dataset",
            DATASET_NAME,
        )
    )

    df = df[
        df["dataset_split"] == "test"
    ].copy()

    preprocessor = DataPreprocessor(
        config
    )

    df = preprocessor.handle_missing_values(
        df,
        max_interpolate_steps=config[
            "model"
        ].get(
            "max_interpolate_steps"
        ),
    )

    df = preprocessor.engineer_features(
        df,
        **config["model"].get(
            "feature_engineering",
            {},
        ),
    )

    all_predictions = []

    for spec in MODEL_SPECS:
        bundle_path = Path(spec["bundle"])

        if not bundle_path.exists():
            print(
                f"[SKIP] Missing bundle: {bundle_path}"
            )
            continue

        bundle = load_bundle(bundle_path)

        source = spec["source"]

        source_df = df[
            df["glucose_source"] == source
        ].copy()

        if source_df.empty:
            print(
                f"[SKIP] No rows for {source}"
            )
            continue

        preprocessor.feature_columns = list(
            bundle["features"]
        )

        X, y, anchors = build_sequences(
            preprocessor,
            source_df,
            source,
            bundle,
        )

        X_scaled = (
            bundle["scaler"].transform(
                X.reshape(
                    -1,
                    X.shape[-1],
                )
            )
            .reshape(X.shape)
        )

        model = bundle["model"]
        pred_fit = model.predict(
            X_scaled.reshape(
                X_scaled.shape[0],
                -1,
            )
        )

        if bundle.get(
            "predict_delta",
            True,
        ):
            predictions = (
                pred_fit + anchors
            )
        else:
            predictions = pred_fit

        metadata = reconstruct_metadata(
            preprocessor,
            source_df,
            bundle,
        )

        if len(metadata) != len(predictions):
            raise RuntimeError(
                f"Metadata/prediction mismatch for {spec['name']}: "
                f"{len(metadata)} vs {len(predictions)}"
            )

        metadata["model"] = spec["name"]
        metadata["prediction"] = predictions
        metadata["error"] = (
            predictions
            - metadata["actual_glucose"].to_numpy()
        )
        metadata["abs_error"] = np.abs(
            metadata["error"]
        )
        metadata["squared_error"] = (
            metadata["error"] ** 2
        )
        metadata["actual_range"] = (
            metadata["actual_glucose"]
            .apply(glucose_range)
        )
        metadata["horizon_bin"] = (
            metadata["actual_elapsed_min"]
            .apply(horizon_bin)
        )
        metadata["clarke_zone"] = [
            clarke_zone(a, p)
            for a, p in zip(
                metadata["actual_glucose"],
                metadata["prediction"],
            )
        ]

        all_predictions.append(
            metadata
        )

        metrics = calculate_all_metrics(
            metadata["actual_glucose"].to_numpy(),
            predictions,
        )

        print("\n" + "=" * 75)
        print(spec["name"])
        print("=" * 75)
        print(
            f"Samples : {len(metadata):,}"
        )

        for key, value in metrics.items():
            print(
                f"{key:14s}: {float(value):.4f}"
            )

        print(
            "\nZone distribution:"
        )
        print(
            metadata["clarke_zone"]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .round(2)
            .sort_index()
            .to_string()
        )

    if not all_predictions:
        raise RuntimeError(
            "No model bundles were found. "
            "Run GBM training first."
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # Overall summary

    summary_rows = []

    for model_name, g in predictions.groupby(
        "model",
        sort=False,
    ):
        rmse = float(
            np.sqrt(
                np.mean(
                    g["squared_error"]
                )
            )
        )
        mae = float(
            g["abs_error"].mean()
        )

        zones = (
            g["clarke_zone"]
            .value_counts()
        )

        total = len(g)

        summary_rows.append(
            {
                "model": model_name,
                "n": total,
                "MAE": mae,
                "RMSE": rmse,
                "A_pct": 100 * zones.get("A", 0) / total,
                "B_pct": 100 * zones.get("B", 0) / total,
                "C_pct": 100 * zones.get("C", 0) / total,
                "D_pct": 100 * zones.get("D", 0) / total,
                "E_pct": 100 * zones.get("E", 0) / total,
                "A+B_pct": 100 * (
                    zones.get("A", 0)
                    + zones.get("B", 0)
                ) / total,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # Patient-level analysis

    patient_rows = []

    for (
        model_name,
        patient_id,
    ), g in predictions.groupby(
        ["model", "patient_id"],
        sort=False,
    ):
        zones = (
            g["clarke_zone"]
            .value_counts()
        )
        total = len(g)

        patient_rows.append(
            {
                "model": model_name,
                "patient_id": patient_id,
                "n": total,
                "MAE": g["abs_error"].mean(),
                "RMSE": np.sqrt(
                    g["squared_error"].mean()
                ),
                "mean_error": g["error"].mean(),
                "median_abs_error": g[
                    "abs_error"
                ].median(),
                "A+B_pct": 100 * (
                    zones.get("A", 0)
                    + zones.get("B", 0)
                ) / total,
                "D+E_pct": 100 * (
                    zones.get("D", 0)
                    + zones.get("E", 0)
                ) / total,
            }
        )

    patient = pd.DataFrame(
        patient_rows
    )

    # Glucose range analysis

    range_rows = []

    for (
        model_name,
        actual_range,
    ), g in predictions.groupby(
        ["model", "actual_range"],
        sort=False,
    ):
        zones = (
            g["clarke_zone"]
            .value_counts()
        )
        total = len(g)

        range_rows.append(
            {
                "model": model_name,
                "actual_range": actual_range,
                "n": total,
                "MAE": g["abs_error"].mean(),
                "RMSE": np.sqrt(
                    g["squared_error"].mean()
                ),
                "mean_error": g["error"].mean(),
                "A+B_pct": 100 * (
                    zones.get("A", 0)
                    + zones.get("B", 0)
                ) / total,
                "D+E_pct": 100 * (
                    zones.get("D", 0)
                    + zones.get("E", 0)
                ) / total,
            }
        )

    glucose_range_result = pd.DataFrame(
        range_rows
    )

    # Horizon analysis

    horizon_rows = []

    for (
        model_name,
        hbin,
    ), g in predictions.groupby(
        ["model", "horizon_bin"],
        sort=False,
    ):
        zones = (
            g["clarke_zone"]
            .value_counts()
        )
        total = len(g)

        horizon_rows.append(
            {
                "model": model_name,
                "horizon_bin": hbin,
                "n": total,
                "MAE": g["abs_error"].mean(),
                "RMSE": np.sqrt(
                    g["squared_error"].mean()
                ),
                "mean_error": g["error"].mean(),
                "A+B_pct": 100 * (
                    zones.get("A", 0)
                    + zones.get("B", 0)
                ) / total,
                "D+E_pct": 100 * (
                    zones.get("D", 0)
                    + zones.get("E", 0)
                ) / total,
            }
        )

    horizon_result = pd.DataFrame(
        horizon_rows
    )

    # Save

    predictions.to_csv(
        OUTPUT_DIR
        / "error_analysis_predictions.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "error_analysis_summary.csv",
        index=False,
    )

    patient.to_csv(
        OUTPUT_DIR
        / "error_analysis_patient.csv",
        index=False,
    )

    glucose_range_result.to_csv(
        OUTPUT_DIR
        / "error_analysis_glucose_range.csv",
        index=False,
    )

    horizon_result.to_csv(
        OUTPUT_DIR
        / "error_analysis_horizon_bins.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("FINAL ERROR ANALYSIS")
    print("=" * 75)

    print("\nOverall:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nPatient-level:")
    print(
        patient.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nGlucose range:")
    print(
        glucose_range_result.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nActual target horizon:")
    print(
        horizon_result.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nFiles:")
    print(
        OUTPUT_DIR
        / "error_analysis_predictions.csv"
    )
    print(
        OUTPUT_DIR
        / "error_analysis_summary.csv"
    )
    print(
        OUTPUT_DIR
        / "error_analysis_patient.csv"
    )
    print(
        OUTPUT_DIR
        / "error_analysis_glucose_range.csv"
    )
    print(
        OUTPUT_DIR
        / "error_analysis_horizon_bins.csv"
    )


if __name__ == "__main__":
    main()