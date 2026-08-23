"""
M5.4-C — SMBG Training Sample Feasibility Audit

Goal
----
Estimate how many usable SMBG training samples remain for combinations of:

    History length : 5, 8, 12 observations
    Prediction     : ~1h, ~2h, ~4h

Quality filters:
    1. Consecutive SMBG interval < 5 min is excluded.
    2. Prediction target must be within 12 hours.
    3. A sample must have the requested number of valid historical
       SMBG observations.
    4. The future target is selected using a time window around the
       requested horizon.

Target windows:
    ~1 hour : 45–75 min
    ~2 hours: 90–150 min
    ~4 hours: 180–300 min

Important
---------
This is an AUDIT only. It does not modify the source dataset or train GBM.

The audit keeps the official OhioT1DM train/test split and never creates
a history across patients or across train/test boundaries.

Input:
    data/raw/ohio_t1dm.csv

Output:
    data/processed/smbg_training_sample_audit.csv
    data/processed/smbg_training_sample_patient.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


HISTORY_LENGTHS = [5, 8, 12]

HORIZONS = {
    "1h": (45.0, 75.0),
    "2h": (90.0, 150.0),
    "4h": (180.0, 300.0),
}

MAX_TARGET_GAP_MIN = 720.0  # 12 hours
MIN_INTERVAL_MIN = 5.0      # near-duplicate filter


REQUIRED_COLUMNS = {
    "patient_id",
    "timestamp",
    "glucose_source",
    "dataset_split",
    "glucose",
}


def prepare_smbg(csv_path):
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
    )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[
        df["glucose_source"] == "FINGER_STICK"
    ].copy()

    df["glucose"] = pd.to_numeric(
        df["glucose"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "patient_id",
            "timestamp",
            "dataset_split",
            "glucose",
        ]
    )

    df = df.sort_values(
        [
            "patient_id",
            "dataset_split",
            "timestamp",
        ]
    ).reset_index(drop=True)

    # Consecutive interval.
    df["interval_min"] = (
        df.groupby(
            ["patient_id", "dataset_split"]
        )["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60.0
    )

    # Keep the first observation of each sequence, then require every
    # historical transition used in a sequence to be >=5 min.
    df["valid_interval"] = (
        df["interval_min"].isna()
        | (df["interval_min"] >= MIN_INTERVAL_MIN)
    )

    return df


def find_candidate_target(
    future_times,
    future_values,
    current_time,
    horizon_min,
    min_window,
    max_window,
):
    """
    Select the future observation closest to the desired horizon,
    provided it falls inside the allowed time window.
    """

    elapsed = (
        (future_times - current_time)
        .dt.total_seconds()
        / 60.0
    )

    mask = (
        (elapsed >= min_window)
        & (elapsed <= max_window)
        & (elapsed <= MAX_TARGET_GAP_MIN)
    )

    if not mask.any():
        return None

    candidate_positions = np.where(
        mask.to_numpy()
    )[0]

    best_position = candidate_positions[
        np.argmin(
            np.abs(
                elapsed.iloc[candidate_positions].to_numpy()
                - horizon_min
            )
        )
    ]

    return {
        "target_index": int(best_position),
        "target_elapsed_min": float(
            elapsed.iloc[best_position]
        ),
        "target_glucose": float(
            future_values.iloc[best_position]
        ),
        "target_offset_min": float(
            abs(
                elapsed.iloc[best_position]
                - horizon_min
            )
        ),
    }

def audit_group(
    group,
    patient_id,
    dataset_split,
):
    group = group.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    times = group["timestamp"]
    values = group["glucose"]

    rows = []

    # A history must consist only of observations whose internal
    # consecutive gaps are >=5 minutes.
    for current_idx in range(
        len(group)
    ):
        for history_len in HISTORY_LENGTHS:

            start_idx = (
                current_idx
                - history_len
                + 1
            )

            if start_idx < 0:
                continue

            history = group.iloc[
                start_idx : current_idx + 1
            ]

            # All internal history transitions must be valid.
            internal_intervals = (
                history["interval_min"]
                .iloc[1:]
            )

            if (
                len(internal_intervals)
                != history_len - 1
            ):
                continue

            if (
                internal_intervals < MIN_INTERVAL_MIN
            ).any():
                continue

            current_time = times.iloc[
                current_idx
            ]

            future = group.iloc[
                current_idx + 1 :
            ]

            if future.empty:
                continue

            future_times = future[
                "timestamp"
            ]

            future_values = future[
                "glucose"
            ]

            for horizon_name, (
                min_window,
                max_window,
            ) in HORIZONS.items():

                horizon_min = {
                    "1h": 60.0,
                    "2h": 120.0,
                    "4h": 240.0,
                }[horizon_name]

                target = find_candidate_target(
                    future_times,
                    future_values,
                    current_time,
                    horizon_min,
                    min_window,
                    max_window,
                )

                if target is None:
                    continue

                rows.append(
                    {
                        "patient_id": patient_id,
                        "dataset_split": dataset_split,
                        "current_timestamp": current_time,
                        "history_length": history_len,
                        "horizon": horizon_name,
                        "target_glucose": target[
                            "target_glucose"
                        ],
                        "target_elapsed_min": target[
                            "target_elapsed_min"
                        ],
                        "target_offset_min": target[
                            "target_offset_min"
                        ],
                    }
                )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit usable SMBG training samples for "
            "history lengths 5/8/12 and horizons 1h/2h/4h."
        )
    )

    parser.add_argument(
        "--csv",
        default="data/raw/ohio_t1dm.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed",
    )

    args = parser.parse_args()

    df = prepare_smbg(args.csv)

    if df.empty:
        raise ValueError(
            "No valid FINGER_STICK observations found."
        )

    print(
        f"Loaded SMBG observations: {len(df):,}"
    )

    all_rows = []

    for (
        patient_id,
        dataset_split,
    ), group in df.groupby(
        ["patient_id", "dataset_split"],
        sort=True,
    ):
        print(
            f"  Auditing {patient_id} "
            f"[{dataset_split}] "
            f"({len(group):,} observations)"
        )

        result = audit_group(
            group,
            patient_id,
            dataset_split,
        )

        if not result.empty:
            all_rows.append(result)

    if not all_rows:
        raise RuntimeError(
            "No feasible SMBG training samples found."
        )

    samples = pd.concat(
        all_rows,
        ignore_index=True,
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Aggregate audit
    # ---------------------------------------------------------

    summary_rows = []

    total_raw_by_split = (
        df.groupby(
            "dataset_split"
        )
        .size()
        .to_dict()
    )

    for (
        dataset_split,
        history_len,
        horizon,
    ), group in samples.groupby(
        [
            "dataset_split",
            "history_length",
            "horizon",
        ],
        sort=True,
    ):
        summary_rows.append(
            {
                "dataset_split": dataset_split,
                "history_length": history_len,
                "horizon": horizon,
                "usable_samples": int(
                    len(group)
                ),
                "patients_with_samples": int(
                    group["patient_id"]
                    .nunique()
                ),
                "median_target_elapsed_min": float(
                    group[
                        "target_elapsed_min"
                    ].median()
                ),
                "p90_target_elapsed_min": float(
                    group[
                        "target_elapsed_min"
                    ].quantile(0.90)
                ),
                "median_target_offset_min": float(
                    group[
                        "target_offset_min"
                    ].median()
                ),
                "p90_target_offset_min": float(
                    group[
                        "target_offset_min"
                    ].quantile(0.90)
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # Add ALL split summary.
    overall_rows = []

    for (
        history_len,
        horizon,
    ), group in samples.groupby(
        [
            "history_length",
            "horizon",
        ],
        sort=True,
    ):
        overall_rows.append(
            {
                "dataset_split": "ALL",
                "history_length": history_len,
                "horizon": horizon,
                "usable_samples": int(
                    len(group)
                ),
                "patients_with_samples": int(
                    group["patient_id"]
                    .nunique()
                ),
                "median_target_elapsed_min": float(
                    group[
                        "target_elapsed_min"
                    ].median()
                ),
                "p90_target_elapsed_min": float(
                    group[
                        "target_elapsed_min"
                    ].quantile(0.90)
                ),
                "median_target_offset_min": float(
                    group[
                        "target_offset_min"
                    ].median()
                ),
                "p90_target_offset_min": float(
                    group[
                        "target_offset_min"
                    ].quantile(0.90)
                ),
            }
        )

    summary = pd.concat(
        [
            pd.DataFrame(overall_rows),
            summary,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Patient-level sample counts
    # ---------------------------------------------------------

    patient_summary = (
        samples.groupby(
            [
                "patient_id",
                "dataset_split",
                "history_length",
                "horizon",
            ],
            as_index=False,
        )
        .agg(
            usable_samples=(
                "target_glucose",
                "size",
            ),
            median_target_elapsed_min=(
                "target_elapsed_min",
                "median",
            ),
            p90_target_elapsed_min=(
                "target_elapsed_min",
                lambda x: x.quantile(0.90),
            ),
            median_target_offset_min=(
                "target_offset_min",
                "median",
            ),
        )
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    summary_path = (
        output_dir
        / "smbg_training_sample_audit.csv"
    )

    patient_path = (
        output_dir
        / "smbg_training_sample_patient.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    patient_summary.to_csv(
        patient_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    print("\n" + "=" * 105)
    print(
        "M5.4-C — SMBG TRAINING SAMPLE FEASIBILITY AUDIT"
    )
    print("=" * 105)

    print(
        "\nFiltering rules:"
    )
    print(
        f"  History lengths : {HISTORY_LENGTHS}"
    )
    print(
        "  Horizons        : 1h, 2h, 4h"
    )
    print(
        "  Target windows  : "
        "1h=45–75m, "
        "2h=90–150m, "
        "4h=180–300m"
    )
    print(
        f"  Min history interval : >= {MIN_INTERVAL_MIN:.0f} min"
    )
    print(
        f"  Max target gap       : <= {MAX_TARGET_GAP_MIN / 60:.0f} h"
    )

    print("\nUsable samples:")
    print(
        summary.sort_values(
            [
                "dataset_split",
                "history_length",
                "horizon",
            ]
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nOutputs:")
    print(
        f"  {summary_path}"
    )
    print(
        f"  {patient_path}"
    )

    print("=" * 105)


if __name__ == "__main__":
    main()