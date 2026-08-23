"""
M5.4 — Target Construction Audit
================================

Audit target availability using STRICTER, clinically interpretable
time tolerances.

Input:
    data/raw/ohio_t1dm.csv

Output:
    data/processed/target_audit.csv
    data/processed/target_audit_summary.csv

The audit does NOT train a model and does NOT modify the dataset.

Key principle:
    horizon = desired elapsed time from anchor observation
    tolerance = maximum acceptable deviation from that horizon

Profiles:
    CGM:
        +30m ±2.5m
        +60m ±2.5m

    FINGER_STICK:
        +30m ±10m
        +60m ±10m
        +120m ±15m
        +240m ±30m

In addition, the script reports a sensitivity analysis using:
    strict
    moderate
    relaxed

This allows us to distinguish:
    1. how many valid targets exist;
    2. how close those targets actually are to the requested horizon;
    3. whether SMBG has enough samples for a defensible model.

IMPORTANT:
This script uses the OFFICIAL dataset_split from the parser.
It does not create a new train/test split.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------

TARGET_PROFILES = {
    "CGM": {
        "30m": (30.0, 2.5),
        "60m": (60.0, 2.5),
    },
    "FINGER_STICK": {
        "30m": (30.0, 10.0),
        "60m": (60.0, 10.0),
        "120m": (120.0, 15.0),
        "240m": (240.0, 30.0),
    },
}


SENSITIVITY_PROFILES = {
    "strict": {
        "CGM": {
            30.0: 2.5,
            60.0: 2.5,
        },
        "FINGER_STICK": {
            30.0: 10.0,
            60.0: 10.0,
            120.0: 15.0,
            240.0: 30.0,
        },
    },
    "moderate": {
        "CGM": {
            30.0: 5.0,
            60.0: 5.0,
        },
        "FINGER_STICK": {
            30.0: 15.0,
            60.0: 15.0,
            120.0: 30.0,
            240.0: 45.0,
        },
    },
    "relaxed": {
        "CGM": {
            30.0: 7.5,
            60.0: 7.5,
        },
        "FINGER_STICK": {
            30.0: 30.0,
            60.0: 30.0,
            120.0: 45.0,
            240.0: 60.0,
        },
    },
}


REQUIRED_COLUMNS = {
    "patient_id",
    "timestamp",
    "glucose",
    "glucose_source",
    "dataset_split",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def nearest_future_target(
    times: pd.Series,
    anchor_idx: int,
    horizon_min: float,
):
    """
    Find the nearest FUTURE glucose observation to:

        anchor_timestamp + horizon_min

    Returns:
        target_idx, actual_offset_min

    actual_offset_min is absolute deviation from the requested horizon.
    """

    target_time = (
        times.iloc[anchor_idx]
        + pd.Timedelta(minutes=float(horizon_min))
    )

    insert = int(
        times.searchsorted(
            target_time,
            side="left",
        )
    )

    candidates = []

    if insert < len(times):
        candidates.append(insert)

    if insert - 1 > anchor_idx:
        candidates.append(insert - 1)

    if not candidates:
        return None, None

    target_idx = min(
        candidates,
        key=lambda idx: abs(
            (
                times.iloc[idx]
                - target_time
            ).total_seconds()
        ),
    )

    if target_idx <= anchor_idx:
        return None, None

    actual_elapsed_min = (
        times.iloc[target_idx]
        - times.iloc[anchor_idx]
    ).total_seconds() / 60.0

    offset_min = abs(
        actual_elapsed_min - horizon_min
    )

    return target_idx, offset_min


def audit_group(
    group: pd.DataFrame,
    horizon_min: float,
    tolerance_min: float,
):
    """
    Audit one patient/source/split group.

    Every glucose observation can act as an anchor.
    The target is accepted only when its actual elapsed time is within
    the requested horizon ± tolerance.
    """

    group = (
        group
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(group) < 2:
        return {
            "anchor_rows": 0,
            "candidate_targets": 0,
            "valid_targets": 0,
            "availability_pct": 0.0,
            "median_actual_elapsed_min": np.nan,
            "median_abs_offset_min": np.nan,
            "p90_abs_offset_min": np.nan,
            "min_actual_elapsed_min": np.nan,
            "max_actual_elapsed_min": np.nan,
        }

    times = pd.to_datetime(
        group["timestamp"]
    )

    anchor_rows = len(group)
    candidate_targets = 0

    valid_targets = 0

    actual_elapsed = []
    abs_offsets = []

    for anchor_idx in range(
        len(group) - 1
    ):
        target_idx, offset_min = (
            nearest_future_target(
                times,
                anchor_idx,
                horizon_min,
            )
        )

        if target_idx is None:
            continue

        candidate_targets += 1

        elapsed_min = (
            times.iloc[target_idx]
            - times.iloc[anchor_idx]
        ).total_seconds() / 60.0

        actual_elapsed.append(
            elapsed_min
        )
        abs_offsets.append(
            offset_min
        )

        if offset_min <= tolerance_min:
            valid_targets += 1

    return {
        "anchor_rows": int(anchor_rows),
        "candidate_targets": int(candidate_targets),
        "valid_targets": int(valid_targets),
        "availability_pct": (
            100.0 * valid_targets / anchor_rows
            if anchor_rows
            else 0.0
        ),
        "candidate_target_pct": (
            100.0 * candidate_targets / anchor_rows
            if anchor_rows
            else 0.0
        ),
        "median_actual_elapsed_min": (
            float(np.median(actual_elapsed))
            if actual_elapsed
            else np.nan
        ),
        "median_abs_offset_min": (
            float(np.median(abs_offsets))
            if abs_offsets
            else np.nan
        ),
        "p90_abs_offset_min": (
            float(np.percentile(abs_offsets, 90))
            if abs_offsets
            else np.nan
        ),
        "min_actual_elapsed_min": (
            float(np.min(actual_elapsed))
            if actual_elapsed
            else np.nan
        ),
        "max_actual_elapsed_min": (
            float(np.max(actual_elapsed))
            if actual_elapsed
            else np.nan
        ),
    }


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def run_audit(
    csv_path: str,
    output_path: str,
):
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
    )

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset tidak memiliki kolom wajib: "
            f"{sorted(missing)}"
        )

    df = df.sort_values(
        [
            "patient_id",
            "glucose_source",
            "dataset_split",
            "timestamp",
        ]
    ).reset_index(drop=True)

    rows = []

    for sensitivity_name, profile in SENSITIVITY_PROFILES.items():

        for source, horizons in profile.items():

            for horizon_min, tolerance_min in horizons.items():

                for split in ["train", "test"]:

                    subset = df[
                        (df["glucose_source"] == source)
                        & (df["dataset_split"] == split)
                    ].copy()

                    if subset.empty:
                        result = {
                            "glucose_source": source,
                            "dataset_split": split,
                            "sensitivity": sensitivity_name,
                            "horizon_min": horizon_min,
                            "tolerance_min": tolerance_min,
                            "patient_count": 0,
                            "rows": 0,
                            "anchor_rows": 0,
                            "candidate_targets": 0,
                            "valid_targets": 0,
                            "availability_pct": 0.0,
                            "candidate_target_pct": 0.0,
                            "median_actual_elapsed_min": np.nan,
                            "median_abs_offset_min": np.nan,
                            "p90_abs_offset_min": np.nan,
                            "min_actual_elapsed_min": np.nan,
                            "max_actual_elapsed_min": np.nan,
                        }

                        rows.append(result)
                        continue

                    group_results = []

                    for _, group in subset.groupby(
                        ["patient_id"],
                        sort=False,
                    ):
                        group_results.append(
                            audit_group(
                                group,
                                horizon_min,
                                tolerance_min,
                            )
                        )

                    # Aggregate from patient-level counts rather than
                    # averaging percentages, so large patients do not
                    # receive artificial equal weighting.
                    anchor_rows = sum(
                        r["anchor_rows"]
                        for r in group_results
                    )

                    candidate_targets = sum(
                        r["candidate_targets"]
                        for r in group_results
                    )

                    valid_targets = sum(
                        r["valid_targets"]
                        for r in group_results
                    )

                    all_elapsed = []
                    all_offsets = []

                    for _, group in subset.groupby(
                        ["patient_id"],
                        sort=False,
                    ):
                        times = (
                            group
                            .sort_values("timestamp")
                            .reset_index(drop=True)["timestamp"]
                        )

                        for anchor_idx in range(
                            len(times) - 1
                        ):
                            target_idx, offset = (
                                nearest_future_target(
                                    times,
                                    anchor_idx,
                                    horizon_min,
                                )
                            )

                            if target_idx is None:
                                continue

                            elapsed = (
                                times.iloc[target_idx]
                                - times.iloc[anchor_idx]
                            ).total_seconds() / 60.0

                            all_elapsed.append(elapsed)
                            all_offsets.append(offset)

                    result = {
                        "glucose_source": source,
                        "dataset_split": split,
                        "sensitivity": sensitivity_name,
                        "horizon_min": horizon_min,
                        "tolerance_min": tolerance_min,
                        "patient_count": int(
                            subset["patient_id"].nunique()
                        ),
                        "rows": int(len(subset)),
                        "anchor_rows": int(anchor_rows),
                        "candidate_targets": int(candidate_targets),
                        "valid_targets": int(valid_targets),
                        "availability_pct": (
                            100.0 * valid_targets / anchor_rows
                            if anchor_rows
                            else 0.0
                        ),
                        "candidate_target_pct": (
                            100.0 * candidate_targets / anchor_rows
                            if anchor_rows
                            else 0.0
                        ),
                        "median_actual_elapsed_min": (
                            float(np.median(all_elapsed))
                            if all_elapsed
                            else np.nan
                        ),
                        "median_abs_offset_min": (
                            float(np.median(all_offsets))
                            if all_offsets
                            else np.nan
                        ),
                        "p90_abs_offset_min": (
                            float(np.percentile(all_offsets, 90))
                            if all_offsets
                            else np.nan
                        ),
                        "min_actual_elapsed_min": (
                            float(np.min(all_elapsed))
                            if all_elapsed
                            else np.nan
                        ),
                        "max_actual_elapsed_min": (
                            float(np.max(all_elapsed))
                            if all_elapsed
                            else np.nan
                        ),
                    }

                    rows.append(result)

    result = pd.DataFrame(rows)

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output,
        index=False,
    )

    # -------------------------------------------------------------
    # Recommended-profile summary
    # -------------------------------------------------------------

    strict = result[
        result["sensitivity"] == "strict"
    ].copy()

    summary = (
        strict[
            [
                "glucose_source",
                "dataset_split",
                "horizon_min",
                "tolerance_min",
                "rows",
                "valid_targets",
                "availability_pct",
                "median_actual_elapsed_min",
                "median_abs_offset_min",
                "p90_abs_offset_min",
            ]
        ]
        .sort_values(
            [
                "glucose_source",
                "horizon_min",
                "dataset_split",
            ]
        )
    )

    summary_path = (
        output.parent
        / "target_audit_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print("M5.4 — TARGET CONSTRUCTION AUDIT")
    print("=" * 110)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nOutput:")
    print(f"  Full sensitivity : {output}")
    print(f"  Strict summary   : {summary_path}")

    print("\nInterpretation:")
    print(
        "  availability_pct = valid target / all anchor observations"
    )
    print(
        "  median_abs_offset_min = median deviation from requested horizon"
    )
    print(
        "  p90_abs_offset_min = 90th percentile deviation"
    )
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit OhioT1DM target construction "
            "with strict/moderate/relaxed time tolerances."
        )
    )

    parser.add_argument(
        "--csv",
        default="data/raw/ohio_t1dm.csv",
    )

    parser.add_argument(
        "--output",
        default="data/processed/target_audit.csv",
    )

    args = parser.parse_args()

    run_audit(
        csv_path=args.csv,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()