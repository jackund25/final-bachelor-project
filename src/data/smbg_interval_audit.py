"""
M5.4-B — SMBG Irregular-Horizon Audit v2

Purpose
-------
Audit the actual interval between consecutive FINGER_STICK observations
and identify suspicious near-duplicate measurements.

Added in v2:
1. Short-interval audit:
      <5 min
      <10 min
      <15 min
2. Interval quality classification:
      suspicious_<5m
      very_short_5_10m
      short_10_15m
      normal_>=15m
3. Cleaned distribution excluding intervals <5 minutes.
4. Candidate horizon coverage:
      <=4 hours
      <=8 hours
      <=12 hours
5. Patient-level and train/test summaries.

This script DOES NOT delete or modify the source CSV.
It only produces audit outputs.

Input:
    data/raw/ohio_t1dm.csv

Outputs:
    data/processed/smbg_interval_audit.csv
    data/processed/smbg_interval_patient.csv
    data/processed/smbg_interval_quality.csv
    data/processed/smbg_horizon_coverage.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "patient_id",
    "timestamp",
    "glucose_source",
    "dataset_split",
}


SHORT_INTERVAL_THRESHOLDS = [
    5.0,
    10.0,
    15.0,
]

HORIZON_CUTOFFS = [
    240.0,   # 4h
    480.0,   # 8h
    720.0,   # 12h
]


def summarize(values):
    values = pd.Series(values).dropna()

    if values.empty:
        return {
            "n_intervals": 0,
            "min_min": np.nan,
            "p10_min": np.nan,
            "p25_min": np.nan,
            "median_min": np.nan,
            "p75_min": np.nan,
            "p90_min": np.nan,
            "p95_min": np.nan,
            "max_min": np.nan,
        }

    return {
        "n_intervals": int(len(values)),
        "min_min": float(values.min()),
        "p10_min": float(values.quantile(0.10)),
        "p25_min": float(values.quantile(0.25)),
        "median_min": float(values.median()),
        "p75_min": float(values.quantile(0.75)),
        "p90_min": float(values.quantile(0.90)),
        "p95_min": float(values.quantile(0.95)),
        "max_min": float(values.max()),
    }


def classify_interval(interval_min):
    if interval_min < 5:
        return "suspicious_<5m"
    if interval_min < 10:
        return "very_short_5_10m"
    if interval_min < 15:
        return "short_10_15m"
    return "normal_>=15m"


def build_intervals(df):
    """
    Build consecutive SMBG intervals within each patient and official split.

    No cross-patient or cross-split interval is ever created.
    """
    df = (
        df.sort_values(
            [
                "patient_id",
                "dataset_split",
                "timestamp",
            ]
        )
        .copy()
    )

    df["next_timestamp"] = df.groupby(
        ["patient_id", "dataset_split"]
    )["timestamp"].shift(-1)

    df["next_glucose"] = df.groupby(
        ["patient_id", "dataset_split"]
    )["glucose"].shift(-1)

    df["interval_min"] = (
        df["next_timestamp"] - df["timestamp"]
    ).dt.total_seconds() / 60.0

    intervals = df[
        df["interval_min"].notna()
        & (df["interval_min"] > 0)
    ].copy()

    intervals["interval_quality"] = (
        intervals["interval_min"]
        .apply(classify_interval)
    )

    intervals["within_4h"] = (
        intervals["interval_min"] <= 240
    )
    intervals["within_8h"] = (
        intervals["interval_min"] <= 480
    )
    intervals["within_12h"] = (
        intervals["interval_min"] <= 720
    )

    return intervals


def patient_summary(intervals):
    rows = []

    for (
        patient_id,
        dataset_split,
    ), group in intervals.groupby(
        ["patient_id", "dataset_split"],
        sort=True,
    ):
        raw = summarize(
            group["interval_min"]
        )

        clean = group[
            group["interval_min"] >= 5
        ]

        clean_stats = summarize(
            clean["interval_min"]
        )

        rows.append(
            {
                "patient_id": patient_id,
                "dataset_split": dataset_split,

                # Raw distribution
                "n_intervals_raw": raw["n_intervals"],
                "raw_min_min": raw["min_min"],
                "raw_p10_min": raw["p10_min"],
                "raw_p25_min": raw["p25_min"],
                "raw_median_min": raw["median_min"],
                "raw_p75_min": raw["p75_min"],
                "raw_p90_min": raw["p90_min"],
                "raw_p95_min": raw["p95_min"],
                "raw_max_min": raw["max_min"],

                # Clean distribution
                "n_intervals_clean_ge5m": clean_stats[
                    "n_intervals"
                ],
                "clean_min_min": clean_stats[
                    "min_min"
                ],
                "clean_p10_min": clean_stats[
                    "p10_min"
                ],
                "clean_p25_min": clean_stats[
                    "p25_min"
                ],
                "clean_median_min": clean_stats[
                    "median_min"
                ],
                "clean_p75_min": clean_stats[
                    "p75_min"
                ],
                "clean_p90_min": clean_stats[
                    "p90_min"
                ],
                "clean_p95_min": clean_stats[
                    "p95_min"
                ],
                "clean_max_min": clean_stats[
                    "max_min"
                ],

                # Short interval counts
                "count_lt5m": int(
                    (group["interval_min"] < 5).sum()
                ),
                "count_lt10m": int(
                    (group["interval_min"] < 10).sum()
                ),
                "count_lt15m": int(
                    (group["interval_min"] < 15).sum()
                ),

                # Horizon coverage
                "pct_within_4h": (
                    100
                    * group["within_4h"].mean()
                ),
                "pct_within_8h": (
                    100
                    * group["within_8h"].mean()
                ),
                "pct_within_12h": (
                    100
                    * group["within_12h"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def quality_summary(intervals):
    rows = []

    for (
        dataset_split,
        quality,
    ), group in intervals.groupby(
        ["dataset_split", "interval_quality"],
        sort=True,
    ):
        rows.append(
            {
                "dataset_split": dataset_split,
                "interval_quality": quality,
                "count": int(len(group)),
                "pct_of_intervals": (
                    100
                    * len(group)
                    / len(
                        intervals[
                            intervals["dataset_split"]
                            == dataset_split
                        ]
                    )
                ),
                "median_interval_min": float(
                    group["interval_min"].median()
                ),
                "min_interval_min": float(
                    group["interval_min"].min()
                ),
                "max_interval_min": float(
                    group["interval_min"].max()
                ),
            }
        )

    return pd.DataFrame(rows)


def horizon_coverage_summary(intervals):
    rows = []

    for split in ["train", "test"]:
        subset = intervals[
            intervals["dataset_split"] == split
        ]

        if subset.empty:
            continue

        for cutoff in HORIZON_CUTOFFS:
            eligible = subset[
                subset["interval_min"] <= cutoff
            ]

            rows.append(
                {
                    "dataset_split": split,
                    "horizon_cutoff_min": cutoff,
                    "horizon_cutoff_hours": (
                        cutoff / 60.0
                    ),
                    "total_intervals": int(
                        len(subset)
                    ),
                    "eligible_intervals": int(
                        len(eligible)
                    ),
                    "eligible_pct": (
                        100
                        * len(eligible)
                        / len(subset)
                    ),
                    "eligible_clean_ge5m": int(
                        (
                            eligible["interval_min"]
                            >= 5
                        ).sum()
                    ),
                    "eligible_clean_pct": (
                        100
                        * (
                            eligible["interval_min"]
                            >= 5
                        ).mean()
                        if len(eligible)
                        else 0.0
                    ),
                }
            )

    # Overall
    for cutoff in HORIZON_CUTOFFS:
        eligible = intervals[
            intervals["interval_min"] <= cutoff
        ]

        rows.append(
            {
                "dataset_split": "ALL",
                "horizon_cutoff_min": cutoff,
                "horizon_cutoff_hours": (
                    cutoff / 60.0
                ),
                "total_intervals": int(
                    len(intervals)
                ),
                "eligible_intervals": int(
                    len(eligible)
                ),
                "eligible_pct": (
                    100
                    * len(eligible)
                    / len(intervals)
                    if len(intervals)
                    else 0.0
                ),
                "eligible_clean_ge5m": int(
                    (
                        eligible["interval_min"]
                        >= 5
                    ).sum()
                ),
                "eligible_clean_pct": (
                    100
                    * (
                        eligible["interval_min"]
                        >= 5
                    ).mean()
                    if len(eligible)
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit SMBG interval quality, "
            "near-duplicates, and realistic horizon coverage."
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

    df = pd.read_csv(
        args.csv,
        parse_dates=["timestamp"],
    )

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    # Only actual SMBG observations.
    df = df[
        df["glucose_source"] == "FINGER_STICK"
    ].copy()

    if df.empty:
        raise ValueError(
            "No FINGER_STICK rows found."
        )

    intervals = build_intervals(df)

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Patient summary
    # ---------------------------------------------------------

    patient_result = patient_summary(
        intervals
    )

    patient_path = (
        output_dir
        / "smbg_interval_patient.csv"
    )

    patient_result.to_csv(
        patient_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Quality summary
    # ---------------------------------------------------------

    quality_result = quality_summary(
        intervals
    )

    quality_path = (
        output_dir
        / "smbg_interval_quality.csv"
    )

    quality_result.to_csv(
        quality_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Horizon coverage
    # ---------------------------------------------------------

    horizon_result = horizon_coverage_summary(
        intervals
    )

    horizon_path = (
        output_dir
        / "smbg_horizon_coverage.csv"
    )

    horizon_result.to_csv(
        horizon_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Raw interval audit
    # ---------------------------------------------------------

    raw_stats = summarize(
        intervals["interval_min"]
    )

    clean = intervals[
        intervals["interval_min"] >= 5
    ]

    clean_stats = summarize(
        clean["interval_min"]
    )

    audit_rows = [
        {
            "scope": "overall_raw",
            **raw_stats,
        },
        {
            "scope": "overall_clean_ge5m",
            **clean_stats,
        },
    ]

    for split in ["train", "test"]:
        subset = intervals[
            intervals["dataset_split"] == split
        ]

        clean_subset = subset[
            subset["interval_min"] >= 5
        ]

        audit_rows.append(
            {
                "scope": f"{split}_raw",
                **summarize(
                    subset["interval_min"]
                ),
            }
        )

        audit_rows.append(
            {
                "scope": f"{split}_clean_ge5m",
                **summarize(
                    clean_subset["interval_min"]
                ),
            }
        )

    audit_result = pd.DataFrame(
        audit_rows
    )

    audit_path = (
        output_dir
        / "smbg_interval_audit.csv"
    )

    audit_result.to_csv(
        audit_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    print("\n" + "=" * 105)
    print("M5.4-B — SMBG IRREGULAR-HORIZON AUDIT v2")
    print("=" * 105)

    print("\nRAW interval distribution:")
    for key in [
        "n_intervals",
        "min_min",
        "p10_min",
        "p25_min",
        "median_min",
        "p75_min",
        "p90_min",
        "p95_min",
        "max_min",
    ]:
        print(
            f"  {key:24s}: "
            f"{raw_stats[key]:.2f}"
            if isinstance(
                raw_stats[key],
                float
            )
            else
            f"  {key:24s}: "
            f"{raw_stats[key]}"
        )

    print("\nCLEAN distribution (interval >= 5 min):")
    for key in [
        "n_intervals",
        "min_min",
        "p10_min",
        "p25_min",
        "median_min",
        "p75_min",
        "p90_min",
        "p95_min",
        "max_min",
    ]:
        print(
            f"  {key:24s}: "
            f"{clean_stats[key]:.2f}"
            if isinstance(
                clean_stats[key],
                float
            )
            else
            f"  {key:24s}: "
            f"{clean_stats[key]}"
        )

    print("\nShort-interval audit:")
    for threshold in SHORT_INTERVAL_THRESHOLDS:
        count = int(
            (
                intervals["interval_min"]
                < threshold
            ).sum()
        )

        pct = (
            100
            * count
            / len(intervals)
        )

        print(
            f"  < {threshold:>2.0f} min : "
            f"{count:5d} intervals "
            f"({pct:6.2f}%)"
        )

    print("\nInterval quality:")
    print(
        quality_result.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nHorizon coverage:")
    print(
        horizon_result.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nPatient summary:")
    print(
        patient_result.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nOutputs:")
    print(f"  {audit_path}")
    print(f"  {patient_path}")
    print(f"  {quality_path}")
    print(f"  {horizon_path}")

    print("=" * 105)


if __name__ == "__main__":
    main()