"""
Temporal audit for unified OhioT1DM.

Checks target availability at real-time horizons separately for CGM and
FINGER_STICK and separately for official train/test splits.

Usage:
    python src/data/temporal_audit.py
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


DEFAULT_HORIZONS = [
    30,
    60,
    90,
    120,
    180,
    240,
    360,
    480,
]


def audit_group(
    df: pd.DataFrame,
    horizon_min: float,
    tolerance_min: float,
) -> dict:
    valid = 0
    total = 0
    offsets = []

    for _, g in df.sort_values(
        ["patient_id", "timestamp"]
    ).groupby(
        ["patient_id", "glucose_source"],
        sort=False,
    ):
        g = g.reset_index(drop=True)

        if len(g) < 2:
            continue

        ts = pd.to_datetime(g["timestamp"])

        for i in range(len(g) - 1):
            total += 1

            target_time = (
                ts.iloc[i]
                + pd.Timedelta(
                    minutes=float(horizon_min)
                )
            )

            j = int(
                ts.searchsorted(
                    target_time,
                    side="left",
                )
            )

            candidates = []

            if j < len(ts):
                candidates.append(j)

            if j - 1 > i:
                candidates.append(j - 1)

            if not candidates:
                continue

            best = min(
                candidates,
                key=lambda k: abs(
                    (
                        ts.iloc[k]
                        - target_time
                    ).total_seconds()
                ),
            )

            if best <= i:
                continue

            offset = abs(
                (
                    ts.iloc[best]
                    - target_time
                ).total_seconds()
            ) / 60.0

            if offset <= tolerance_min:
                valid += 1
                offsets.append(offset)

    return {
        "rows": int(len(df)),
        "candidate_anchors": int(total),
        "valid_targets": int(valid),
        "availability_pct": (
            100.0 * valid / total
            if total else 0.0
        ),
        "median_target_offset_min": (
            float(np.median(offsets))
            if offsets else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/raw/ohio_t1dm.csv",
    )
    parser.add_argument(
        "--tolerance-cgm",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--tolerance-smbg",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=float,
        default=DEFAULT_HORIZONS,
    )
    args = parser.parse_args()

    df = pd.read_csv(
        args.csv,
        parse_dates=["timestamp"],
    )

    required = {
        "patient_id",
        "timestamp",
        "glucose",
        "glucose_source",
        "dataset_split",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    rows = []

    for source in ["CGM", "FINGER_STICK"]:
        tolerance = (
            args.tolerance_cgm
            if source == "CGM"
            else args.tolerance_smbg
        )

        for split in ["train", "test"]:
            subset = df[
                (df["glucose_source"] == source)
                & (df["dataset_split"] == split)
            ].copy()

            for horizon in args.horizons:
                result = audit_group(
                    subset,
                    horizon,
                    tolerance,
                )

                rows.append(
                    {
                        "glucose_source": source,
                        "dataset_split": split,
                        "horizon_min": horizon,
                        "tolerance_min": tolerance,
                        **result,
                    }
                )

    result = pd.DataFrame(rows)

    out = Path(
        "data/processed/temporal_audit.csv"
    )
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        out,
        index=False,
    )

    print("\nTEMPORAL AUDIT")
    print("=" * 90)
    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )
    print("=" * 90)
    print(
        f"Saved: {out}"
    )


if __name__ == "__main__":
    main()