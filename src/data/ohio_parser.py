"""Parser for OhioT1DM XML format into standardized CSV for the project."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def parse_ohio_xml(xml_path: Path | str) -> pd.DataFrame:
    """Parse a single OhioT1DM XML file and return DataFrame with standardized columns."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    patient_id = root.get("id", "unknown")
    weight = root.get("weight", "99")
    insulin_type = root.get("insulin_type", "unknown")

    rows = []

    # Parse glucose_level events (5-minute readings)
    glucose_events = {}
    glucose_elem = root.find("glucose_level")
    if glucose_elem is not None:
        for event in glucose_elem.findall("event"):
            ts_str = event.get("ts")
            value_str = event.get("value")
            if ts_str and value_str:
                try:
                    ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    glucose = float(value_str)
                    glucose_events[ts] = glucose
                except (ValueError, TypeError):
                    pass

    # Parse meal events
    meals = {}
    meal_elem = root.find("meal")
    if meal_elem is not None:
        for meal_event in meal_elem.findall("event"):
            ts_str = meal_event.get("ts")
            carbs_str = meal_event.get("carbs")
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    carbs = float(carbs_str) if carbs_str else 0.0
                    meals[ts] = carbs
                except (ValueError, TypeError):
                    pass

    # Parse insulin (bolus + basal)
    bolus_times = set()
    basal_info = {}
    bolus_elem = root.find("bolus")
    if bolus_elem is not None:
        for insulin_event in bolus_elem.findall("event"):
            ts_str = insulin_event.get("ts")
            dose_str = insulin_event.get("dose")
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    dose = float(dose_str) if dose_str else 0.0
                    bolus_times.add(ts)
                    basal_info[ts] = {
                        "insulin": dose,
                        "insulin_type": "bolus",
                    }
                except (ValueError, TypeError):
                    pass

    basal_elem = root.find("basal")
    if basal_elem is not None:
        for insulin_event in basal_elem.findall("event"):
            ts_str = insulin_event.get("ts")
            dose_str = insulin_event.get("dose")
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    dose = float(dose_str) if dose_str else 0.0
                    if ts not in basal_info:
                        basal_info[ts] = {"insulin": dose, "insulin_type": "basal"}
                    else:
                        basal_info[ts]["insulin"] += dose
                except (ValueError, TypeError):
                    pass

    # Parse life events
    exercise_times = set()
    work_times = set()
    sleep_times = set()
    stress_level_events = {}
    illness_times = set()

    exercise_elem = root.find("exercise")
    if exercise_elem is not None:
        for event in exercise_elem.findall("event"):
            ts_str = event.get("ts")
            duration_str = event.get("duration")
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    exercise_times.add(ts)
                except (ValueError, TypeError):
                    pass

    work_elem = root.find("work")
    if work_elem is not None:
        for event in work_elem.findall("event"):
            ts_str = event.get("ts")
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    work_times.add(ts)
                except (ValueError, TypeError):
                    pass

    sleep_elem = root.find("sleep")
    if sleep_elem is not None:
        for event in sleep_elem.findall("event"):
            ts_str = event.get("ts")
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    sleep_times.add(ts)
                except (ValueError, TypeError):
                    pass

    stressors_elem = root.find("stressors")
    if stressors_elem is not None:
        for event in stressors_elem.findall("event"):
            ts_str = event.get("ts")
            level_str = event.get("level")
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    level = int(level_str) if level_str else 5
                    stress_level_events[ts] = level
                except (ValueError, TypeError):
                    pass

    illness_elem = root.find("illness")
    if illness_elem is not None:
        for event in illness_elem.findall("event"):
            ts_str = event.get("ts")
            if ts_str:
                try:
                    ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                    illness_times.add(ts)
                except (ValueError, TypeError):
                    pass

    # Aggregate events by timestamp (5-minute intervals from glucose readings)
    for ts in sorted(glucose_events.keys()):
        row = {
            "timestamp": ts.isoformat(),
            "patient_id": f"ohio_{patient_id}",
            "glucose": float(glucose_events[ts]),
            "carbs": float(meals.get(ts, 0.0)),
            "insulin": float(basal_info.get(ts, {}).get("insulin", 0.0)),
            "activity": 1 if ts in exercise_times else 0,
            "stress": float(stress_level_events.get(ts, 5)),
            "sleep": 1 if ts in sleep_times else 0,
            "work": 1 if ts in work_times else 0,
            "illness": 1 if ts in illness_times else 0,
            "meal_type": "bolus" if ts in bolus_times else "none",
            "source": "ohio_t1dm",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def process_ohio_dataset(ohio_root_dir: str | Path, output_csv: str | Path) -> None:
    """Process all OhioT1DM XML files and write merged CSV."""
    ohio_root = Path(ohio_root_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    xml_files = sorted(ohio_root.glob("**/[0-9]*-ws-*.xml"))

    for xml_file in xml_files:
        try:
            df = parse_ohio_xml(xml_file)
            if not df.empty:
                all_dfs.append(df)
                print(f"✓ Parsed {xml_file.name}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Error parsing {xml_file.name}: {e}")
            continue

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
        combined.to_csv(output_path, index=False)
        print(f"\n✓ Merged dataset saved: {output_path}")
        print(f"  Total rows: {len(combined)}")
        print(f"  Patients: {combined['patient_id'].nunique()}")
        print(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    else:
        print("No valid XML files found.")


if __name__ == "__main__":
    process_ohio_dataset(
        ohio_root_dir="data/raw/OhioT1DM",
        output_csv="data/raw/ohio_t1dm_merged.csv",
    )
