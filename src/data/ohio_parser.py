"""Parser untuk OhioT1DM (format XML) menjadi CSV standar proyek.

Perbaikan penting (30 Jun 2026): versi sebelumnya membuang hampir seluruh data
insulin & makanan karena (1) salah nama atribut XML dan (2) join timestamp eksak.
Versi ini:
- Bolus  : membaca atribut `ts_begin` + `dose` (sebelumnya keliru mencari `ts`).
- Basal  : membaca `ts` + `value` (rate), diperlakukan sebagai step-function.
- Meal   : membaca `ts` + `carbs`.
- Alignment event ke grid CGM 5-menit memakai `merge_asof` (nearest, toleransi
  2.5 menit) — bukan kecocokan timestamp eksak.
- finger_stick (SMBG nyata) dipertahankan dalam dataset yang sama
  dengan metadata `glucose_source`.
- `dataset_split` mempertahankan pembagian resmi OhioT1DM:
  `*_training.xml` -> `train`, `*_testing.xml` -> `test`.
- Output akhir berupa satu CSV gabungan: `ohio_t1dm.csv`.

Catatan dataset (hasil scan 24 file): kanal `stressors` nyaris kosong (7 event di
seluruh 12 pasien), sehingga fitur `stress` secara inheren tak informatif di OhioT1DM.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET

import pandas as pd

_TS_FORMAT = "%d-%m-%Y %H:%M:%S"
_BIN_TOL = pd.Timedelta("2min30s")   # toleransi alignment ke grid CGM 5-menit
_STEP_HOURS = 5.0 / 60.0             # 1 langkah CGM = 5 menit (untuk basal delivered)


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, _TS_FORMAT)
    except (ValueError, TypeError):
        return None


def _events_to_df(root: ET.Element, tag: str, ts_attr: str, value_attrs: List[str]) -> pd.DataFrame:
    """Ekstrak <tag><event .../></tag> menjadi DataFrame[ts, <value_attrs...>]."""
    cols = ["ts"] + value_attrs
    el = root.find(tag)
    if el is None:
        return pd.DataFrame(columns=cols)
    rows = []
    for ev in el.findall("event"):
        ts = _parse_ts(ev.get(ts_attr))
        if ts is None:
            continue
        row = {"ts": ts}
        for a in value_attrs:
            raw = ev.get(a)
            try:
                row[a] = float(raw) if raw not in (None, "") else 0.0
            except (ValueError, TypeError):
                row[a] = 0.0
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def _align_sum(events: pd.DataFrame, base_ts: pd.Series, value_col: str) -> pd.Series:
    """Jumlahkan nilai event ke baris base (CGM) terdekat dalam toleransi bin."""
    out = pd.Series(0.0, index=base_ts.index)
    if events.empty:
        return out
    base_ref = pd.DataFrame({"ts": base_ts.values, "_idx": base_ts.index}).sort_values("ts")
    matched = pd.merge_asof(
        events[["ts", value_col]].sort_values("ts"),
        base_ref, on="ts", direction="nearest", tolerance=_BIN_TOL,
    ).dropna(subset=["_idx"])
    if matched.empty:
        return out
    summed = matched.groupby("_idx")[value_col].sum()
    out.loc[summed.index] = summed.values
    return out


def _align_presence(events: pd.DataFrame, base_ts: pd.Series) -> pd.Series:
    """Tandai 1 bila ada event dalam toleransi bin, selain itu 0."""
    out = pd.Series(0, index=base_ts.index, dtype=int)
    if events.empty:
        return out
    base_ref = pd.DataFrame({"ts": base_ts.values, "_idx": base_ts.index}).sort_values("ts")
    ev = events.copy()
    ev["_flag"] = 1
    matched = pd.merge_asof(
        ev[["ts", "_flag"]].sort_values("ts"),
        base_ref, on="ts", direction="nearest", tolerance=_BIN_TOL,
    ).dropna(subset=["_idx"])
    if not matched.empty:
        out.loc[matched["_idx"].unique()] = 1
    return out


def _basal_rate_stepwise(basal: pd.DataFrame, base_ts: pd.Series) -> pd.Series:
    """Rate basal berlaku sejak ts hingga event berikutnya (step-function)."""
    out = pd.Series(0.0, index=base_ts.index)
    if basal.empty:
        return out
    base_ref = pd.DataFrame({"ts": base_ts.values, "_idx": base_ts.index}).sort_values("ts")
    merged = pd.merge_asof(
        base_ref, basal[["ts", "value"]].sort_values("ts"),
        on="ts", direction="backward",
    )
    merged["value"] = merged["value"].fillna(0.0)
    out.loc[merged["_idx"].values] = merged["value"].values
    return out



def _get_dataset_split(xml_path: Path | str) -> str:
    """
    Feature creation / metadata engineering:
    infer the official OhioT1DM split from the source XML filename.

        *_training.xml -> train
        *_testing.xml  -> test

    This does not alter timestamps or create a new random split.
    """
    name = Path(xml_path).name.lower()

    if "-training.xml" in name:
        return "train"

    if "-testing.xml" in name:
        return "test"

    raise ValueError(
        f"Tidak dapat menentukan dataset_split dari nama file: {name}"
    )


def _build_feature_frame(
    root: ET.Element,
    base: pd.DataFrame,
    patient_id: str,
    dataset_split: str,
    glucose_source: str,
) -> pd.DataFrame:
    """Bangun frame fitur ter-align untuk timeline `base` (kolom: ts, glucose)."""
    base = base.sort_values("ts").reset_index(drop=True)
    ts = base["ts"]

    meal = _events_to_df(root, "meal", "ts", ["carbs"])
    bolus = _events_to_df(root, "bolus", "ts_begin", ["dose"])
    basal = _events_to_df(root, "basal", "ts", ["value"])
    exercise = _events_to_df(root, "exercise", "ts", ["intensity"])
    stressors = _events_to_df(root, "stressors", "ts", [])
    sleep = _events_to_df(root, "sleep", "ts", [])
    work = _events_to_df(root, "work", "ts", [])
    illness = _events_to_df(root, "illness", "ts", [])

    carbs = _align_sum(meal, ts, "carbs")
    bolus_dose = _align_sum(bolus, ts, "dose")
    basal_rate = _basal_rate_stepwise(basal, ts)
    basal_delivered = basal_rate * _STEP_HOURS
    insulin = bolus_dose + basal_delivered
    activity = _align_sum(exercise, ts, "intensity")

    df = pd.DataFrame({
        "timestamp": ts.dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "patient_id": f"ohio_{patient_id}",
        "glucose": base["glucose"].astype(float).values,
        "carbs": carbs.values,
        "insulin": insulin.values,
        "bolus_dose": bolus_dose.values,
        "basal_rate": basal_rate.values,
        "activity": activity.values,
        "stress": _align_presence(stressors, ts).values,
        "sleep": _align_presence(sleep, ts).values,
        "work": _align_presence(work, ts).values,
        "illness": _align_presence(illness, ts).values,
    })
    df["meal_type"] = (df["carbs"] > 0).map({True: "meal", False: "none"})

    # Feature creation / metadata engineering:
    # these fields do not exist in the raw XML; they are created here
    # from the source file and the type of glucose observation.
    df["source"] = "ohio_t1dm"
    df["glucose_source"] = glucose_source
    df["dataset_split"] = dataset_split

    return df


def parse_ohio_xml(xml_path: Path | str) -> pd.DataFrame:
    """Parse satu file XML OhioT1DM → DataFrame pada timeline CGM (5-menit)."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    patient_id = root.get("id", "unknown")

    cgm = _events_to_df(root, "glucose_level", "ts", ["value"]).rename(columns={"value": "glucose"})
    # Drop pembacaan glukosa tidak valid (<=0): mustahil secara fisiologis, indikasi
    # error sensor atau entri.
    cgm = cgm[cgm["glucose"] > 0].reset_index(drop=True)
    if cgm.empty:
        return pd.DataFrame()
    dataset_split = _get_dataset_split(xml_path)

    return _build_feature_frame(
        root,
        cgm,
        patient_id,
        dataset_split,
        "CGM",
    )


def parse_ohio_fingerstick(xml_path: Path | str) -> pd.DataFrame:
    """Parse satu file XML → DataFrame pada timeline SMBG nyata (finger_stick)."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    patient_id = root.get("id", "unknown")

    fs = _events_to_df(root, "finger_stick", "ts", ["value"]).rename(columns={"value": "glucose"})
    # Drop finger_stick tidak valid (<=0): ditemukan entri glukosa 0 mg/dL yang
    # mustahil secara fisiologis.
    fs = fs[fs["glucose"] > 0].reset_index(drop=True)
    if fs.empty:
        return pd.DataFrame()
    dataset_split = _get_dataset_split(xml_path)

    return _build_feature_frame(
        root,
        fs,
        patient_id,
        dataset_split,
        "FINGER_STICK",
    )


def _merge_and_write(parser_fn, xml_files: List[Path], output_path: Path, label: str) -> Optional[pd.DataFrame]:
    frames = []
    for xml_file in xml_files:
        try:
            df = parser_fn(xml_file)
            if not df.empty:
                frames.append(df)
                print(f"  + {xml_file.name}: {len(df)} baris")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! gagal {xml_file.name}: {exc}")
    if not frames:
        print(f"  ({label}) tidak ada data.")
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"  -> {label}: {output_path}  ({len(combined):,} baris, {combined['patient_id'].nunique()} pasien)")
    return combined


def process_ohio_dataset(
    ohio_root_dir: str | Path = "data/raw/OhioT1DM",
    output_csv: str | Path = "data/raw/ohio_t1dm.csv",
) -> None:
    """
    Proses semua XML OhioT1DM menjadi SATU CSV gabungan.

    Setiap row mempertahankan metadata:

        glucose_source
            - CGM
            - FINGER_STICK

        dataset_split
            - train
            - test

    Pembagian train/test mengikuti file resmi OhioT1DM:
        *_training.xml -> train
        *_testing.xml  -> test

    Tidak dilakukan random split.

    Output:
        data/raw/ohio_t1dm.csv

    File output akan di-overwrite setiap kali parser dijalankan.
    """

    ohio_root = Path(ohio_root_dir)
    output_path = Path(output_csv)

    xml_files = sorted(
        ohio_root.glob("**/[0-9]*-ws-*.xml")
    )

    if not xml_files:
        print(f"Tidak ada file XML di {ohio_root}")
        return

    print(
        f"Memproses {len(xml_files)} file XML "
        f"dari {ohio_root}"
    )

    # =========================================================
    # PARSE CGM
    # =========================================================

    print(
        "\n[1/2] Timeline CGM (5-menit)"
    )

    cgm_frames = []

    for xml_file in xml_files:
        try:
            df = parse_ohio_xml(xml_file)

            if not df.empty:
                cgm_frames.append(df)

                print(
                    f"  + CGM  {xml_file.name}: "
                    f"{len(df)} baris "
                    f"[{df['dataset_split'].iloc[0]}]"
                )

        except Exception as exc:
            print(
                f"  ! gagal CGM {xml_file.name}: "
                f"{exc}"
            )

    # =========================================================
    # PARSE FINGER-STICK / SMBG
    # =========================================================

    print(
        "\n[2/2] Timeline finger_stick (SMBG nyata)"
    )

    smbg_frames = []

    for xml_file in xml_files:
        try:
            df = parse_ohio_fingerstick(xml_file)

            if not df.empty:
                smbg_frames.append(df)

                print(
                    f"  + SMBG {xml_file.name}: "
                    f"{len(df)} baris "
                    f"[{df['dataset_split'].iloc[0]}]"
                )

        except Exception as exc:
            print(
                f"  ! gagal SMBG {xml_file.name}: "
                f"{exc}"
            )

    if not cgm_frames and not smbg_frames:
        print("Tidak ada data yang berhasil diparse.")
        return

    # =========================================================
    # COMBINE CGM + SMBG
    # =========================================================

    frames = []

    if cgm_frames:
        frames.extend(cgm_frames)

    if smbg_frames:
        frames.extend(smbg_frames)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # =========================================================
    # SORT
    # =========================================================

    combined = (
        combined
        .sort_values(
            [
                "patient_id",
                "timestamp",
                "glucose_source",
            ]
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # WRITE ONE DATASET
    # =========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Intentional overwrite:
    # setiap run menghasilkan dataset fresh dari XML.
    combined.to_csv(
        output_path,
        index=False,
    )

    # =========================================================
    # SUMMARY
    # =========================================================

    split_counts = (
        combined["dataset_split"]
        .value_counts()
        .to_dict()
    )

    source_counts = (
        combined["glucose_source"]
        .value_counts()
        .to_dict()
    )

    patient_counts = (
        combined["patient_id"]
        .nunique()
    )

    print(
        "\n=================================================="
    )

    print(
        f"Dataset gabungan: {output_path}"
    )

    print(
        f"Total rows      : {len(combined):,}"
    )

    print(
        f"Total patients  : {patient_counts}"
    )

    print(
        f"Glucose source  : {source_counts}"
    )

    print(
        f"Dataset split   : {split_counts}"
    )

    print(
        "=================================================="
    )

    # =========================================================
    # CGM SUMMARY
    # =========================================================

    if cgm_frames:

        cgm = pd.concat(
            cgm_frames,
            ignore_index=True,
        )

        nz_columns = [
            "carbs",
            "insulin",
            "bolus_dose",
            "basal_rate",
            "activity",
            "stress",
        ]

        nz = {
            column: round(
                (
                    cgm[column] > 0
                ).mean() * 100,
                2,
            )
            for column in nz_columns
        }

        print(
            "\nRingkasan CGM "
            "(% baris non-nol):"
        )

        print(nz)

    # =========================================================
    # SOURCE/SPLIT CROSS-TAB
    # =========================================================

    cross_tab = pd.crosstab(
        combined["glucose_source"],
        combined["dataset_split"],
    )

    print(
        "\nDistribusi source × split:"
    )

    print(cross_tab)


if __name__ == "__main__":
    process_ohio_dataset()