"""Bukti klaim "prediksi multimodal": kontribusi insulin & karbohidrat pada prediksi.

Laporan mengklaim bahwa dengan representasi berbasis fisiologi (IOB/COB), kontribusi
gabungan insulin dan karbohidrat naik dari ~3,4% (fitur mentah per-catatan) menjadi ~22,3%
(fitur engineered) pada horizon +30 menit. Skrip ini menghitung kedua angka tersebut secara
langsung dari feature_importances_ Random Forest, sehingga klaim itu dapat direproduksi.

Dua model dilatih pada split per-pasien yang sama:
  A. Fitur MENTAH      : [glucose, carbs, insulin, activity]        -> target glukosa absolut
  B. Fitur ENGINEERED  : [glucose, glucose_delta, iob, cob,
                          activity, hour_sin, hour_cos]             -> target delta glukosa

Karena Random Forest memakai jendela look-back yang di-flatten, importance dijumlahkan
lintas seluruh langkah waktu untuk tiap fitur.

Keluaran: results/eval_prediksi/feature_importance.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import DataPreprocessor  # noqa: E402

HORIZON = 6  # +30 menit
SEED = 42

RAW_FEATURES = ["glucose", "carbs", "insulin", "activity"]
ENGINEERED_FEATURES = [
    "glucose", "glucose_delta", "iob", "cob", "activity", "hour_sin", "hour_cos",
]
# Fitur yang merepresentasikan insulin & karbohidrat pada tiap skema
RAW_MULTIMODAL = ["carbs", "insulin"]
ENG_MULTIMODAL = ["iob", "cob"]


def run(features: list[str], predict_delta: bool, cfg: dict, engineered: bool) -> dict:
    mc = cfg["model"]
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])

    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(df)
    if engineered:
        df = pre.engineer_features(df, **mc["feature_engineering"])
    pre.feature_columns = list(features)

    patients = sorted(df["patient_id"].unique())
    train_df, _ = pre.split_by_patient(df, patients[-2:])

    X, y, anchor = pre.create_sequences(
        train_df, mc["sequence_length"], HORIZON, return_anchor=True
    )
    n, seq, n_feat = X.shape
    target = (y - anchor) if predict_delta else y

    model = RandomForestRegressor(
        n_estimators=mc["random_forest"]["n_estimators"],
        max_depth=mc["random_forest"]["max_depth"],
        min_samples_split=mc["random_forest"]["min_samples_split"],
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X.reshape(n, seq * n_feat), target)

    # importance dijumlahkan lintas langkah waktu, lalu dinormalisasi ke persen
    imp = model.feature_importances_.reshape(seq, n_feat).sum(axis=0)
    imp = 100.0 * imp / imp.sum()
    return {f: round(float(v), 2) for f, v in zip(features, imp)}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    raw = run(RAW_FEATURES, predict_delta=False, cfg=cfg, engineered=False)
    eng = run(ENGINEERED_FEATURES, predict_delta=True, cfg=cfg, engineered=True)

    raw_mm = round(sum(raw[f] for f in RAW_MULTIMODAL), 2)
    eng_mm = round(sum(eng[f] for f in ENG_MULTIMODAL), 2)

    out = {
        "horizon_min": HORIZON * 5,
        "catatan": (
            "Importance dijumlahkan lintas langkah look-back, dinormalisasi ke persen. "
            "Kontribusi multimodal = insulin + karbohidrat."
        ),
        "fitur_mentah": {
            "features": raw,
            "kontribusi_insulin_karbohidrat_%": raw_mm,
        },
        "fitur_engineered": {
            "features": eng,
            "kontribusi_insulin_karbohidrat_%": eng_mm,
        },
    }

    print("Fitur mentah      :", raw)
    print(f"  -> insulin+carbs: {raw_mm}%")
    print("Fitur engineered  :", eng)
    print(f"  -> iob+cob      : {eng_mm}%")

    dest = ROOT / "results/eval_prediksi/feature_importance.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDisimpan ke {dest}")


if __name__ == "__main__":
    main()
