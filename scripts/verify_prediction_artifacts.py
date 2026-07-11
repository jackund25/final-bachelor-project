#!/usr/bin/env python3
"""Verifikasi artefak prediksi: apakah model tersimpan MEREPRODUKSI angka laporan.

Skrip ini merealisasikan "Verifikasi Fungsional" (Bab V.9). Berbeda dari sekadar
mengecek keberadaan berkas, skrip ini memuat bundle inferensi yang benar-benar dipakai
aplikasi, menjalankannya ulang pada split hold-out yang sama, lalu membandingkan
metriknya terhadap metrik yang tercatat (models/rf_metrics_h{6,12}.json) dan terhadap
angka yang dilaporkan pada Bab VI. Perbedaan sekecil apa pun akan ditandai GAGAL.

Jalankan pada environment proyek (conda: diabetes-ta) agar versi pustaka sesuai
Tabel V.2 (scikit-learn 1.3.0): model di-pickle dengan versi tersebut.

    python scripts/verify_prediction_artifacts.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.utils.metrics import calculate_all_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
DATA_CSV = PROJECT_ROOT / "data" / "raw" / "ohio_t1dm_merged.csv"

# Angka yang dilaporkan pada Bab VI (Tabel VI.1). Toleransi ketat: artefak harus
# mereproduksi metrik ini, bukan sekadar mendekatinya.
REPORTED = {
    6:  {"RMSE": 22.60, "MAE": 15.10, "Clarke_A+B": 94.35},
    12: {"RMSE": 34.24, "MAE": 24.83, "Clarke_A+B": 86.94},
}
TOL = 0.05  # mg/dL / poin persen


def evaluate_horizon(horizon: int, cfg: dict) -> dict:
    """Muat bundle inferensi, jalankan pada hold-out, kembalikan metrik."""
    mc = cfg["model"]
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])

    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(df)
    df = pre.engineer_features(df, **mc["feature_engineering"])
    pre.feature_columns = list(mc["engineered_features"])

    patients = sorted(df["patient_id"].unique())
    _, test_df = pre.split_by_patient(df, patients[-2:])

    X, y, anchor = pre.create_sequences(
        test_df, mc["sequence_length"], horizon, return_anchor=True
    )

    with open(MODELS_DIR / f"rf_inference_bundle_h{horizon}.pkl", "rb") as f:
        bundle = pickle.load(f)

    n, seq, n_feat = X.shape
    X_scaled = bundle["scaler"].transform(X.reshape(-1, n_feat)).reshape(n, seq * n_feat)
    pred = bundle["model"].predict(X_scaled)
    if bundle["predict_delta"]:
        pred = pred + anchor

    return calculate_all_metrics(y, pred)


def main() -> int:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    ok = True

    if not DATA_CSV.exists():
        logger.error(f"Dataset tidak ditemukan: {DATA_CSV}")
        return 1

    for horizon, reported in REPORTED.items():
        label = f"h{horizon} (+{horizon * 5} menit)"
        bundle_path = MODELS_DIR / f"rf_inference_bundle_h{horizon}.pkl"
        metrics_path = MODELS_DIR / f"rf_metrics_h{horizon}.json"

        if not bundle_path.exists():
            logger.error(f"[{label}] bundle hilang: {bundle_path.name}")
            ok = False
            continue

        actual = evaluate_horizon(horizon, cfg)
        actual["Clarke_A+B"] = actual["Clarke_A"] + actual["Clarke_B"]

        logger.info(
            f"[{label}] reproduksi: RMSE {actual['RMSE']:.2f} | "
            f"MAE {actual['MAE']:.2f} | Clarke A+B {actual['Clarke_A+B']:.2f}%"
        )

        # 1) Cocok dengan metrik yang tersimpan saat pelatihan?
        if metrics_path.exists():
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in ("RMSE", "MAE", "Clarke_A+B"):
                if abs(actual[key] - saved[key]) > TOL:
                    logger.error(
                        f"[{label}] {key}: reproduksi {actual[key]:.2f} != "
                        f"tersimpan {saved[key]:.2f}"
                    )
                    ok = False
        else:
            logger.error(f"[{label}] metrik tersimpan hilang: {metrics_path.name}")
            ok = False

        # 2) Cocok dengan angka yang DILAPORKAN di Bab VI?
        for key, expected in reported.items():
            if abs(actual[key] - expected) > TOL:
                logger.error(
                    f"[{label}] {key}: reproduksi {actual[key]:.2f} != "
                    f"laporan {expected:.2f}"
                )
                ok = False

    logger.info("=" * 66)
    if ok:
        logger.info("HASIL: artefak mereproduksi seluruh angka Bab VI. VERIFIED.")
        return 0
    logger.error("HASIL: artefak TIDAK mereproduksi angka laporan. FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
