"""Pemuatan data dan pembentukan jendela CGM bersama untuk rantai konformal.

Tiga skrip memakai modul ini: ``conformal_calibration.py`` (menghasilkan q),
``eval_cakupan_conformal.py`` (penaksir cakupan jujur T12b), dan
``eval_conformal_per_rentang.py`` (cakupan per rentang glukosa). Ketiganya HARUS
membentuk jendela dengan cara yang sama, karena q yang dikalibrasi skrip pertama
dipakai oleh dua skrip lainnya; jendela yang berbeda membuat cakupan yang diukur
bukan milik prosedur yang sebenarnya dijalankan.

Sebelumnya ketiga skrip menyalin blok pemuatan dan pembentukan jendelanya
masing-masing. Salinan itu ikut usang bersama-sama ketika ``create_sequences``
diganti ``create_time_horizon_sequences`` pada refactor modality-aware, dan
ketiganya berhenti dapat dijalankan sejak 23 Agustus 2026 tanpa ada yang
menyadarinya sampai berkas hasilnya diperiksa ulang. Satu sumber kebenaran di
sini mencegah pengulangannya.

Dua catatan tentang perpindahan API:

1. Data dimuat dari ``ohio_t1dm.csv`` lalu disaring ke CGM, bukan dari
   ``ohio_t1dm_merged.csv``. Kedua jalan menghasilkan baris yang SAMA (166.533
   baris, 12 pasien), tetapi berkas merged tidak punya kolom ``glucose_source``
   yang kini diwajibkan ``engineer_features``.

2. Horizon kini bersatuan MENIT, bukan jumlah langkah baris, dan jendela dipilih
   berdasarkan jarak waktu sebenarnya. Parameter jendela diambil dari
   ``model.source_profiles.CGM`` — sumber yang sama dengan
   ``scripts/train_condition_classifier.py``, sehingga jendela di sini sepadan
   dengan jendela yang dipakai melatih model produksi.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import DiabetesDataLoader  # noqa: E402
from src.data.preprocessor import DataPreprocessor  # noqa: E402

SOURCE = "CGM"
BERKAS = "ohio_t1dm.csv"


def profil_cgm(cfg: dict) -> dict:
    """Parameter jendela CGM dari config.model.source_profiles.CGM."""
    m = cfg["model"]
    prof = (m.get("source_profiles") or {}).get(SOURCE, {})
    return {
        "sequence_length": int(
            prof.get("sequence_length", m.get("sequence_length", 12))
        ),
        "target_tolerance_min": float(prof.get("target_tolerance_min", 2.5)),
        "max_history_gap_min": prof.get("max_history_gap_min", 30),
        "min_history_interval_min": float(prof.get("min_history_interval_min", 0.0)),
        "prediction_horizons_min": list(prof.get("prediction_horizons_min", [])),
    }


def muat_cgm(cfg: dict):
    """Muat data CGM dan rekayasa fiturnya. Mengembalikan ``(df, feats)``."""
    m = cfg["model"]

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv(BERKAS)

    if "glucose_source" not in df.columns:
        raise SystemExit(
            f"{BERKAS} tidak memiliki kolom glucose_source; "
            "dataset unified belum dibangun ulang."
        )

    df = df[df["glucose_source"].astype(str).str.upper() == SOURCE]
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(
        df, max_interpolate_steps=m.get("max_interpolate_steps")
    )

    use_eng = bool(m.get("use_engineered", True))
    if use_eng:
        df = prep.engineer_features(df, **(m.get("feature_engineering", {}) or {}))

    feats = list(m["engineered_features"] if use_eng else m["features"])
    kurang = [f for f in feats if f not in df.columns]
    if kurang:
        raise SystemExit("Fitur tidak tersedia setelah rekayasa: " + ", ".join(kurang))

    return df, feats


def seqs_cgm(cfg: dict, df, feats, pasien, horizon_min: float):
    """Jendela ``(X, y, anchor)`` untuk daftar pasien pada horizon (MENIT)."""
    prep = DataPreprocessor(cfg)
    prep.feature_columns = list(feats)
    p = profil_cgm(cfg)

    return prep.create_time_horizon_sequences(
        df[df["patient_id"].isin(pasien)],
        sequence_length=p["sequence_length"],
        horizon_min=float(horizon_min),
        target_tolerance_min=p["target_tolerance_min"],
        max_history_gap_min=p["max_history_gap_min"],
        return_anchor=True,
        min_history_interval_min=p["min_history_interval_min"],
    )
