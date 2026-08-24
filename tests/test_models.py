"""Kontrak lengan pembanding Random Forest.

Berkas ini disesuaikan ketika parser OhioT1DM disatukan (Agustus 2026). Yang berubah
pada kontraknya:

1. Data masuk berupa SATU berkas gabungan bermetadata ``glucose_source`` dan
   ``dataset_split``, bukan dua CSV terpisah.
2. Jendela dibentuk dari WAKTU NYATA (menit). API berbasis langkah dicabut.
3. Pembagian latih/uji mengikuti ``dataset_split`` resmi, bukan menyisihkan dua
   pasien terakhir.
4. RF adalah lengan PEMBANDING, bukan produksi. Artefak modelnya tidak lagi disimpan
   secara bawaan — satu berkas berukuran ratusan megabita sedangkan yang dibutuhkan
   pembahasan hanyalah metriknya.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.rf_model import (
    RandomForestGlucoseModel,
    train_random_forest_from_config,
)


def test_rf_model_train_predict_and_reload(tmp_path):
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(120, 12, 5))
    y_train = rng.normal(loc=120, scale=20, size=(120,))
    X_test = rng.normal(size=(24, 12, 5))

    model = RandomForestGlucoseModel(
        {
            "data": {"seed": 42},
            "model": {
                "random_forest": {
                    "n_estimators": 50,
                    "max_depth": 8,
                    "min_samples_split": 2,
                }
            },
        }
    )
    history = model.train(X_train, y_train)
    preds = model.predict(X_test)

    assert "train_RMSE" in history
    assert len(preds) == len(X_test)

    model_path = tmp_path / "rf.pkl"
    model.save(str(model_path))
    assert model_path.exists()

    reloaded = RandomForestGlucoseModel()
    reloaded.load(str(model_path))
    preds_reload = reloaded.predict(X_test)

    assert np.allclose(preds, preds_reload)


def _tulis_dataset_gabungan(data_dir: Path) -> None:
    """Dataset sintetis bergaya OhioT1DM gabungan: bermetadata sumber dan pembagian."""
    timestamps = np.arange(
        "2024-01-01T00:00", "2024-01-02T00:00", dtype="datetime64[5m]"
    )
    batas_latih = int(len(timestamps) * 0.7)

    rows = []
    for idx, patient_id in enumerate(["P001", "P002", "P003", "P004"]):
        for pos, t in enumerate(timestamps):
            glucose = 95 + idx * 5 + np.sin(len(rows) / 20.0) * 10
            rows.append(
                {
                    "patient_id": patient_id,
                    "timestamp": str(t),
                    "glucose": float(glucose),
                    "carbs": float((len(rows) % 6 == 0) * 30),
                    "insulin": float((len(rows) % 6 == 0) * 3),
                    "activity": int(len(rows) % 4 == 0) * 15,
                    "stress": int(4 + (len(rows) % 3)),
                    # Metadata hasil parser gabungan. Pembagian resmi bersifat
                    # temporal DALAM-pasien: tiap pasien muncul di kedua sisi.
                    "glucose_source": "CGM",
                    "dataset_split": "train" if pos < batas_latih else "test",
                }
            )

    pd.DataFrame(rows).to_csv(data_dir / "ohio_t1dm.csv", index=False)


def _config_uji(data_dir: Path) -> dict:
    return {
        "data": {
            "output_dir": str(data_dir),
            "seed": 42,
            "unified_dataset": "ohio_t1dm.csv",
        },
        "model": {
            "sequence_length": 12,
            "features": ["glucose", "carbs", "insulin", "activity", "stress"],
            "predict_delta": False,
            "random_forest": {
                "n_estimators": 30,
                "max_depth": 8,
                "min_samples_split": 2,
            },
            "source_profiles": {
                "CGM": {
                    "sequence_length": 12,
                    "prediction_horizons_min": [30],
                    "target_tolerance_min": 2.5,
                    "max_history_gap_min": 30,
                    "min_history_interval_min": 0,
                }
            },
        },
    }


@pytest.fixture()
def lingkungan_uji(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    _tulis_dataset_gabungan(data_dir)

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_config_uji(data_dir), f)

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path, cfg_path
    finally:
        os.chdir(cwd)


def test_train_random_forest_from_config_runs(lingkungan_uji):
    tmp_path, cfg_path = lingkungan_uji

    hasil = train_random_forest_from_config(str(cfg_path))

    # Kembalian kini dipetakan per horizon, karena satu modalitas dapat memiliki
    # lebih dari satu horizon produksi.
    assert "h30m" in hasil
    assert "RMSE" in hasil["h30m"]

    berkas_metrik = tmp_path / "models" / "rf_cgm_h30m_metrics.json"
    assert berkas_metrik.exists()

    with open(berkas_metrik, encoding="utf-8") as f:
        metrik = json.load(f)

    assert "Clarke_A+B" in metrik
    assert metrik["source"] == "CGM"
    assert metrik["horizon_min"] == 30.0
    # Provenance pembagian WAJIB tercatat pada artefak, supaya angka ini tidak
    # pernah terbaca sebagai hasil pembagian lintas-pasien.
    assert "within-patient" in metrik["split"]


def test_artefak_model_tidak_disimpan_secara_bawaan(lingkungan_uji):
    """RF lengan pembanding: metrik cukup, artefaknya ratusan megabita."""
    tmp_path, cfg_path = lingkungan_uji

    train_random_forest_from_config(str(cfg_path))

    assert not (tmp_path / "models" / "rf_cgm_h30m.pkl").exists()
    assert not (
        tmp_path / "models" / "rf_cgm_h30m_inference_bundle.pkl"
    ).exists()


def test_artefak_model_disimpan_bila_diminta(lingkungan_uji):
    tmp_path, cfg_path = lingkungan_uji

    train_random_forest_from_config(str(cfg_path), simpan_model=True)

    assert (tmp_path / "models" / "rf_cgm_h30m.pkl").exists()
    assert (tmp_path / "models" / "rf_cgm_h30m_inference_bundle.pkl").exists()


def test_metadata_gabungan_yang_hilang_ditolak_dengan_jelas(tmp_path):
    """Gagal senyap adalah pokok seluruh audit pipeline ini.

    Dataset pra-perubahan parser tidak memiliki glucose_source/dataset_split. Ia harus
    ditolak dengan pesan yang menyebut cara memperbaikinya, bukan menghasilkan angka
    yang tampak sah di atas data yang salah.
    """
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "patient_id": "P001",
                "timestamp": "2024-01-01T00:00",
                "glucose": 100.0,
                "carbs": 0.0,
                "insulin": 0.0,
                "activity": 0,
                "stress": 4,
            }
        ]
    ).to_csv(data_dir / "ohio_t1dm.csv", index=False)

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_config_uji(data_dir), f)

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError, match="glucose_source|dataset_split"):
            train_random_forest_from_config(str(cfg_path))
    finally:
        os.chdir(cwd)
