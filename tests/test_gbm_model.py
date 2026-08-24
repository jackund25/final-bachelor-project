"""Tes prediktor produksi GBM, terutama jalur ketidakpastian per sampel.

Yang diuji di sini bukan akurasinya (itu urusan skrip evaluasi), melainkan KONTRAK
yang diandalkan aplikasi: bundle memuat model kuantil, sigma per sampel dapat
dihitung, dan nilainya tidak pernah negatif. Bila salah satu putus, interval
prediksi di UI menghilang tanpa error — persis kegagalan senyap yang menjadi alasan
modul ini dibuat.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.gbm_model import (
    STD_METHOD_QUANTILE,
    GBMGlucoseModel,
    train_gbm_from_config,
)


def _config(**model_extra):
    cfg = {"data": {"seed": 42}, "model": {"gradient_boosting": {"random_state": 42}}}
    cfg["model"].update(model_extra)
    return cfg


def test_gbm_train_predict_and_reload(tmp_path):
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(120, 12, 5))
    y_train = rng.normal(loc=120, scale=20, size=(120,))
    X_test = rng.normal(size=(24, 12, 5))

    model = GBMGlucoseModel(_config())
    history = model.train(X_train, y_train)
    preds = model.predict(X_test)

    assert "train_RMSE" in history
    assert len(preds) == len(X_test)

    model_path = tmp_path / "gbm.pkl"
    model.save(str(model_path))
    assert model_path.exists()

    reloaded = GBMGlucoseModel(_config())
    reloaded.load(str(model_path))
    assert np.allclose(preds, reloaded.predict(X_test))


def test_predict_std_satu_nilai_per_sampel_dan_tak_pernah_negatif():
    """Sigma wajib sepanjang masukan dan non-negatif.

    Kuantil 0,025 dan 0,975 dilatih TERPISAH sehingga tidak dijamin berurutan
    (*quantile crossing*). Tanpa pemangkasan pada nol, selisih negatif akan
    menghasilkan interval terbalik di UI.
    """
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(200, 6, 4))
    y_train = rng.normal(loc=120, scale=25, size=(200,))
    X_test = rng.normal(size=(40, 6, 4))

    model = GBMGlucoseModel(_config())
    model.train(X_train, y_train)
    sigma = model.predict_std(X_test)

    assert sigma is not None
    assert sigma.shape == (len(X_test),)
    assert np.all(sigma >= 0.0)
    assert np.all(np.isfinite(sigma))


def test_predict_std_none_tanpa_model_kuantil():
    """Tanpa model kuantil, sigma HARUS None — bukan nol.

    Nol akan menghasilkan interval selebar nol yang tampak sangat yakin, padahal
    artinya ketidakpastian tidak diketahui. Pemanggil wajib dapat membedakannya.
    """
    rng = np.random.default_rng(11)
    model = GBMGlucoseModel(_config(), with_uncertainty=False)
    model.train(rng.normal(size=(80, 6, 4)), rng.normal(loc=120, scale=20, size=(80,)))

    assert model.predict_std(rng.normal(size=(10, 6, 4))) is None


def test_bundle_memuat_model_kuantil(tmp_path):
    """Bundle produksi wajib membawa model kuantil, kalau tidak UI kehilangan interval.

    Kontrak artefaknya berubah ketika parser OhioT1DM disatukan: penamaan kini
    memuat modalitas dan horizon dalam MENIT (``gbm_cgm_h30m_*``), bukan jumlah
    langkah (``gbm_*_h6``), dan kembalian fungsi dipetakan per horizon.
    """
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamps = np.arange("2024-01-01T00:00", "2024-01-02T00:00", dtype="datetime64[5m]")
    batas_latih = int(len(timestamps) * 0.7)

    rows = []
    for idx, patient_id in enumerate(["P001", "P002", "P003", "P004"]):
        for pos, t in enumerate(timestamps):
            rows.append({
                "patient_id": patient_id,
                "timestamp": str(t),
                "glucose": float(95 + idx * 5 + np.sin(len(rows) / 20.0) * 10),
                "carbs": float((len(rows) % 6 == 0) * 30),
                "insulin": float((len(rows) % 6 == 0) * 3),
                "activity": int(len(rows) % 4 == 0) * 15,
                "stress": int(4 + (len(rows) % 3)),
                # Metadata wajib dari parser gabungan.
                "glucose_source": "CGM",
                "dataset_split": "train" if pos < batas_latih else "test",
            })
    pd.DataFrame(rows).to_csv(data_dir / "ohio_t1dm.csv", index=False)

    config = {
        "data": {
            "output_dir": str(data_dir),
            "seed": 42,
            "unified_dataset": "ohio_t1dm.csv",
        },
        "model": {
            "sequence_length": 12,
            "predict_delta": False,
            "features": ["glucose", "carbs", "insulin", "activity", "stress"],
            "gradient_boosting": {"random_state": 42},
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
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import os

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        metrics = train_gbm_from_config(str(cfg_path))
    finally:
        os.chdir(cwd)

    assert "h30m" in metrics
    assert "RMSE" in metrics["h30m"]

    bundle_path = tmp_path / "models" / "gbm_cgm_h30m_inference_bundle.pkl"
    assert bundle_path.exists()

    import pickle

    bundle = pickle.loads(bundle_path.read_bytes())
    assert bundle["std_method"] == STD_METHOD_QUANTILE
    assert bundle["std_models"]["low"] is not None
    assert bundle["std_models"]["high"] is not None
    assert bundle["model_family"] == "HistGradientBoostingRegressor"

    metrics_json = json.loads(
        (tmp_path / "models" / "gbm_cgm_h30m_metrics.json").read_text(encoding="utf-8"))
    assert "Clarke_A+B" in metrics_json
