"""Uji terima Fase 6 — pasien yang HANYA punya catatan logbook harus dapat diprediksi.

Sebelum perbaikan ini, `app/streamlit_app.py` menyusun daftar pasien dari dataset saja dan
menjalankan penjaga jumlah baris sebelum penggabungan logbook, sehingga catatan dokter atas
pasien baru tersimpan rapi tetapi tidak pernah sampai ke model.

Data yang dipakai di sini adalah **jendela nyata OhioT1DM**, bukan deret karangan, sehingga
jitter dan pola kejadiannya sepadan dengan data pelatihan. Rinciannya pada
`arsip/SKENARIO_UJI_FASE6.md`.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor
from src.logbook import LAYAK, TIDAK_CUKUP, gabung_dengan_dataset, periksa_kelayakan

BUNDLE = Path("models/gbm_inference_bundle_h6.pkl")

# Skenario A "Budi": 13 catatan berjarak 5 menit mulai 16.42, diambil dari jendela nyata
# ohio_570 indeks 13366. Format: (glukosa, karbohidrat, bolus, aktivitas).
SKENARIO_A = [
    (113, 0, 5.6, 0), (113, 90, 5.6, 0), (111, 0, 0.0, 0), (109, 0, 0.0, 0),
    (110, 0, 0.0, 0), (116, 0, 0.0, 0), (126, 0, 0.0, 0), (136, 0, 0.0, 0),
    (143, 0, 0.0, 0), (151, 0, 0.0, 0), (155, 0, 0.0, 0), (164, 0, 0.0, 0),
    (170, 0, 0.0, 0),
]


def _logbook(nama: str, mulai: datetime, baris, menit: int = 5,
             sumber: str = "CGM") -> pd.DataFrame:
    """Susun catatan logbook seperti yang ditulis halaman Input Logbook.

    ``glucose_source`` WAJIB ikut sejak parser OhioT1DM disatukan: kontrak data
    menolak bingkai tanpa kolom itu, dan pembentukan jendela mengelompokkan per
    (pasien, modalitas). Jalur produksi sudah memenuhinya — ``POST /api/logbook``
    memvalidasi nilainya terhadap {CGM, FINGER_STICK} — sehingga penambahan di sini
    menyelaraskan uji dengan perilaku yang sebenarnya berjalan.

    Skenario A memakai jendela nyata ohio_570 berjarak 5 menit, yaitu cadence CGM.
    """
    return pd.DataFrame({
        "timestamp": [mulai + timedelta(minutes=menit * i) for i in range(len(baris))],
        "patient_id": nama,
        "glucose": [float(b[0]) for b in baris],
        "carbs": [float(b[1]) for b in baris],
        "insulin": [float(b[2]) for b in baris],
        "activity": [int(b[3]) for b in baris],
        "glucose_source": sumber,
    })


@pytest.fixture(scope="module")
def bundel():
    if not BUNDLE.exists():
        pytest.skip("bundle model produksi tidak tersedia")
    return pickle.load(open(BUNDLE, "rb"))


def test_pasien_logbook_murni_lolos_kelayakan(bundel):
    """Deret 13 catatan berjarak 5 menit harus dinyatakan LAYAK meski tanpa data dataset."""
    lb = _logbook("Budi", datetime(2026, 8, 19, 16, 42), SKENARIO_A)
    kosong = pd.DataFrame(columns=["timestamp", "patient_id", "glucose",
                                   "carbs", "insulin", "activity"])

    hasil = gabung_dengan_dataset(kosong, lb, "Budi")

    assert hasil.n_dataset == 0, "tidak boleh ada baris dataset pada pasien logbook-murni"
    assert hasil.n_manual == len(SKENARIO_A)

    kelayakan = periksa_kelayakan(hasil.deret, bundel["sequence_length"],
                                  max_gap_steps=6, cadence_min=5.0)
    assert kelayakan.verdict == LAYAK, kelayakan.alasan
    assert kelayakan.boleh_diprediksi


def test_prakiraan_pasien_logbook_murni_sesuai_skenario(bundel):
    """Skenario A harus menghasilkan prakiraan hiperglikemia, sesuai berkas skenario."""
    lb = _logbook("Budi", datetime(2026, 8, 19, 16, 42), SKENARIO_A)
    fitur = DataPreprocessor({}).engineer_features(lb, **bundel["feature_engineering"])
    w = fitur.tail(bundel["sequence_length"]).reset_index(drop=True)

    # IOB wajib memakai insulin yang diketik dokter; bila NaN, bolusnya hilang diam-diam.
    assert not w["iob"].isna().any()
    assert w["iob"].max() > 0, "bolus yang diketik dokter harus tercermin pada IOB"

    X = w[bundel["features"]].values.astype(float)
    if bundel.get("scaler") is not None:
        X = bundel["scaler"].transform(X)
    pred = float(bundel["model"].predict(X.reshape(1, -1))[0]) + float(w["glucose"].iloc[-1])

    assert pred == pytest.approx(187.5, abs=1.0), f"prakiraan {pred:.1f}, diharapkan sekitar 187,5"
    assert pred > 180.0, "kondisi terprediksi harus hiperglikemia"


def test_jendela_renggang_ditolak_bukan_ditambal(bundel):
    """Catatan berjarak 45 menit melampaui batas 30 menit dan harus DITOLAK."""
    lb = _logbook("Budi", datetime(2026, 8, 19, 16, 42), SKENARIO_A, menit=45)
    kosong = pd.DataFrame(columns=["timestamp", "patient_id", "glucose",
                                   "carbs", "insulin", "activity"])

    hasil = gabung_dengan_dataset(kosong, lb, "Budi")
    kelayakan = periksa_kelayakan(hasil.deret, bundel["sequence_length"],
                                  max_gap_steps=6, cadence_min=5.0)

    assert not kelayakan.boleh_diprediksi
    assert "melampaui batas" in kelayakan.alasan


def test_catatan_kurang_dari_jendela_ditolak(bundel):
    """Sebelas catatan tidak cukup bagi jendela dua belas baris."""
    lb = _logbook("Budi", datetime(2026, 8, 19, 16, 42), SKENARIO_A[:11])
    kosong = pd.DataFrame(columns=["timestamp", "patient_id", "glucose",
                                   "carbs", "insulin", "activity"])

    hasil = gabung_dengan_dataset(kosong, lb, "Budi")
    kelayakan = periksa_kelayakan(hasil.deret, bundel["sequence_length"],
                                  max_gap_steps=6, cadence_min=5.0)

    assert kelayakan.verdict == TIDAK_CUKUP
    assert kelayakan.n_baris == 11
