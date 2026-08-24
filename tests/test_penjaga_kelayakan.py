"""Penjaga kelayakan jendela — realisasi M5.7 clinical readiness rules.

MENGAPA BERKAS INI ADA.

Sebelum penjaga ini, backend hanya memeriksa SATU syarat: jumlah baris riwayat. Jendela
yang memuat jeda sensor berjam-jam, atau pembacaan finger-stick berjarak dua menit, tetap
diteruskan ke model — dan model tetap mengeluarkan angka. Padahal pelatihan MEMBUANG
jendela semacam itu lewat ``max_history_gap_min`` dan ``min_history_interval_min``.

Akibatnya prediksi disajikan atas jendela yang modelnya tidak pernah lihat, tanpa tanda
apa pun bagi dokter. Itu kegagalan senyap, jenis paling mahal pada alat bantu klinis, dan
justru yang dinilai Skenario 4 pengujian dokter ("data tidak memadai").

Yang dijaga berkas ini:

1. Kriteria penyajian IDENTIK dengan kriteria pelatihan, dan dibaca dari BUNDLE model —
   bukan dari config, yang bisa sudah berubah sesudah model dilatih.
2. CGM dan finger-stick memakai batas yang BERBEDA, sesuai profil temporalnya.
3. Aturannya hidup di SATU tempat; jalur berbasis langkah hanya membungkus.
"""

import datetime as dt

import pandas as pd
import pytest

from src.logbook import (
    LAYAK,
    LUAR_SEBARAN,
    TERLALU_RAPAT,
    TIDAK_CUKUP,
    periksa_kelayakan,
    periksa_kelayakan_menit,
)


def _jendela(n, interval_menit, jeda_di=None, jeda_menit=0):
    base = dt.datetime(2026, 8, 24, 8, 0)
    rows, t = [], base
    for i in range(n):
        rows.append({
            "timestamp": t,
            "glucose": 150.0 + i,
            "carbs": 0.0,
            "insulin": 0.0,
            "activity": 0,
        })
        t += dt.timedelta(minutes=interval_menit + (jeda_menit if i == jeda_di else 0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Kriteria dasar
# ---------------------------------------------------------------------------

def test_jendela_rapat_dinyatakan_layak():
    k = periksa_kelayakan_menit(_jendela(12, 5), sequence_length=12, max_gap_min=30)

    assert k.verdict == LAYAK
    assert k.boleh_diprediksi
    assert k.jeda_maks_menit == 5.0


def test_baris_kurang_ditolak_sebelum_memeriksa_jeda():
    k = periksa_kelayakan_menit(_jendela(8, 5), sequence_length=12, max_gap_min=30)

    assert k.verdict == TIDAK_CUKUP
    assert not k.boleh_diprediksi
    assert "8 dari 12" in k.alasan


def test_jeda_melampaui_batas_ditolak():
    """Jendela yang melintasi jeda sensor tidak pernah dilihat model saat pelatihan."""
    k = periksa_kelayakan_menit(
        _jendela(12, 5, jeda_di=5, jeda_menit=85), sequence_length=12, max_gap_min=30
    )

    assert k.verdict == LUAR_SEBARAN
    assert not k.boleh_diprediksi
    assert k.jeda_maks_menit == 90.0
    assert k.batas_jeda_menit == 30.0
    # Alasannya harus menyebut ANGKA, bukan sekadar "tidak layak": dokter perlu tahu
    # seberapa jauh datanya menyimpang, bukan hanya bahwa ia menyimpang.
    assert "90" in k.alasan and "30" in k.alasan


def test_jeda_tepat_di_batas_masih_diterima():
    k = periksa_kelayakan_menit(
        _jendela(12, 5, jeda_di=5, jeda_menit=25), sequence_length=12, max_gap_min=30
    )

    assert k.verdict == LAYAK, "hanya jeda yang MELAMPAUI batas yang boleh ditolak"


def test_observasi_terlalu_rapat_ditolak():
    """Pembacaan near-duplicate lazim pada finger-stick dan dibuang saat pelatihan."""
    k = periksa_kelayakan_menit(
        _jendela(8, 2), sequence_length=8, max_gap_min=720, min_interval_min=5
    )

    assert k.verdict == TERLALU_RAPAT
    assert not k.boleh_diprediksi
    assert k.interval_min_menit == 2.0
    assert k.batas_interval_menit == 5.0


def test_batas_tidak_diketahui_tidak_dianggap_layak():
    """Tidak tahu ≠ boleh. Ketiadaan batas harus menahan prediksi, bukan meloloskannya."""
    k = periksa_kelayakan_menit(_jendela(12, 5), sequence_length=12, max_gap_min=None)

    assert k.verdict == LUAR_SEBARAN
    assert not k.boleh_diprediksi


# ---------------------------------------------------------------------------
# Pembungkus berbasis langkah
# ---------------------------------------------------------------------------

def test_pembungkus_langkah_sepadan_dengan_jalur_menit():
    """Satu aturan, dua satuan. Bila keduanya menyimpang, penjaga kehilangan arti."""
    df = _jendela(12, 5, jeda_di=5, jeda_menit=85)

    langkah = periksa_kelayakan(df, sequence_length=12, max_gap_steps=6, cadence_min=5)
    menit = periksa_kelayakan_menit(df, sequence_length=12, max_gap_min=30)

    assert langkah.verdict == menit.verdict
    assert langkah.jeda_maks_menit == menit.jeda_maks_menit
    # Medan berbasis langkah tetap terisi bagi pemanggil lama.
    assert langkah.jeda_maks_langkah == 18.0
    assert langkah.batas_langkah == 6


# ---------------------------------------------------------------------------
# Integrasi dengan bundle produksi
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def layanan():
    from backend.services.prediction_service import PredictionService

    return PredictionService()


def _artefak(layanan, sumber):
    arts = layanan._load_horizons(sumber)
    if not arts:
        pytest.skip(f"bundle produksi {sumber} tidak tersedia")
    return arts[0]


def test_batas_dibaca_dari_bundle_bukan_config(layanan):
    """Bundle merekam nilai yang BENAR-BENAR dipakai saat model itu dilatih.

    Config dapat sudah berubah sesudahnya; memakai config berarti menjaga dengan
    aturan yang berbeda dari aturan pembentuk modelnya.
    """
    cgm = _artefak(layanan, "CGM")

    assert cgm["sequence_length"] == 12
    assert cgm["max_history_gap_min"] == 30


def test_profil_cgm_dan_finger_stick_berbeda(layanan):
    """Dua modalitas, dua karakter temporal, dua batas."""
    cgm = _artefak(layanan, "CGM")
    fs = _artefak(layanan, "FINGER_STICK")

    assert cgm["sequence_length"] != fs["sequence_length"]
    assert fs["max_history_gap_min"] > cgm["max_history_gap_min"]
    assert fs["min_history_interval_min"] > 0, (
        "finger-stick membuang pembacaan near-duplicate; CGM tidak perlu"
    )


def test_penjaga_backend_menolak_jendela_berjeda(layanan):
    cgm = _artefak(layanan, "CGM")

    k = layanan._periksa_kelayakan(
        _jendela(12, 5, jeda_di=5, jeda_menit=85), cgm
    )

    assert not k.boleh_diprediksi
    assert k.verdict == LUAR_SEBARAN


def test_penjaga_backend_meloloskan_finger_stick_renggang(layanan):
    """Renggang itu WAJAR bagi finger-stick — jangan diperlakukan seperti CGM."""
    fs = _artefak(layanan, "FINGER_STICK")

    k = layanan._periksa_kelayakan(_jendela(8, 180), fs)

    assert k.boleh_diprediksi, k.alasan
