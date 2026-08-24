"""Jendela latih tidak boleh melintasi jeda sensor.

Baris OhioT1DM tidak berjarak seragam: 0,6% interval melebihi 5 menit dan yang
terpanjang mencapai 118 jam. Tanpa penyaringan, sebuah jendela dapat memperlakukan
lompatan berjam-jam sebagai satu langkah 5 menit.

PERUBAHAN API (Agustus 2026). ``create_sequences(prediction_horizon=<langkah>,
max_gap_steps=<langkah>)`` dicabut dan digantikan ``create_time_horizon_sequences``
yang bekerja dalam MENIT. Perubahan ini memberi satu jaminan tambahan secara
struktural: target tidak lagi dipilih berdasarkan POSISI BARIS melainkan berdasarkan
WAKTU BERLALU dari jangkar. Akibatnya jendela yang targetnya jatuh setelah jeda
otomatis gugur — dulu itu harus diperiksa terpisah lewat ``max_gap_steps``.

Yang tersisa untuk dijaga ``max_history_gap_min`` adalah jeda DI DALAM jendela
masukan, yang tetap tidak terdeteksi oleh pemilihan target.
"""

import numpy as np
import pandas as pd

from src.data.preprocessor import DataPreprocessor

SEQ = 12
HORIZON_MIN = 30.0
TOLERANSI_MIN = 2.5
# 30 menit = enam langkah cadence CGM 5-menit. Nilai yang sama dipakai profil CGM
# produksi pada config.model.source_profiles.
BATAS_JEDA_MIN = 30.0


def _prep():
    return DataPreprocessor(
        {"model": {"features": ["glucose", "carbs", "insulin", "activity"]}}
    )


def _frame(timestamps, patient="P1"):
    n = len(timestamps)
    return pd.DataFrame({
        "patient_id": patient,
        "timestamp": timestamps,
        "glucose": np.linspace(100, 160, n),
        "carbs": 0.0,
        "insulin": 0.0,
        "activity": 0.0,
        # Wajib sejak parser disatukan: jendela dibentuk per (pasien, modalitas).
        "glucose_source": "CGM",
    })


def _jendela(df, batas_jeda_min):
    return _prep().create_time_horizon_sequences(
        df,
        sequence_length=SEQ,
        horizon_min=HORIZON_MIN,
        target_tolerance_min=TOLERANSI_MIN,
        max_history_gap_min=batas_jeda_min,
    )


def test_jendela_tidak_melintasi_jeda():
    """Dua segmen rapat dipisah jeda 2 jam.

    Tanpa batas jeda, jendela yang jangkarnya berada di segmen kedua tetap boleh
    mewarisi riwayat dari segmen pertama. Dengan batas, jendela semacam itu gugur.
    """
    ts = list(pd.date_range("2024-01-01", periods=60, freq="5min"))
    ts += list(pd.date_range(ts[-1] + pd.Timedelta("2h"), periods=60, freq="5min"))
    df = _frame(ts)

    tanpa_batas, _ = _jendela(df, None)
    dengan_batas, _ = _jendela(df, BATAS_JEDA_MIN)

    # 43 jendela per segmen (jangkar 11..53 dan 71..113), ditambah 11 jendela yang
    # riwayatnya melintasi jeda (jangkar 60..70).
    assert len(tanpa_batas) == 97
    assert len(dengan_batas) == 86
    assert len(dengan_batas) == 43 * 2
    assert len(dengan_batas) < len(tanpa_batas)


def test_target_setelah_jeda_gugur_bahkan_tanpa_batas_jeda():
    """Jaminan struktural dari pemilihan target berbasis waktu.

    Enam jangkar terakhir segmen pertama (indeks 54..59) tidak memiliki observasi
    mana pun pada +30 menit, karena observasi berikutnya baru muncul 2 jam kemudian.
    Jendela-jendela itu gugur TANPA perlu batas jeda — dulu inilah yang harus
    diperiksa terpisah lewat pemeriksaan jalur menuju target.
    """
    ts = list(pd.date_range("2024-01-01", periods=60, freq="5min"))
    ts += list(pd.date_range(ts[-1] + pd.Timedelta("2h"), periods=60, freq="5min"))

    tanpa_batas, _ = _jendela(_frame(ts), None)

    # Bila target dipilih per POSISI BARIS, seluruh 103 posisi penuh akan terbentuk.
    posisi_penuh = 120 - (SEQ + 6) + 1
    assert posisi_penuh == 103
    assert len(tanpa_batas) == 97, "enam jendela bertarget-setelah-jeda harus gugur"


def test_jeda_tepat_di_batas_masih_diterima():
    """Hanya jeda yang MELEBIHI batas yang membuang jendela."""
    ts = list(pd.date_range("2024-01-01", periods=30, freq="5min"))
    ts += list(pd.date_range(ts[-1] + pd.Timedelta("30min"), periods=30, freq="5min"))

    hasil, _ = _jendela(_frame(ts), BATAS_JEDA_MIN)

    # Jeda 30 menit tepat sama dengan batas, sehingga riwayat yang melintasinya tetap
    # sah dan kedua segmen tersambung.
    assert len(hasil) == 38


def test_jeda_satu_langkah_melebihi_batas_dibuang():
    ts = list(pd.date_range("2024-01-01", periods=30, freq="5min"))
    ts += list(pd.date_range(ts[-1] + pd.Timedelta("35min"), periods=30, freq="5min"))

    hasil, _ = _jendela(_frame(ts), BATAS_JEDA_MIN)

    # Jeda 35 menit melebihi batas: kedua segmen terpisah, 13 jendela masing-masing.
    assert len(hasil) == 26
    assert len(hasil) == 13 * 2


def test_segmentasi_dilakukan_per_pasien():
    """Batas antar-pasien tidak boleh dianggap jeda waktu, dan sebaliknya."""
    ts_a = list(pd.date_range("2024-01-01", periods=40, freq="5min"))
    ts_b = list(pd.date_range("2024-06-01", periods=40, freq="5min"))
    df = pd.concat([_frame(ts_a, "P1"), _frame(ts_b, "P2")], ignore_index=True)

    hasil, _ = _jendela(df, BATAS_JEDA_MIN)

    # Lompatan Januari->Juni antar-pasien tidak membuang apa pun: pengelompokan
    # memisahkan keduanya lebih dulu.
    assert len(hasil) == 46
    assert len(hasil) == 23 * 2


def test_segmentasi_juga_memisahkan_modalitas():
    """CGM dan finger-stick adalah dua aliran observasi, bukan satu deret.

    Bila keduanya digabung dalam satu deret, jarak antar-modalitas akan terbaca
    sebagai jeda sensor — atau lebih buruk, sebagai observasi berturutan yang rapat.
    """
    ts = list(pd.date_range("2024-01-01", periods=40, freq="5min"))
    cgm = _frame(ts, "P1")
    smbg = _frame(ts, "P1")
    smbg["glucose_source"] = "FINGER_STICK"
    df = pd.concat([cgm, smbg], ignore_index=True)

    hasil, _ = _jendela(df, BATAS_JEDA_MIN)

    # Dua aliran terpisah untuk pasien yang SAMA: masing-masing menghasilkan 23.
    assert len(hasil) == 46


def test_batas_interpolasi_membuang_runtun_nan_panjang():
    """Runtun NaN melebihi batas dibuang, bukan diisi paksa."""
    ts = pd.date_range("2024-01-01", periods=40, freq="5min")
    df = _frame(list(ts))
    df.loc[10:25, "glucose"] = np.nan  # 16 baris NaN berurutan, jauh di atas batas 6

    hasil = _prep().handle_missing_values(df, max_interpolate_steps=6)

    assert hasil["glucose"].isna().sum() == 0, "tidak boleh ada NaN tersisa"
    assert len(hasil) < len(df), "runtun NaN panjang seharusnya membuang baris"
