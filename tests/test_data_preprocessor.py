"""Kontrak praproses dan pembentukan jendela.

PERUBAHAN API (Agustus 2026). ``create_sequences`` yang bekerja dalam LANGKAH dicabut
dan digantikan ``create_time_horizon_sequences`` yang bekerja dalam MENIT, seiring
parser OhioT1DM disatukan menjadi satu berkas bermetadata ``glucose_source``.

Konsekuensinya untuk berkas ini:

* kontrak data kini MEWAJIBKAN kolom ``glucose_source``, sehingga seluruh bingkai uji
  harus membawanya;
* jendela dikelompokkan per (pasien, modalitas), bukan per pasien saja;
* target ditentukan oleh waktu berlalu, bukan oleh posisi baris.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor


def _build_config():
    return {
        "model": {
            "features": ["glucose", "carbs", "insulin", "activity", "stress"],
        },
    }


def test_pembentukan_jendela_menuntut_baris_yang_cukup():
    """Empat baris tidak cukup untuk jendela 4 + satu target."""
    preprocessor = DataPreprocessor(_build_config())
    df = pd.DataFrame(
        {
            "patient_id": ["P001"] * 4,
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="5min"),
            "glucose": [100.0, 101.0, 102.0, 103.0],
            "carbs": [0.0, 0.0, 0.0, 0.0],
            "insulin": [0.0, 0.0, 0.0, 0.0],
            "activity": [0, 0, 0, 0],
            "stress": [5, 5, 5, 5],
            "glucose_source": "CGM",
        }
    )

    with pytest.raises(ValueError, match="No valid time-horizon sequences"):
        preprocessor.create_time_horizon_sequences(
            df,
            sequence_length=4,
            horizon_min=5.0,
            target_tolerance_min=2.5,
        )


def test_jendela_tidak_melintasi_batas_pasien():
    preprocessor = DataPreprocessor(_build_config())
    df = pd.DataFrame(
        {
            "patient_id": ["P001"] * 3 + ["P002"] * 3,
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00",
                "2024-01-01 00:05",
                "2024-01-01 00:10",
                "2024-01-02 00:00",
                "2024-01-02 00:05",
                "2024-01-02 00:10",
            ]),
            "glucose": [100.0, 101.0, 102.0, 110.0, 111.0, 112.0],
            "carbs": [0.0] * 6,
            "insulin": [0.0] * 6,
            "activity": [0] * 6,
            "stress": [5, 5, 5, 6, 6, 6],
            "glucose_source": "CGM",
        }
    )

    X, y = preprocessor.create_time_horizon_sequences(
        df,
        sequence_length=2,
        horizon_min=5.0,
        target_tolerance_min=2.5,
    )

    # Satu jendela per pasien. Bila batas pasien diabaikan, jendela ketiga akan
    # memasangkan riwayat P001 dengan target P002 — dan selisih satu harinya tidak
    # akan terlihat dari angka mana pun.
    assert X.shape == (2, 2, 5)
    assert y.tolist() == [102.0, 112.0]


def test_jendela_tidak_melintasi_batas_modalitas():
    """CGM dan finger-stick milik pasien yang sama adalah dua aliran terpisah.

    Menggabungkannya membuat pembacaan finger-stick terbaca seolah-olah observasi
    CGM berturutan, padahal karakteristik temporalnya berbeda jauh.
    """
    preprocessor = DataPreprocessor(_build_config())
    waktu = pd.date_range("2024-01-01", periods=3, freq="5min")
    df = pd.DataFrame(
        {
            "patient_id": ["P001"] * 6,
            "timestamp": list(waktu) * 2,
            "glucose": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
            "carbs": [0.0] * 6,
            "insulin": [0.0] * 6,
            "activity": [0] * 6,
            "stress": [5] * 6,
            "glucose_source": ["CGM"] * 3 + ["FINGER_STICK"] * 3,
        }
    )

    X, y = preprocessor.create_time_horizon_sequences(
        df,
        sequence_length=2,
        horizon_min=5.0,
        target_tolerance_min=2.5,
    )

    assert X.shape == (2, 2, 5)
    assert sorted(y.tolist()) == [102.0, 202.0]


def test_handle_missing_values_sorts_and_fills():
    preprocessor = DataPreprocessor(_build_config())
    df = pd.DataFrame(
        {
            "patient_id": ["P001", "P001", "P001"],
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:10", "2024-01-01 00:00", "2024-01-01 00:05"]
            ),
            "glucose": [102.0, np.nan, 101.0],
            "carbs": [0.0, 15.0, np.nan],
            "insulin": [0.0, 1.5, np.nan],
            "activity": [0, np.nan, 5],
            "stress": [5, 6, np.nan],
            "glucose_source": "CGM",
        }
    )

    cleaned = preprocessor.handle_missing_values(df)

    assert cleaned["timestamp"].is_monotonic_increasing
    assert cleaned.isnull().sum().sum() == 0


def test_iob_tidak_nan_pada_deret_gabungan():
    """Deret gabungan dataset + logbook manual tidak boleh menghasilkan IOB NaN.

    Baris dataset membawa kolom ``bolus_dose``, baris manual tidak. Tanpa penambalan,
    ``engineer_features`` memilih ``bolus_dose`` yang NaN pada baris manual, sehingga
    insulin yang diketik dokter TIDAK PERNAH terpakai. Kegagalan itu senyap: model
    gradient boosting menerima NaN secara bawaan sehingga tidak ada galat yang muncul.
    """
    preprocessor = DataPreprocessor(_build_config())
    df = pd.DataFrame(
        {
            "patient_id": ["P001"] * 6,
            "timestamp": pd.date_range("2024-01-01 00:00", periods=6, freq="5min"),
            "glucose": [120.0, 122.0, 124.0, 126.0, 128.0, 130.0],
            "carbs": [0.0, 0.0, 0.0, 20.0, np.nan, np.nan],
            "insulin": [0.05, 0.05, 0.05, 3.0, 0.0, 0.0],
            # tiga baris pertama dari dataset, tiga terakhir dari logbook manual
            "bolus_dose": [0.0, 0.0, 0.0, np.nan, np.nan, np.nan],
            "activity": [0, 0, 0, 0, 0, 0],
            "glucose_source": "CGM",
        }
    )

    hasil = preprocessor.engineer_features(df)

    assert not hasil["iob"].isna().any(), "IOB tidak boleh NaN pada baris manual"
    assert not hasil["cob"].isna().any(), "COB tidak boleh NaN pada baris manual"
    # Insulin 3,0 yang diketik dokter harus benar-benar masuk ke IOB, bukan diabaikan.
    assert hasil["iob"].iloc[3] == pytest.approx(3.0)
    assert hasil["iob"].iloc[4] < hasil["iob"].iloc[3], "IOB harus meluruh sesudahnya"
