"""Jendela latih tidak boleh melintasi jeda sensor.

Baris OhioT1DM tidak berjarak seragam: 0,6% interval melebihi 5 menit dan yang
terpanjang mencapai 118 jam. Karena create_sequences() membentuk jendela per
POSISI BARIS, tanpa penyaringan sebuah jendela dapat memperlakukan lompatan
berjam-jam sebagai satu langkah 5 menit.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor

SEQ, HORIZON = 12, 6
SPAN = SEQ + HORIZON


def _prep():
    return DataPreprocessor({"model": {"features": ["glucose", "carbs", "insulin", "activity"]}})


def _frame(timestamps, patient="P1"):
    n = len(timestamps)
    return pd.DataFrame({
        "patient_id": patient,
        "timestamp": timestamps,
        "glucose": np.linspace(100, 160, n),
        "carbs": 0.0,
        "insulin": 0.0,
        "activity": 0.0,
    })


def test_windows_never_span_a_gap():
    """Dua segmen rapat dipisah jeda 2 jam harus menghasilkan tepat dua blok jendela."""
    ts = list(pd.date_range("2024-01-01", periods=60, freq="5min"))
    ts += list(pd.date_range(ts[-1] + pd.Timedelta("2h"), periods=60, freq="5min"))
    df = _frame(ts)
    pre = _prep()

    tanpa_batas, _ = pre.create_sequences(df, SEQ, HORIZON, max_gap_steps=None)
    dengan_batas, _ = pre.create_sequences(df, SEQ, HORIZON, max_gap_steps=6,
                                           source_interval_min=5)

    assert len(tanpa_batas) == 120 - SPAN + 1        # perilaku lama: seluruh posisi
    assert len(dengan_batas) == (60 - SPAN + 1) * 2  # tepat dua segmen mulus
    assert len(dengan_batas) < len(tanpa_batas)


def test_gap_exactly_at_limit_is_kept():
    """Jeda tepat sebesar batas masih sah; hanya yang MELEBIHI batas yang dibuang."""
    ts = list(pd.date_range("2024-01-01", periods=30, freq="5min"))
    # Jeda 30 menit = 6 langkah = tepat di batas.
    ts += list(pd.date_range(ts[-1] + pd.Timedelta("30min"), periods=30, freq="5min"))
    df = _frame(ts)
    pre = _prep()

    hasil, _ = pre.create_sequences(df, SEQ, HORIZON, max_gap_steps=6, source_interval_min=5)
    assert len(hasil) == 60 - SPAN + 1, "jeda tepat di batas seharusnya tidak membuang apa pun"


def test_gap_one_step_over_limit_is_dropped():
    ts = list(pd.date_range("2024-01-01", periods=30, freq="5min"))
    # Jeda 35 menit = 7 langkah = satu langkah melebihi batas.
    ts += list(pd.date_range(ts[-1] + pd.Timedelta("35min"), periods=30, freq="5min"))
    df = _frame(ts)
    pre = _prep()

    hasil, _ = pre.create_sequences(df, SEQ, HORIZON, max_gap_steps=6, source_interval_min=5)
    assert len(hasil) == (30 - SPAN + 1) * 2


def test_gap_check_covers_path_to_target_not_just_input_window():
    """Jeda yang jatuh di antara akhir jendela dan targetnya juga harus membuang jendela.

    Target berada `horizon` langkah setelah jendela masukan. Jeda di rentang itu
    membuat target tidak lagi berjarak `horizon` langkah sebenarnya, sehingga
    label yang dipelajari model salah.
    """
    # 18 baris rapat, lalu jeda panjang, lalu sisanya: satu-satunya jendela penuh
    # (i=0) memiliki jeda di antara akhir input (indeks 11) dan target (indeks 17).
    ts = list(pd.date_range("2024-01-01", periods=14, freq="5min"))
    ts += list(pd.date_range(ts[-1] + pd.Timedelta("6h"), periods=14, freq="5min"))
    df = _frame(ts)
    pre = _prep()

    tanpa_batas, _ = pre.create_sequences(df, SEQ, HORIZON, max_gap_steps=None)
    assert len(tanpa_batas) > 0, "tanpa batas seharusnya tetap membentuk jendela"

    with pytest.raises(ValueError):
        # Seluruh jendela memuat jeda -> tidak ada yang tersisa, dan itu memang benar.
        pre.create_sequences(df, SEQ, HORIZON, max_gap_steps=6, source_interval_min=5)


def test_segmentation_is_per_patient():
    """Batas antar-pasien tidak boleh dianggap jeda waktu, dan sebaliknya."""
    ts_a = list(pd.date_range("2024-01-01", periods=40, freq="5min"))
    ts_b = list(pd.date_range("2024-06-01", periods=40, freq="5min"))  # jauh di masa depan
    df = pd.concat([_frame(ts_a, "P1"), _frame(ts_b, "P2")], ignore_index=True)
    pre = _prep()

    hasil, _ = pre.create_sequences(df, SEQ, HORIZON, max_gap_steps=6, source_interval_min=5)
    # Dua pasien, masing-masing 40 baris rapat: lompatan Januari->Juni antar-pasien
    # tidak boleh membuang apa pun karena groupby memisahkannya lebih dulu.
    assert len(hasil) == (40 - SPAN + 1) * 2


def test_interpolation_limit_drops_long_nan_runs():
    """Runtun NaN melebihi batas dibuang, bukan diisi paksa."""
    ts = pd.date_range("2024-01-01", periods=40, freq="5min")
    df = _frame(list(ts))
    df.loc[10:25, "glucose"] = np.nan  # 16 baris NaN berurutan, jauh di atas batas 6
    pre = _prep()

    hasil = pre.handle_missing_values(df, max_interpolate_steps=6)
    assert hasil["glucose"].isna().sum() == 0, "tidak boleh ada NaN tersisa"
    assert len(hasil) < len(df), "runtun NaN panjang seharusnya membuang baris"
