"""Tes T4.2 — logbook manual masuk ke jalur prediksi, lengkap dengan penjaganya.

Yang paling penting diuji di sini BUKAN bahwa penggabungan berhasil, melainkan bahwa
jendela yang tidak sepadan dengan sebaran pelatihan tertangkap. Kalau penjaga itu longgar,
aplikasi akan menampilkan prediksi atas jendela renggang tanpa tanda apa pun — dan
seorang dokter tidak punya cara membedakannya dari prediksi yang sah.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.logbook import (
    LAYAK, LUAR_SEBARAN, SUMBER_DATASET, SUMBER_MANUAL, TIDAK_CUKUP,
    baca_logbook, gabung_dengan_dataset, periksa_kelayakan,
)

PID = "ohio_559"


def deret_dataset(n=24, mulai="2024-01-01 08:00", tiap_menit=5, pid=PID):
    ts = pd.date_range(mulai, periods=n, freq=f"{tiap_menit}min")
    return pd.DataFrame({
        "timestamp": ts, "patient_id": pid,
        "glucose": [110.0 + i for i in range(n)],
        "carbs": 0.0, "insulin": 0.0, "activity": 0,
    })


def baris_manual(waktu, pid=PID, glucose=150.0, **kw):
    baris = {"timestamp": pd.Timestamp(waktu), "patient_id": pid, "glucose": glucose,
             "carbs": 40.0, "insulin": 2.0, "activity": 10,
             "stress": 7, "sleep": 0, "work": 1, "illness": 0,
             "meal_type": "makan siang", "notes": "tes", "source": "manual"}
    baris.update(kw)
    return baris


# ── membaca ──────────────────────────────────────────────────────────────────
def test_berkas_belum_ada_bukan_kesalahan(tmp_path):
    """Dokter yang belum mencatat apa pun adalah keadaan normal, bukan error."""
    df = baca_logbook(tmp_path / "belum_ada.csv")
    assert df.empty
    assert "glucose" in df.columns


def test_berkas_rusak_tidak_menjatuhkan_aplikasi(tmp_path):
    p = tmp_path / "rusak.csv"
    p.write_text("ini,bukan\ncsv;yang;benar,,,\n", encoding="utf-8")
    assert baca_logbook(p).empty or "timestamp" in baca_logbook(p).columns


def test_baris_bertimestamp_rusak_dibuang(tmp_path):
    p = tmp_path / "lb.csv"
    p.write_text("timestamp,patient_id,glucose\nbukan-tanggal,ohio_559,150\n"
                 "2024-01-01 09:00:00,ohio_559,160\n", encoding="utf-8")
    df = baca_logbook(p)
    assert len(df) == 1
    assert df["glucose"].iloc[0] == 160


# ── menggabungkan ────────────────────────────────────────────────────────────
def test_logbook_kosong_meninggalkan_dataset_apa_adanya():
    ds = deret_dataset()
    h = gabung_dengan_dataset(ds, pd.DataFrame(), PID)
    assert h.n_dataset == 24 and h.n_manual == 0
    assert not h.ada_manual


def test_catatan_manual_benar_benar_masuk_deret():
    ds = deret_dataset()
    lb = pd.DataFrame([baris_manual("2024-01-01 10:00")])
    h = gabung_dengan_dataset(ds, lb, PID)
    assert h.n_manual == 1
    assert SUMBER_MANUAL in set(h.deret["sumber"])
    assert h.deret["timestamp"].is_monotonic_increasing


def test_pasien_lain_tidak_ikut_tercampur():
    ds = deret_dataset()
    lb = pd.DataFrame([baris_manual("2024-01-01 10:00", pid="ohio_570")])
    h = gabung_dengan_dataset(ds, lb, PID)
    assert h.n_manual == 0


def test_catatan_tanpa_glukosa_dibuang():
    """Glukosa adalah fitur jangkar; tanpa itu barisnya tak bisa dipakai model."""
    ds = deret_dataset()
    lb = pd.DataFrame([baris_manual("2024-01-01 10:00", glucose=None)])
    h = gabung_dengan_dataset(ds, lb, PID)
    assert h.n_manual == 0


def test_catatan_manual_menimpa_baris_dataset_pada_waktu_sama():
    ds = deret_dataset()
    waktu = ds["timestamp"].iloc[5]
    lb = pd.DataFrame([baris_manual(waktu, glucose=999.0)])
    h = gabung_dengan_dataset(ds, lb, PID)
    assert h.n_manual_menimpa == 1
    assert len(h.deret) == 24  # menimpa, bukan menambah
    cocok = h.deret[h.deret["timestamp"] == waktu]
    assert float(cocok["glucose"].iloc[0]) == 999.0
    assert cocok["sumber"].iloc[0] == SUMBER_MANUAL


def test_kolom_di_luar_model_dilaporkan_diabaikan():
    """stress/sleep/work/illness tidak boleh diam-diam masuk sebagai fitur."""
    ds = deret_dataset()
    lb = pd.DataFrame([baris_manual("2024-01-01 10:00")])
    h = gabung_dengan_dataset(ds, lb, PID)
    for kol in ("stress", "sleep", "work", "illness", "meal_type", "notes"):
        assert kol in h.kolom_diabaikan, f"{kol} harus dilaporkan sebagai diabaikan"
        assert kol not in h.deret.columns, f"{kol} tidak boleh ikut ke deret model"


def test_kolom_numerik_kosong_diisi_nol_bukan_nan():
    ds = deret_dataset()
    lb = pd.DataFrame([baris_manual("2024-01-01 10:00", carbs=None, insulin=None)])
    h = gabung_dengan_dataset(ds, lb, PID)
    manual = h.deret[h.deret["sumber"] == SUMBER_MANUAL]
    assert float(manual["carbs"].iloc[0]) == 0.0
    assert float(manual["insulin"].iloc[0]) == 0.0


# ── penjaga kelayakan — bagian terpenting ────────────────────────────────────
def test_jendela_rapat_dinyatakan_layak():
    w = deret_dataset(12)
    w["sumber"] = SUMBER_DATASET
    k = periksa_kelayakan(w, sequence_length=12, max_gap_steps=6, cadence_min=5)
    assert k.verdict == LAYAK and k.boleh_diprediksi
    assert k.jeda_maks_langkah == 1.0


def test_jendela_kurang_baris_ditolak():
    w = deret_dataset(8)
    k = periksa_kelayakan(w, sequence_length=12, max_gap_steps=6)
    assert k.verdict == TIDAK_CUKUP and not k.boleh_diprediksi


def test_catatan_manual_berjarak_jauh_membuat_jendela_luar_sebaran():
    """Inti T4.2: catatan manual pada waktu bebas biasanya melanggar cadence pelatihan."""
    ds = deret_dataset(11)
    lb = pd.DataFrame([baris_manual("2024-01-01 20:00")])  # ~11 jam setelah baris terakhir
    h = gabung_dengan_dataset(ds, lb, PID)
    k = periksa_kelayakan(h.deret, sequence_length=12, max_gap_steps=6, cadence_min=5)
    assert k.verdict == LUAR_SEBARAN and not k.boleh_diprediksi
    assert k.n_manual == 1
    assert k.jeda_maks_langkah > 6
    assert "melampaui batas" in k.alasan


def test_catatan_manual_tepat_di_sela_cgm_tetap_layak():
    """Catatan yang dimasukkan berdekatan dengan pembacaan CGM tidak boleh ikut ditolak."""
    ds = deret_dataset(11)
    lb = pd.DataFrame([baris_manual(ds["timestamp"].iloc[-1] + pd.Timedelta(minutes=5))])
    h = gabung_dengan_dataset(ds, lb, PID)
    k = periksa_kelayakan(h.deret, sequence_length=12, max_gap_steps=6, cadence_min=5)
    assert k.verdict == LAYAK
    assert k.n_manual == 1


def test_jeda_tepat_pada_batas_masih_layak():
    """Batasnya inklusif, persis seperti create_sequences(max_gap_steps=...)."""
    ts = list(pd.date_range("2024-01-01 08:00", periods=11, freq="5min"))
    ts.append(ts[-1] + pd.Timedelta(minutes=30))  # tepat 6 langkah
    w = pd.DataFrame({"timestamp": ts, "glucose": 120.0, "sumber": SUMBER_DATASET})
    k = periksa_kelayakan(w, sequence_length=12, max_gap_steps=6, cadence_min=5)
    assert k.verdict == LAYAK
    assert k.jeda_maks_langkah == 6.0


def test_jeda_sedikit_di_atas_batas_ditolak():
    ts = list(pd.date_range("2024-01-01 08:00", periods=11, freq="5min"))
    ts.append(ts[-1] + pd.Timedelta(minutes=35))  # 7 langkah
    w = pd.DataFrame({"timestamp": ts, "glucose": 120.0, "sumber": SUMBER_DATASET})
    k = periksa_kelayakan(w, sequence_length=12, max_gap_steps=6, cadence_min=5)
    assert k.verdict == LUAR_SEBARAN


def test_max_gap_steps_tak_diketahui_tidak_dianggap_layak():
    """Tidak tahu bukan berarti aman — sikapnya harus konservatif."""
    w = deret_dataset(12)
    k = periksa_kelayakan(w, sequence_length=12, max_gap_steps=None)
    assert k.verdict == LUAR_SEBARAN and not k.boleh_diprediksi


def test_hanya_dua_belas_baris_terakhir_yang_dinilai():
    """Jeda lama di awal deret tidak boleh memblokir jendela terkini yang rapat."""
    lama = deret_dataset(3, mulai="2024-01-01 00:00")
    baru = deret_dataset(12, mulai="2024-01-02 00:00")
    w = pd.concat([lama, baru], ignore_index=True)
    w["sumber"] = SUMBER_DATASET
    k = periksa_kelayakan(w, sequence_length=12, max_gap_steps=6, cadence_min=5)
    assert k.verdict == LAYAK


@pytest.mark.parametrize("cadence", [5, 15])
def test_cadence_ikut_diperhitungkan(cadence):
    w = deret_dataset(12, tiap_menit=cadence)
    w["sumber"] = SUMBER_DATASET
    k = periksa_kelayakan(w, sequence_length=12, max_gap_steps=6, cadence_min=cadence)
    assert k.verdict == LAYAK
