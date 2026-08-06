"""Tes peringatan divergensi (A2).

Versi lama hanya menyala bila kondisi kini "Dalam Target", sehingga tiga dari enam
perpindahan kategori tidak pernah tertangkap — termasuk ayunan hipoglikemia ke
hiperglikemia. Berkas ini menguji keenamnya secara eksplisit.
"""

import pytest

from src.alerts import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    evaluate_divergence,
)
from src.constants import CLASS_HYPER, CLASS_HYPO, CLASS_NORMAL

HORIZON = 30

# Nilai wakil tiap kategori (ambang: <70 hipo, >180 hiper).
G = {CLASS_HYPO: 55.0, CLASS_NORMAL: 120.0, CLASS_HYPER: 230.0}

# Keenam perpindahan kategori yang mungkin, beserta kegentingan yang diharapkan.
ENAM_PERPINDAHAN = [
    (CLASS_NORMAL, CLASS_HYPO, SEVERITY_WARNING),
    (CLASS_NORMAL, CLASS_HYPER, SEVERITY_WARNING),
    (CLASS_HYPO, CLASS_NORMAL, SEVERITY_INFO),
    (CLASS_HYPER, CLASS_NORMAL, SEVERITY_INFO),
    (CLASS_HYPO, CLASS_HYPER, SEVERITY_CRITICAL),
    (CLASS_HYPER, CLASS_HYPO, SEVERITY_CRITICAL),
]


@pytest.mark.parametrize("dari,ke,kegentingan", ENAM_PERPINDAHAN)
def test_keenam_perpindahan_menyalakan_peringatan(dari, ke, kegentingan):
    a = evaluate_divergence(G[dari], G[ke], HORIZON)

    assert a is not None, f"perpindahan {dari} -> {ke} tidak menyalakan peringatan"
    assert a.from_class == dari
    assert a.to_class == ke
    assert a.severity == kegentingan


@pytest.mark.parametrize("dari,ke,_kegentingan", ENAM_PERPINDAHAN)
def test_pesan_menyebut_kedua_kategori(dari, ke, _kegentingan):
    """Pesan harus menyebut kategori ASAL dan TUJUAN agar arahnya terbaca."""
    a = evaluate_divergence(G[dari], G[ke], HORIZON)

    assert a.from_label in a.message
    assert a.to_label in a.message
    assert a.from_label != a.to_label


@pytest.mark.parametrize("kelas", [CLASS_HYPO, CLASS_NORMAL, CLASS_HYPER])
def test_kategori_sama_tidak_divergen(kelas):
    assert evaluate_divergence(G[kelas], G[kelas], HORIZON) is None


def test_perubahan_besar_dalam_satu_kategori_tidak_divergen():
    """Yang diuji perpindahan KATEGORI, bukan besar selisih angka (Tabel III.1)."""
    assert evaluate_divergence(75.0, 175.0, HORIZON) is None


def test_arah_perpindahan_menghasilkan_pesan_berbeda():
    """hipo->hiper dan hiper->hipo sama-sama kritis tetapi tidak boleh sepesan."""
    naik = evaluate_divergence(G[CLASS_HYPO], G[CLASS_HYPER], HORIZON)
    turun = evaluate_divergence(G[CLASS_HYPER], G[CLASS_HYPO], HORIZON)

    assert naik.severity == turun.severity == SEVERITY_CRITICAL
    assert naik.message != turun.message
    # Penyebab klinis yang disebut memang berbeda arah.
    assert "karbohidrat berlebihan" in naik.message
    assert "insulin berlebihan" in turun.message


def test_ayunan_hipo_ke_hiper_dulu_tidak_tertangkap():
    """Regresi eksplisit atas cacat A2: syarat lama menuntut kondisi kini normal."""
    a = evaluate_divergence(G[CLASS_HYPO], G[CLASS_HYPER], HORIZON)

    assert a is not None
    assert a.from_class != CLASS_NORMAL  # justru kasus yang dulu terlewat
    assert a.is_deteriorating


def test_kembali_ke_target_tidak_ditandai_memburuk():
    a = evaluate_divergence(G[CLASS_HYPO], G[CLASS_NORMAL], HORIZON)

    assert a.severity == SEVERITY_INFO
    assert not a.is_deteriorating


def test_ambang_batas_memakai_definisi_constants():
    """70 dan 180 termasuk normal; 69,9 hipo dan 180,1 hiper."""
    assert evaluate_divergence(70.0, 180.0, HORIZON) is None
    assert evaluate_divergence(70.0, 180.1, HORIZON).to_class == CLASS_HYPER
    assert evaluate_divergence(69.9, 120.0, HORIZON).from_class == CLASS_HYPO


def test_horizon_dan_angka_ikut_terbawa():
    a = evaluate_divergence(112.0, 58.0, 60)

    assert a.horizon_minutes == 60
    assert a.current_glucose == 112.0
    assert a.predicted_glucose == 58.0
    assert "60 menit" in a.message
    assert "58 mg/dL" in a.message
