"""Tes interval konformal per horizon (A3).

Cacat yang dikunci di sini: faktor 3,3 dulu dihardcode di dua tempat pada
app/streamlit_app.py, berasal dari kalibrasi yang lebih tua daripada modelnya, dan
dipakai untuk horizon mana pun.
"""

import json

import pytest

from src import conformal


@pytest.fixture
def kalibrasi(tmp_path, monkeypatch):
    """Arahkan modul ke direktori kalibrasi sementara berisi dua horizon berbeda."""
    monkeypatch.setattr(conformal, "CALIBRATION_DIR", tmp_path)
    conformal.clear_cache()

    (tmp_path / "conformal_h6.json").write_text(json.dumps({
        "horizon_steps": 6, "horizon_min": 30, "max_gap_steps": 6,
        "levels": {"95": {"conformal_normalized": {"q": 3.0, "coverage%": 95.2},
                          "conformal_absolute": {"q": 50.0, "coverage%": 96.1}},
                   "90": {"conformal_normalized": {"q": 2.2, "coverage%": 90.4},
                          "conformal_absolute": {"q": 37.0, "coverage%": 92.0}}},
    }), encoding="utf-8")
    (tmp_path / "conformal_h12.json").write_text(json.dumps({
        "horizon_steps": 12, "horizon_min": 60, "max_gap_steps": 6,
        "levels": {"95": {"conformal_normalized": {"q": 4.5, "coverage%": 95.8},
                          "conformal_absolute": {"q": 80.0, "coverage%": 96.3}}},
    }), encoding="utf-8")

    yield tmp_path
    conformal.clear_cache()


def test_faktor_berbeda_per_horizon(kalibrasi):
    """Inti A3: satu faktor tidak boleh dipakai untuk dua horizon."""
    assert conformal.conformal_factor(6, level=95) == 3.0
    assert conformal.conformal_factor(12, level=95) == 4.5


def test_lebar_interval_mengikuti_faktor_horizonnya(kalibrasi):
    """Lebar ditentukan faktor horizon yang dipakai, bukan satu faktor global.

    Catatan: fixture ini memakai q h12 > q h6 untuk menguji mekanismenya. Pada data
    NYATA justru sebaliknya (h6 q=3,31 vs h12 q=2,96) -- interval h12 tetap lebih lebar
    karena std antar-pohon di h12 jauh lebih besar, bukan karena faktornya. Jadi tes ini
    sengaja TIDAK mengklaim "horizon lebih panjang selalu lebih lebar".
    """
    lo6, hi6 = conformal.prediction_interval(150.0, 10.0, 6, level=95)
    lo12, hi12 = conformal.prediction_interval(150.0, 10.0, 12, level=95)

    assert (hi6 - lo6) == pytest.approx(2 * 3.0 * 10.0)
    assert (hi12 - lo12) == pytest.approx(2 * 4.5 * 10.0)
    assert (hi6 - lo6) != (hi12 - lo12)


def test_faktor_berbeda_per_tingkat_cakupan(kalibrasi):
    assert conformal.conformal_factor(6, level=90) == 2.2
    assert conformal.conformal_factor(6, level=95) == 3.0


def test_interval_simetris_terhadap_prediksi(kalibrasi):
    lo, hi = conformal.prediction_interval(150.0, 10.0, 6, level=95)

    assert lo == pytest.approx(150.0 - 3.0 * 10.0)
    assert hi == pytest.approx(150.0 + 3.0 * 10.0)


def test_horizon_tanpa_kalibrasi_mengembalikan_none(kalibrasi):
    """TIDAK ada faktor cadangan yang dihardcode -- lebih baik kosong daripada salah."""
    assert conformal.conformal_factor(24) is None
    assert conformal.prediction_interval(150.0, 10.0, 24) is None
    assert conformal.coverage_achieved(24) is None


def test_tanpa_std_mengembalikan_none(kalibrasi):
    assert conformal.prediction_interval(150.0, None, 6) is None


def test_cakupan_terukur_dibaca_apa_adanya(kalibrasi):
    assert conformal.coverage_achieved(6, level=95) == 95.2
    assert conformal.coverage_achieved(12, level=95) == 95.8


def test_tingkat_cakupan_tak_terkalibrasi_mengembalikan_none(kalibrasi):
    """h12 hanya punya level 95; meminta 90 tidak boleh diam-diam memakai 95."""
    assert conformal.conformal_factor(12, level=90) is None


def test_berkas_rusak_tidak_melempar(kalibrasi):
    (kalibrasi / "conformal_h99.json").write_text("{bukan json", encoding="utf-8")
    conformal.clear_cache()

    assert conformal.conformal_factor(99) is None


def test_varian_absolut_tersedia(kalibrasi):
    q = conformal.conformal_factor(6, level=95, variant=conformal.VARIANT_ABSOLUTE)

    assert q == 50.0
