"""Tes pengukur waktu per tahap (A4, KNF-10)."""

import pytest

from src.timing import (
    NETWORK_STAGES,
    STAGE_FEATURES,
    STAGE_GENERATE,
    STAGE_ORDER,
    STAGE_RETRIEVE,
    StageTimer,
    percentile,
)


def test_durasi_tercatat_per_tahap():
    t = StageTimer()
    with t.measure(STAGE_FEATURES):
        sum(range(10000))

    assert STAGE_FEATURES in t.stages
    assert t.stages[STAGE_FEATURES] > 0


def test_tahap_yang_melempar_tetap_tercatat():
    """Permintaan yang gagal di tengah justru paling menarik waktunya.

    Kalau hanya yang sukses dicatat, p95 terlihat lebih baik daripada kenyataan.
    """
    t = StageTimer()
    with pytest.raises(ValueError):
        with t.measure(STAGE_RETRIEVE):
            raise ValueError("gagal")

    assert STAGE_RETRIEVE in t.stages
    assert t.stages[STAGE_RETRIEVE] >= 0


def test_pemanggilan_berulang_diakumulasi():
    t = StageTimer()
    t.record(STAGE_FEATURES, 0.1)
    t.record(STAGE_FEATURES, 0.2)

    assert t.stages[STAGE_FEATURES] == pytest.approx(0.3)


def test_pemisahan_lokal_dan_jaringan():
    t = StageTimer()
    t.record(STAGE_FEATURES, 0.3)
    t.record(STAGE_RETRIEVE, 0.1)
    t.record(STAGE_GENERATE, 3.6)

    assert t.local_total == pytest.approx(0.4)
    assert t.network_total == pytest.approx(3.6)
    assert t.total == pytest.approx(4.0)
    assert t.network_share() == pytest.approx(0.9)


def test_hanya_generasi_yang_dihitung_jaringan():
    """Kalau tahap lain ikut terhitung jaringan, proporsi LLM jadi menyesatkan."""
    assert NETWORK_STAGES == frozenset({STAGE_GENERATE})


def test_share_none_saat_kosong():
    assert StageTimer().network_share() is None


def test_as_dict_urut_sesuai_alur_dan_memuat_agregat():
    t = StageTimer()
    t.record(STAGE_GENERATE, 2.0)   # sengaja dicatat lebih dulu
    t.record(STAGE_FEATURES, 0.5)

    d = t.as_dict()
    kunci_tahap = [k for k in d if not k.startswith("_")]

    assert kunci_tahap == [STAGE_FEATURES, STAGE_GENERATE]
    assert kunci_tahap == [s for s in STAGE_ORDER if s in kunci_tahap]
    assert d["_total"] == pytest.approx(2.5)
    assert d["_lokal"] == pytest.approx(0.5)
    assert d["_jaringan"] == pytest.approx(2.0)


def test_tahap_tak_terduga_tetap_dilaporkan():
    """Tahap di luar STAGE_ORDER tidak boleh hilang diam-diam dari laporan."""
    t = StageTimer()
    t.record("tahap_baru", 1.0)

    d = t.as_dict()

    assert d["tahap_baru"] == pytest.approx(1.0)
    assert d["_total"] == pytest.approx(1.0)


@pytest.mark.parametrize("p,harapan", [(0, 1.0), (50, 3.0), (100, 5.0)])
def test_percentile_batas_dan_tengah(p, harapan):
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], p) == pytest.approx(harapan)


def test_percentile_interpolasi():
    # p75 dari [1,2,3,4] -> indeks 2,25 -> 3 + 0,25*(4-3) = 3,25
    assert percentile([1.0, 2.0, 3.0, 4.0], 75) == pytest.approx(3.25)


def test_percentile_satu_nilai():
    assert percentile([7.0], 95) == pytest.approx(7.0)


def test_percentile_kosong_melempar():
    with pytest.raises(ValueError):
        percentile([], 95)


def test_percentile_tidak_bergantung_urutan_masukan():
    acak = [5.0, 1.0, 4.0, 2.0, 3.0]

    assert percentile(acak, 50) == pytest.approx(3.0)
