"""Uji verifikasi sitasi berbasis angka.

Kasus-kasus di bawah diturunkan dari perilaku NYATA yang terekam pada 24 Agustus
2026, bukan dikarang: advisory hipoglikemia yang menyebut ambang Level 1/2, dan
kekhawatiran bahwa ambang itu berasal dari ingatan model alih-alih dari dokumen.
"""

from src.rag.verifikasi_sitasi import verifikasi_angka_bersitasi


def test_angka_yang_ada_pada_potongan_dinyatakan_bersumber_dokumen():
    potongan = [
        {"text": "Nilai peringatan hipoglikemia (Level 1) <= 70 mg/dL. "
                 "Hipoglikemia signifikan secara klinis (Level 2) < 54 mg/dL."},
    ]
    hasil = verifikasi_angka_bersitasi(
        "Ambang Level 1 adalah 70 mg/dL dan Level 2 adalah 54 mg/dL [S1].",
        potongan,
    )
    assert hasil.tidak_terverifikasi == []
    assert hasil.proporsi_terverifikasi == 1.0
    assert {t.nilai for t in hasil.temuan} >= {70.0, 54.0}


def test_angka_yang_tidak_ada_pada_potongan_ditandai():
    """Inti modul: ambang benar secara klinis tetapi TIDAK ada di dokumen yang ditunjuk."""
    potongan = [
        {"text": "Jika glukokortikoid dikurangi atau dihentikan, pemantauan "
                 "glukosa darah mungkin perlu dilanjutkan."},
    ]
    hasil = verifikasi_angka_bersitasi(
        "Hipoglikemia signifikan secara klinis adalah di bawah 54 mg/dL [S1].",
        potongan,
    )
    assert len(hasil.tidak_terverifikasi) == 1
    assert hasil.tidak_terverifikasi[0].nilai == 54.0
    assert hasil.tidak_terverifikasi[0].penanda == [1]


def test_angka_dari_data_pasien_tidak_dianggap_klaim_pedoman():
    """Glukosa pasien sah walau tidak ada di dokumen — ia data, bukan klaim pedoman."""
    potongan = [{"text": "Pemantauan glukosa darah dilakukan secara ketat."}]
    hasil = verifikasi_angka_bersitasi(
        "Glukosa saat ini 95.0 mg/dL dengan prediksi 62.0 mg/dL [S1].",
        potongan,
        konteks_pasien="Glukosa sekarang : 95.0 mg/dL\nGlukosa prediksi : 62.0 mg/dL",
    )
    assert hasil.tidak_terverifikasi == []
    assert all(t.status == "data_pasien" for t in hasil.temuan)


def test_nomor_pada_penanda_tidak_ikut_diperiksa():
    """"[S4]" tidak boleh menghasilkan temuan angka 4."""
    potongan = [{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}]
    hasil = verifikasi_angka_bersitasi("Pemantauan dilakukan ketat [S4].", potongan)
    assert hasil.temuan == []


def test_penanda_gabungan_dibaca_seluruhnya():
    potongan = [
        {"text": "Pemantauan setiap 3 jam pertama."},
        {"text": "Ambang 54 mg/dL."},
    ]
    hasil = verifikasi_angka_bersitasi(
        "Pantau tiap 3 jam dan waspadai 54 mg/dL [S1, S2].",
        potongan,
    )
    assert hasil.tidak_terverifikasi == []
    assert {t.ditemukan_pada for t in hasil.temuan} == {1, 2}


def test_kalimat_tanpa_penanda_tidak_diperiksa():
    """Aturan 8 SYSTEM_PROMPT membolehkan menulis tanpa penanda; itu bukan pelanggaran."""
    hasil = verifikasi_angka_bersitasi(
        "Kadar glukosa turun 33 mg/dL dalam 30 menit.",
        [{"text": "tidak relevan"}],
    )
    assert hasil.temuan == []
    assert hasil.proporsi_terverifikasi == 1.0


def test_desimal_koma_dan_titik_disamakan():
    potongan = [{"text": "Insulin on board tercatat 3.40 unit."}]
    hasil = verifikasi_angka_bersitasi(
        "Insulin aktif sebesar 3,40 unit [S1].", potongan
    )
    assert hasil.tidak_terverifikasi == []


def test_penanda_di_luar_jangkauan_tidak_meledak():
    """Model kadang menulis [S9] padahal hanya ada 5 potongan."""
    hasil = verifikasi_angka_bersitasi(
        "Ambang 54 mg/dL [S9].", [{"text": "hanya satu potongan"}]
    )
    assert len(hasil.tidak_terverifikasi) == 1


def test_jawaban_kosong_aman():
    assert verifikasi_angka_bersitasi("", [{"text": "x"}]).temuan == []
