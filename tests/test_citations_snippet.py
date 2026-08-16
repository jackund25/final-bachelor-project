"""Tes kontrak kutipan Sumber Rujukan.

Kutipan pada UI dipakai untuk MENCOCOKKAN hasil penelusuran dengan halaman dokumen
aslinya — bukti KNF-08 saat sidang. Tiga sifat yang wajib dipegang:

1. potongan panjang dipotong pada batas KALIMAT bila ada (Gao dkk. 2023 §V.A.1),
   dan jatuh kembali ke batas kata hanya bila tidak ada batas kalimat yang layak;
2. potongan pendek TIDAK ditandai terpotong;
3. teks utuh selalu tersedia, sehingga bagian yang tidak tampil tetap dapat diperiksa;
4. penanda keutuhan kalimat melaporkan potongan HASIL INDEXING, bukan hasil tampilan.
"""

from src.rag.citations import (
    SNIPPET_CHARS_DEFAULT,
    berakhir_di_batas_kalimat,
    bermula_di_batas_kalimat,
    build_source_list,
    potong_batas_kalimat,
)

_META = {
    "halaman_cetak": 23,
    "halaman_cetak_valid": True,
    "judul_lengkap": "Pedoman Petunjuk Praktis Terapi Insulin",
    "lembaga": "PERKENI",
    "tahun": 2021,
    "nama_dokumen": "KB-03_PERKENI-2021_Terapi-Insulin.pdf",
}

_PANJANG = (
    "Sebagai regimen awal dapat digunakan insulin basal dengan dosis 0,2 unit/kgBB "
    "per hari. Titrasi dilakukan setiap tiga hari sebesar 2-4 unit sampai sasaran "
    "glukosa darah puasa tercapai. Bila terjadi hipoglikemia, turunkan dosis 4 unit "
    "atau 10-20 persen dari dosis sebelumnya. "
) * 5


def _satu(teks: str, **kwargs):
    return build_source_list([{"text": teks, "source": "KB-03.pdf", "metadata": _META}], **kwargs)[0]


def test_snippet_berhenti_di_batas_kalimat():
    """Keluhan pengguna: kutipan berhenti di tengah kalimat sehingga maknanya kabur."""
    row = _satu(_PANJANG)

    assert row["snippet_terpotong"] is True
    assert row["cara_potong"] == "batas_kalimat"
    assert len(row["snippet"]) <= SNIPPET_CHARS_DEFAULT
    # Inti kontraknya: kutipan diakhiri tanda akhir kalimat, bukan elipsis.
    assert row["snippet"].endswith(".")
    assert not row["snippet"].endswith("…")
    assert _PANJANG.split()[0] in row["snippet"]


def test_titik_singkatan_dan_enumerasi_tidak_dianggap_akhir_kalimat():
    """Penjaga singkatan. Tanpa ini 'PERKENI hal. 12' terpotong jadi 'PERKENI hal.'."""
    teks = (
        "Sasaran terapi mengikuti PERKENI 2021 hal. 12 dan tabel 3. "
        "Rekomendasi dr. Andi menyebut dosis 0,2 unit/kgBB. "
        "Kadar 7.5 mmol/L setara 135 mg/dL. "
    ) * 8
    hasil = potong_batas_kalimat(teks, 300)

    assert hasil["cara_potong"] == "batas_kalimat"
    for palsu in ("hal.", "dr.", "tabel 3."):
        assert not hasil["snippet"].endswith(palsu)
    # Bilangan desimal tidak boleh menjadi titik potong.
    assert "7.5" in hasil["snippet"] or hasil["snippet"].endswith("kgBB.")


def test_jatuh_ke_batas_kata_bila_tak_ada_batas_kalimat_layak():
    """Daftar bernomor tanpa kalimat utuh harus tetap menghasilkan kutipan panjang.

    Kutipan pendek tidak berguna untuk mencocokkan ke PDF sumber (KNF-08), jadi
    lebih baik berhenti di batas kata daripada memangkas terlalu banyak.
    """
    teks = "a. Defek genetik sel beta b. Resistensi insulin c. Penyakit pankreas " * 12
    hasil = potong_batas_kalimat(teks, 300)

    assert hasil["cara_potong"] == "batas_kata"
    assert hasil["snippet"].endswith("…")
    assert len(hasil["snippet"]) > 300 * 0.6


def test_penanda_keutuhan_kalimat_menggambarkan_potongan_bukan_tampilan():
    """Menjawab 'apakah ada konteks yang tertinggal' di tepi potongan."""
    utuh = _satu("Ambang hipoglikemia adalah 70 mg/dL.")
    assert utuh["mulai_kalimat_utuh"] is True
    assert utuh["akhir_kalimat_utuh"] is True

    # Potongan yang diwarisi separuh kalimat dari potongan sebelumnya.
    sebagian = _satu("dan seterusnya dititrasi setiap tiga hari sampai sasaran")
    assert sebagian["mulai_kalimat_utuh"] is False
    assert sebagian["akhir_kalimat_utuh"] is False

    assert bermula_di_batas_kalimat("Hiperglikemia adalah kondisi.") is True
    assert berakhir_di_batas_kalimat("Hiperglikemia adalah kondisi.") is True


def test_teks_lengkap_selalu_utuh_dan_n_char_konsisten():
    row = _satu(_PANJANG)
    rapat = " ".join(_PANJANG.split())

    assert row["teks_lengkap"] == rapat
    assert row["n_char"] == len(rapat)
    assert len(row["teks_lengkap"]) > len(row["snippet"])


def test_potongan_pendek_tidak_ditandai_terpotong():
    """Penanda harus membedakan 'potongan memang pendek' dari 'tampilan dipotong'."""
    row = _satu("Ambang hipoglikemia adalah 70 mg/dL.")

    assert row["snippet_terpotong"] is False
    assert row["snippet"] == row["teks_lengkap"]
    assert not row["snippet"].endswith("…")


def test_halaman_tetap_dari_metadata_bukan_dari_teks():
    """Nomor halaman tidak boleh ikut berubah ketika panjang kutipan berubah."""
    pendek = _satu(_PANJANG, snippet_chars=50)
    panjang = _satu(_PANJANG, snippet_chars=900)

    assert pendek["page_label"] == "Hal. 23"
    assert panjang["page_label"] == "Hal. 23"
    assert pendek["teks_lengkap"] == panjang["teks_lengkap"]
