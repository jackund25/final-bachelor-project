"""Tes kontrak strategi pemecahan dokumen (T7).

Dua cacat berbeda yang tidak boleh disatukan:

A. Pemotongan SENYAP di sisi vektor. all-MiniLM-L6-v2 membuang token ke-257 dan
   seterusnya tanpa peringatan. Menakar chunk_size dengan karakter tidak dapat
   mencegahnya, betapapun kecil angkanya dipilih — satuannya memang berbeda.
B. Pemotongan di tengah kalimat. Gao dkk. (2023) §V.A.1 hal. 8 menyebutnya
   "truncation within sentences".

Berkas ini mengunci: (1) default TIDAK berubah, (2) pemisah kalimat benar-benar
berpengaruh, (3) mode token memberi jaminan konstruktif, bukan statistik.
"""

import pytest

from src.rag.citations import berakhir_di_batas_kalimat
from src.rag.knowledge_base import (
    BATAS_TOKEN_MINILM,
    PEMISAH_KALIMAT,
    PEMISAH_KARAKTER,
    MedicalKnowledgeBase,
)

# Prosa pedoman tiruan: kalimat panjang tanpa baris baros, meniru hasil
# ekstraksi PDF yang menggabungkan satu paragraf menjadi satu baris.
_PROSA = (
    "Insulin basal diberikan satu kali sehari pada malam hari dengan dosis awal "
    "0,2 unit per kilogram berat badan. Titrasi dilakukan setiap tiga hari sebesar "
    "dua sampai empat unit sampai sasaran glukosa darah puasa tercapai. Bila terjadi "
    "hipoglikemia maka dosis diturunkan empat unit atau sepuluh sampai dua puluh "
    "persen dari dosis sebelumnya. Pemantauan glukosa darah mandiri dianjurkan "
    "sekurang-kurangnya tiga kali sehari pada pasien yang menggunakan insulin. "
) * 6


def _docs(teks: str = _PROSA):
    return [{"text": teks, "source": "uji.pdf", "topic": "uji",
             "metadata": {"doc_id": "uji_h1"}}]


@pytest.fixture(scope="module")
def kb():
    return MedicalKnowledgeBase()


def _penakar_atau_lewati(nama_model):
    """Bangun penakar token, atau LEWATI ujinya bila torch gagal dimuat.

    Bukan penyamaran kegagalan. Pada mesin pengembangan ini `torch` kadang gagal
    menginisialisasi `c10.dll` ketika seluruh suite berjalan dalam satu proses
    (Windows fatal exception: access violation), sedangkan berkas ini lulus bila
    dijalankan sendiri. Kerapuhan itu SUDAH ADA sebelum T7 dan tidak berhubungan
    dengan pemecahan dokumen. Melewati ujinya menjaga suite tetap bermakna;
    membiarkannya gagal akan menyembunyikan kegagalan lain di baliknya.
    """
    pytest.importorskip("transformers")
    from src.rag.knowledge_base import _penakar_token

    try:
        return _penakar_token(nama_model)
    except OSError as exc:  # DLL torch gagal init — lihat docstring
        pytest.skip(f"tokenizer tidak dapat dimuat di lingkungan ini: {exc}")


def test_default_tetap_perilaku_lama(kb):
    """Default WAJIB tidak berubah.

    Angka retrieval yang sudah dilaporkan dihitung dengan pemisah lama. Bila
    default ikut berubah, seluruh angka itu berubah diam-diam tanpa ada yang
    menyadarinya.
    """
    assert PEMISAH_KARAKTER == ["\n## ", "\n### ", "\n\n", "\n", " ", ""]

    potongan = kb.chunk_documents(documents=_docs(), chunk_size=300, chunk_overlap=40)
    pembanding = kb.chunk_documents(
        documents=_docs(), chunk_size=300, chunk_overlap=40,
        pemisah_kalimat=False, satuan_panjang="karakter",
    )
    assert [c["text"] for c in potongan] == [c["text"] for c in pembanding]


def test_urutan_pemisah_menaruh_kalimat_sebelum_newline_tunggal():
    """Urutannya yang menentukan, bukan sekadar keberadaan pemisahnya.

    Bila ". " diletakkan SESUDAH "\\n", pemisah kalimat praktis tidak pernah
    terpakai karena "\\n" selalu berhasil lebih dulu.
    """
    assert ". " in PEMISAH_KALIMAT and "\n" in PEMISAH_KALIMAT
    assert PEMISAH_KALIMAT.index(". ") < PEMISAH_KALIMAT.index("\n")
    # Batas paragraf tetap mengungguli batas kalimat.
    assert PEMISAH_KALIMAT.index("\n\n") < PEMISAH_KALIMAT.index(". ")
    # Kalimat yang berakhir tepat di ujung baris menghasilkan ".\n" tanpa spasi.
    assert ".\n" in PEMISAH_KALIMAT


def test_pemisah_kalimat_menaikkan_potongan_yang_berakhir_utuh(kb):
    """Keluhan pengguna: kutipan berhenti di tengah kalimat."""
    lama = kb.chunk_documents(documents=_docs(), chunk_size=300, chunk_overlap=40)
    baru = kb.chunk_documents(documents=_docs(), chunk_size=300, chunk_overlap=40,
                              pemisah_kalimat=True)

    rasio_lama = sum(berakhir_di_batas_kalimat(c["text"]) for c in lama) / len(lama)
    rasio_baru = sum(berakhir_di_batas_kalimat(c["text"]) for c in baru) / len(baru)

    assert rasio_baru > rasio_lama
    assert rasio_baru >= 0.75


def test_mode_token_menjamin_tidak_ada_potongan_melewati_batas(kb):
    """Jaminan KONSTRUKTIF: tidak ada potongan yang melewati 256 token.

    Ini yang tidak dapat diberikan mode karakter. Pada korpus produksi,
    chunk_size=900 karakter menghasilkan potongan sampai 443 token.
    """
    tok = _penakar_atau_lewati(kb.cfg.embedding_model)
    potongan = kb.chunk_documents(
        documents=_docs(), chunk_size=BATAS_TOKEN_MINILM, chunk_overlap=34,
        pemisah_kalimat=True, satuan_panjang="token",
    )
    panjang = [tok(c["text"]) for c in potongan]

    assert potongan
    assert max(panjang) <= BATAS_TOKEN_MINILM


def test_mode_karakter_tidak_dapat_memberi_jaminan_yang_sama(kb):
    """Kontras yang menjadi alasan mode token ada.

    Bukan uji atas kode kita melainkan atas premisnya: selama panjang ditakar
    dengan karakter, tidak ada nilai chunk_size yang menjamin muat di jendela
    model. Bila suatu saat premis ini tidak lagi benar, uji ini gagal dan mode
    token layak ditinjau ulang.
    """
    tok = _penakar_atau_lewati(kb.cfg.embedding_model)
    # 900 karakter adalah nilai produksi sebelum T7.
    potongan = kb.chunk_documents(documents=_docs(), chunk_size=900, chunk_overlap=120)

    assert max(tok(c["text"]) for c in potongan) > BATAS_TOKEN_MINILM
