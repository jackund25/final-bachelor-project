"""Tes pembatas laju embedding terkelola (_EmbeddingsBerlaju).

Free-tier Gemini membatasi 100 permintaan embedding per menit per model.
Mengindeks 2.061 potongan sekaligus melampauinya dan gagal di tengah jalan dengan
429 — meninggalkan indeks SEPARUH TERISI, keadaan yang lebih berbahaya daripada
gagal total karena tampak berhasil. Terverifikasi terjadi pada jalan pertama T10.

Seluruh tes di sini memakai penyedia tiruan dan menambal time.sleep, sehingga
tidak menyentuh jaringan maupun menunggu sungguhan.
"""

import pytest

from src.rag import knowledge_base as kb


class _PenyediaTiruan:
    """Penyedia embedding palsu yang mencatat pemanggilan."""

    def __init__(self, gagal_pada=(), dim=4):
        self.panggilan = []          # daftar ukuran tiap kelompok
        self.n_panggilan = 0
        self.gagal_pada = set(gagal_pada)
        self.dim = dim
        self.model = "models/tiruan"

    def embed_documents(self, texts):
        self.n_panggilan += 1
        if self.n_panggilan in self.gagal_pada:
            raise RuntimeError("429 You exceeded your current quota")
        self.panggilan.append(len(texts))
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text):
        self.n_panggilan += 1
        if self.n_panggilan in self.gagal_pada:
            raise RuntimeError("429 You exceeded your current quota")
        return [0.0] * self.dim


@pytest.fixture(autouse=True)
def tanpa_tidur(monkeypatch):
    """Jeda dicatat, bukan dijalani — supaya tes tidak memakan menit."""
    dicatat = []
    monkeypatch.setattr(kb.time if hasattr(kb, "time") else __import__("time"),
                        "sleep", lambda d: dicatat.append(d))
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda d: dicatat.append(d))
    return dicatat


def test_dokumen_dipecah_menjadi_kelompok_kecil():
    """Batasnya menghitung KONTEN, sehingga satu panggilan besar akan ditolak."""
    dalam = _PenyediaTiruan()
    e = kb._EmbeddingsBerlaju(dalam, per_menit=90, ukuran_kelompok=30)

    hasil = e.embed_documents([f"teks {i}" for i in range(95)])

    assert len(hasil) == 95
    assert dalam.panggilan == [30, 30, 30, 5]
    assert max(dalam.panggilan) <= 30


def test_429_dicoba_ulang_dan_akhirnya_berhasil(tanpa_tidur):
    """Kegagalan kuota bersifat sementara; menyerah seketika membuang pekerjaan."""
    dalam = _PenyediaTiruan(gagal_pada={1})
    e = kb._EmbeddingsBerlaju(dalam, per_menit=90, ukuran_kelompok=10)

    hasil = e.embed_documents([f"teks {i}" for i in range(10)])

    assert len(hasil) == 10
    assert dalam.n_panggilan == 2          # sekali gagal, sekali berhasil
    assert any(d >= 30 for d in tanpa_tidur)  # mundur bertahap benar-benar dijeda


def test_galat_selain_kuota_TIDAK_dicoba_ulang():
    """Kunci API salah tidak akan membaik dengan menunggu — gagal cepat lebih baik."""
    class _Rusak(_PenyediaTiruan):
        def embed_documents(self, texts):
            self.n_panggilan += 1
            raise RuntimeError("401 API key tidak sah")

    dalam = _Rusak()
    e = kb._EmbeddingsBerlaju(dalam, per_menit=90, ukuran_kelompok=10)

    with pytest.raises(RuntimeError, match="401"):
        e.embed_documents(["a", "b"])
    assert dalam.n_panggilan == 1


def test_menyerah_setelah_batas_percobaan():
    """Kuota harian yang habis tidak boleh membuat proses menggantung selamanya."""
    dalam = _PenyediaTiruan(gagal_pada=set(range(1, 99)))
    e = kb._EmbeddingsBerlaju(dalam, per_menit=90, ukuran_kelompok=10, maks_coba=3)

    with pytest.raises(RuntimeError, match="429"):
        e.embed_documents(["a"])
    assert dalam.n_panggilan == 3


def test_atribut_penyedia_diteruskan():
    """Pemanggil memeriksa .model untuk mencatat model yang BENAR-BENAR dijalankan."""
    dalam = _PenyediaTiruan()
    e = kb._EmbeddingsBerlaju(dalam)

    assert e.model == "models/tiruan"


def test_embed_query_juga_dibatasi():
    """Evaluasi menembakkan ratusan kueri berturut-turut dan ikut memakan kuota."""
    dalam = _PenyediaTiruan()
    e = kb._EmbeddingsBerlaju(dalam, per_menit=60, ukuran_kelompok=30)

    v = e.embed_query("kadar glukosa 55 mg/dL")

    assert len(v) == 4
    assert dalam.n_panggilan == 1
