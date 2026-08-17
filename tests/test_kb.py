"""Uji pemecahan dokumen dan pemuatan potongan cadangan.

DIUBAH 17 Agustus 2026. Berkas ini semula menguji ``create_manual_kb``, yakni jalur yang
menulis potongan berisi prosa susunan sendiri tanpa nomor halaman sumber. Jalur itu dicabut
karena keberadaannya di dalam indeks membuat klaim keterlacakan sitasi tidak benar.

Tesnya TIDAK dihapus, melainkan dialihkan ke perilaku yang menggantikannya, sehingga
cakupan atas pemecahan dokumen dan pemuatan cadangan tetap terjaga.
"""
import json

from src.rag.knowledge_base import MedicalKnowledgeBase


def _dokumen_pedoman():
    """Dua dokumen yang menyerupai keluaran ekstraksi halaman pedoman.

    Metadata sitasi disertakan lengkap, sebab justru itulah yang membedakannya dari
    jalur manual yang dicabut.
    """
    return [
        {
            "text": (
                "Hipoglikemia ditegakkan bila kadar glukosa darah berada di bawah "
                "70 mg/dL. Penanganan awal mengikuti aturan 15-15, yaitu pemberian "
                "15 gram karbohidrat kerja cepat lalu pemeriksaan ulang setelah "
                "15 menit. Bila kadar masih di bawah 70 mg/dL, langkah tersebut diulang."
            ),
            "source": "KB-09_ISPAD-2022_Ch12-Hipoglikemia.pdf",
            "topic": "hipoglikemia",
            "metadata": {
                "doc_id": "KB-09", "chunk_id": "KB-09_ch_001",
                "domain": "pedoman", "subdomain": "hipoglikemia",
                "sumber": "ISPAD 2022", "halaman": 3, "halaman_cetak": 155,
            },
        },
        {
            "text": (
                "Aktivitas fisik teratur menurunkan kebutuhan insulin dan memperbaiki "
                "sensitivitas insulin. Penyesuaian dosis sebelum latihan diperlukan agar "
                "tidak terjadi hipoglikemia selama maupun sesudah aktivitas berlangsung."
            ),
            "source": "KB-11_ISPAD-2022_Ch14-Aktivitas-Fisik.pdf",
            "topic": "aktivitas fisik",
            "metadata": {
                "doc_id": "KB-11", "chunk_id": "KB-11_ch_001",
                "domain": "pedoman", "subdomain": "aktivitas_fisik",
                "sumber": "ISPAD 2022", "halaman": 5, "halaman_cetak": 210,
            },
        },
    ]


def test_chunk_documents_menghasilkan_potongan_bermetadata(tmp_path):
    kb = MedicalKnowledgeBase(kb_dir=str(tmp_path))
    chunks = kb.chunk_documents(documents=_dokumen_pedoman())

    assert chunks
    assert all(isinstance(c.get("text"), str) and c["text"].strip() for c in chunks)
    # Tiap potongan wajib membawa sumbernya; inilah dasar klaim keterlacakan sitasi.
    assert all(c.get("source") for c in chunks)


def test_muat_potongan_cadangan_membaca_berkas_ekspor(tmp_path):
    cadangan = _dokumen_pedoman()
    (tmp_path / "fallback_chunks.json").write_text(
        json.dumps(cadangan, ensure_ascii=False), encoding="utf-8")

    kb = MedicalKnowledgeBase(kb_dir=str(tmp_path))
    docs = kb.muat_potongan_cadangan()

    assert len(docs) == 2
    # Cadangan pun harus tertelusur sampai halaman cetaknya, bukan sekadar ada.
    assert all(d["metadata"].get("halaman_cetak") for d in docs)


def test_muat_potongan_cadangan_mengembalikan_kosong_bila_berkas_tak_ada(tmp_path):
    """Ketiadaan berkas bukan galat fatal; pemanggil yang memutuskan artinya."""
    kb = MedicalKnowledgeBase(kb_dir=str(tmp_path))

    assert kb.muat_potongan_cadangan() == []
