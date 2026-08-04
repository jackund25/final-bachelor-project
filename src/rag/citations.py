"""Pembentukan sitasi dari metadata chunk — satu sumber kebenaran.

PRINSIP: nomor halaman yang ditampilkan kepada dokter HARUS berasal dari metadata
chunk, tidak pernah dari teks yang dihasilkan LLM. Model bahasa tidak boleh menjadi
sumber nomor halaman karena nilainya tidak dapat dijamin.

Modul ini dipakai tiga konsumen — expander "Sumber Rujukan" di Streamlit, log
keputusan dokter, dan ringkasan sumber di advisor_chain. Menaruh aturan halaman
di masing-masing konsumen menjamin ketiganya menyimpang seiring waktu.

Skema metadata halaman dihasilkan oleh scripts/reingest_kb.py:
    kb_id, source_id, nama_dokumen, lembaga, tahun, judul_lengkap,
    halaman_pdf (int), halaman_cetak (int; 0 = tidak valid), halaman_cetak_valid (bool)

Chunk manual_kb.json tidak punya medan tersebut dan ditangani lewat fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

PAGE_UNKNOWN = "Hal. tidak tercatat"

# Judul lengkap artikel jurnal memuat ekor sitasi yang panjang
# (mis. KB-05: "... Diabetologia 64(12):2609-2652"), dipotong agar UI terbaca.
_TITLE_MAX_CHARS = 90


def format_page_label(meta: Mapping[str, Any]) -> str:
    """Label halaman yang dapat dibaca dokter, murni dari metadata.

    Tiga kasus:
    - Halaman bernomor cetak      -> "Hal. 1322"
    - Front matter (offset < 1)   -> "Hal. berkas 3 (bagian depan, tanpa nomor cetak)"
    - Tanpa info halaman sama sekali (mis. manual_kb) -> "Hal. tidak tercatat"
    """
    if not meta:
        return PAGE_UNKNOWN

    cetak = meta.get("halaman_cetak")
    valid = meta.get("halaman_cetak_valid")
    pdf = meta.get("halaman_pdf")

    if valid and isinstance(cetak, int) and cetak >= 1:
        return f"Hal. {cetak}"

    if isinstance(pdf, int) and pdf >= 1:
        # Halaman ada, tapi tidak punya nomor cetak. Tampilkan nomor halaman
        # BERKAS disertai keterangan supaya dokter tahu ini bukan nomor cetak.
        return f"Hal. berkas {pdf} (bagian depan, tanpa nomor cetak)"

    # Skema lama (manual_kb) menyimpan "halaman" sebagai string, kerap "N/A".
    legacy = meta.get("halaman")
    if isinstance(legacy, str) and legacy.strip() and legacy.strip().upper() != "N/A":
        return f"Hal. {legacy.strip()}"
    if isinstance(legacy, int) and legacy >= 1:
        return f"Hal. {legacy}"

    return PAGE_UNKNOWN


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def document_title(meta: Mapping[str, Any], fallback: str = "") -> str:
    """Judul tampilan dokumen, dipotong agar muat di UI."""
    judul = (
        meta.get("judul_lengkap")
        or meta.get("judul")
        or meta.get("topic")
        or fallback
        or "Dokumen tanpa judul"
    )
    return _truncate(str(judul), _TITLE_MAX_CHARS)


def build_source_list(
    retrieved_docs: Sequence[Mapping[str, Any]],
    snippet_chars: int = 200,
) -> List[Dict[str, Any]]:
    """Ratakan retrieved_docs pipeline menjadi baris siap tampil / siap audit.

    Nilai kembalian sengaja berupa dict datar berisi nilai yang SUDAH diresolusi
    (termasuk page_label), supaya baris yang sama dapat disimpan ke log keputusan
    dan merekam persis apa yang dilihat dokter saat mengambil keputusan.
    """
    rows: List[Dict[str, Any]] = []

    for idx, doc in enumerate(retrieved_docs or [], start=1):
        meta = dict(doc.get("metadata") or {})
        fallback_name = str(doc.get("source") or "")
        similarity = doc.get("similarity")

        rows.append(
            {
                "rank": doc.get("rank", idx),
                "kb_id": meta.get("kb_id", ""),
                "source_id": meta.get("source_id", ""),
                "nama_dokumen": meta.get("nama_dokumen") or fallback_name,
                "judul_lengkap": document_title(meta, fallback_name),
                "lembaga": meta.get("lembaga") or meta.get("sumber") or "",
                "tahun": meta.get("tahun", ""),
                "halaman_pdf": meta.get("halaman_pdf"),
                "halaman_cetak": meta.get("halaman_cetak"),
                "page_label": format_page_label(meta),
                "similarity": (
                    float(similarity) if isinstance(similarity, (int, float)) else None
                ),
                "snippet": _truncate(str(doc.get("text") or ""), snippet_chars),
                "chunk_id": meta.get("chunk_id", ""),
            }
        )

    return rows
