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

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

PAGE_UNKNOWN = "Hal. tidak tercatat"

# Judul lengkap artikel jurnal memuat ekor sitasi yang panjang
# (mis. KB-05: "... Diabetologia 64(12):2609-2652"), dipotong agar UI terbaca.
_TITLE_MAX_CHARS = 90

# Panjang kutipan yang ditampilkan pada daftar Sumber Rujukan.
#
# Dinaikkan dari 200 ke 700 karakter (kira-kira 4-5 baris pada lebar UI). Alasannya
# bukan estetika: kutipan sepanjang satu baris tidak cukup untuk MENCOCOKKAN hasil
# penelusuran dengan halaman dokumen aslinya, padahal justru kemampuan itu yang
# menjadi bukti KNF-08 (keterlacakan sitasi) saat sidang.
#
# Batas atasnya sendiri adalah rag.chunk_size (900), sehingga sebagian besar potongan
# kini tampil hampir utuh. Sisanya tetap dapat dibuka lewat teks_lengkap.
SNIPPET_CHARS_DEFAULT = 700


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
    """Potong pada BATAS KATA terdekat, bukan di tengah kata.

    Kutipan pada UI dipakai untuk mencocokkan hasil penelusuran dengan dokumen
    aslinya. Potongan di tengah kata ("hipoglikem") menyulitkan pencarian teks pada
    PDF sumber, sehingga batas kata bukan sekadar soal kerapian.
    """
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    potong = text[:limit]
    spasi = potong.rfind(" ")
    # Hanya mundur ke batas kata bila tidak memangkas terlalu banyak; kata tunggal
    # yang sangat panjang (mis. URL) tetap dipotong keras daripada hilang seluruhnya.
    if spasi > limit * 0.6:
        potong = potong[:spasi]
    return potong.rstrip(" ,;:.") + "…"


# --------------------------------------------------------------------------
# Pemotongan pada BATAS KALIMAT (Masalah B)
#
# WHAT  : penentu batas kalimat untuk prosa klinis berbahasa Indonesia.
# WHO   : dipakai potong_batas_kalimat() di modul ini; konsumennya daftar
#         "Sumber Rujukan" pada Streamlit dan log keputusan dokter.
# WHERE : src/rag/citations.py, hulu dari build_source_list().
# WHEN  : setiap kali satu baris rujukan dibentuk, yaitu tiap kali dokter
#         menjalankan satu konsultasi.
# WHY   : Gao dkk. (2023) §V.A.1 hal. 8 menyebut kelemahan pemotongan
#         berukuran tetap sebagai "truncation within sentences" — potongan
#         yang berhenti di tengah kalimat. Batas KATA (_truncate di atas)
#         tidak menyelesaikannya: ia hanya menjamin kata terakhir utuh,
#         bukan kalimatnya. Manning dkk. (2009) hal. 217 menegaskan bahwa
#         satuan yang dikembalikan kepada pembaca dalam passage retrieval
#         adalah "passage" dengan batas yang bermakna, bukan potongan
#         sembarang.
# HOW   : kandidat batas dicari dengan regex, lalu DISARING oleh tiga
#         penjaga di bawah supaya titik yang bukan akhir kalimat tidak
#         dianggap batas. Tanpa penyaringan itu "PERKENI 2021 hal. 12"
#         akan terpotong menjadi "PERKENI 2021 hal." — lebih buruk
#         daripada batas kata.
# --------------------------------------------------------------------------

# Penjaga 1 — kata yang berakhir titik tetapi BUKAN akhir kalimat.
# Dikumpulkan dari prosa pedoman PERKENI/IDAI/ADA yang menjadi korpus.
_SINGKATAN_BUKAN_AKHIR = {
    # gelar dan sapaan
    "dr", "drg", "prof", "sp", "ns", "yth",
    # rujukan dan penomoran naskah
    "dkk", "dll", "dsb", "tsb", "hal", "no", "nomor", "tab", "gbr", "gambar",
    "bab", "ed", "vol", "cet", "jl", "et", "al", "fig", "pp", "vs",
    # satuan yang kerap ditulis bertitik
    "mg", "ml", "dl", "kg", "mmol", "mm", "cm", "jam", "thn", "min", "maks",
}

# Penjaga 2 — huruf tunggal berakhir titik hampir selalu inisial nama
# ("Reimers, N.") atau penanda enumerasi ("a."), bukan akhir kalimat.
_MAKS_HURUF_INISIAL = 1

# Penjaga 3 — angka pendek berakhir titik adalah penanda daftar bernomor
# ("1. Terapi insulin"), bukan akhir kalimat.
_MAKS_DIGIT_ENUMERASI = 2

# Kandidat batas: tanda akhir kalimat, boleh diikuti kutip/kurung penutup,
# lalu WAJIB diikuti spasi. Lookahead spasi ini sekaligus menyelamatkan
# bilangan desimal ("7.5", "10.000") tanpa penjaga tambahan.
_POLA_KANDIDAT = re.compile(r'[.!?]["\'\)\]]?(?=\s)')

# Setelah batas kalimat, huruf berikutnya lazim kapital atau angka. Bila
# huruf kecil, hampir pasti titiknya milik singkatan yang lolos penjaga 1.
_POLA_LANJUTAN_SAH = re.compile(r'["\'\(\[]?[A-Z0-9•\-–]')

# Ambang bawah: bila batas kalimat terdekat memangkas lebih dari 40% jatah
# tampilan, kutipannya jadi terlalu pendek untuk dicocokkan ke PDF sumber —
# padahal itulah gunanya (bukti KNF-08). Pada kasus itu jatuh kembali ke
# batas kata.
_RASIO_MIN_BATAS_KALIMAT = 0.6


def _kandidat_batas_kalimat(text: str) -> List[int]:
    """Indeks tepat SESUDAH tiap tanda akhir kalimat yang lolos penjaga."""
    batas: List[int] = []
    for m in _POLA_KANDIDAT.finditer(text):
        akhir = m.end()

        # Penjaga 1-3: periksa token tepat sebelum tanda baca.
        sebelum = re.search(r'([A-Za-z0-9]+)$', text[: m.start()])
        if sebelum:
            token = sebelum.group(1)
            if token.isdigit():
                if len(token) <= _MAKS_DIGIT_ENUMERASI:
                    continue
            elif token.lower() in _SINGKATAN_BUKAN_AKHIR:
                continue
            elif len(token) <= _MAKS_HURUF_INISIAL:
                continue

        # Penjaga 4: apa yang menyusul harus tampak seperti awal kalimat.
        lanjutan = text[akhir:].lstrip()
        if lanjutan and not _POLA_LANJUTAN_SAH.match(lanjutan):
            continue

        batas.append(akhir)
    return batas


def berakhir_di_batas_kalimat(text: str) -> bool:
    """True bila teks berhenti pada tanda akhir kalimat.

    Dipakai untuk menandai potongan yang SUDAH terpotong sejak proses
    indexing, bukan oleh tampilan — dua sebab yang tidak boleh tertukar
    saat mendiagnosis keluhan "teksnya terpotong".
    """
    ekor = (text or "").rstrip()
    return bool(ekor) and ekor[-1] in '.!?"\')]' and bool(
        re.search(r'[.!?]["\'\)\]]?$', ekor)
    )


def bermula_di_batas_kalimat(text: str) -> bool:
    """True bila teks tampak dimulai pada awal kalimat, bukan di tengahnya.

    Awalan huruf kecil menandakan potongan mewarisi separuh kalimat dari
    potongan sebelumnya — inilah "konteks yang tertinggal" yang ditanyakan
    pengguna. Ditandai, bukan diperbaiki di sini: perbaikannya ada di sisi
    indexing (src/rag/knowledge_base.py), sedangkan modul ini hanya
    melaporkan apa adanya.
    """
    awal = (text or "").lstrip()
    if not awal:
        return True
    return bool(_POLA_LANJUTAN_SAH.match(awal))


def potong_batas_kalimat(text: str, limit: int) -> Dict[str, Any]:
    """Potong teks pada batas kalimat terakhir yang masih muat dalam ``limit``.

    Mengembalikan dict, bukan string, karena pemanggil perlu tahu BUKAN HANYA
    hasil potongnya melainkan juga apakah pemotongan terjadi dan dengan cara
    apa. Tanpa itu, "potongan memang sependek itu" tidak dapat dibedakan dari
    "tampilannya yang memotong" — pembedaan yang diperlukan saat sidang.
    """
    teks = " ".join((text or "").split())

    if len(teks) <= limit:
        return {
            "snippet": teks,
            "dipotong": False,
            "cara_potong": "utuh",
            "n_kalimat": len(_kandidat_batas_kalimat(teks + " ")),
        }

    # Kandidat dicari pada teks + spasi supaya kalimat terakhir yang tepat
    # berakhir di posisi limit tetap terlihat oleh lookahead.
    batas = [b for b in _kandidat_batas_kalimat(teks + " ") if b <= limit]

    if batas and batas[-1] >= limit * _RASIO_MIN_BATAS_KALIMAT:
        return {
            "snippet": teks[: batas[-1]].rstrip(),
            "dipotong": True,
            "cara_potong": "batas_kalimat",
            "n_kalimat": len(batas),
        }

    # Jatuh kembali ke batas kata: lebih baik kutipan panjang yang berhenti
    # di tengah kalimat daripada kutipan yang terlalu pendek untuk dicocokkan.
    return {
        "snippet": _truncate(teks, limit),
        "dipotong": True,
        "cara_potong": "batas_kata",
        "n_kalimat": len(batas),
    }


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
    snippet_chars: int = SNIPPET_CHARS_DEFAULT,
) -> List[Dict[str, Any]]:
    """Ratakan retrieved_docs pipeline menjadi baris siap tampil / siap audit.

    Nilai kembalian sengaja berupa dict datar berisi nilai yang SUDAH diresolusi
    (termasuk page_label), supaya baris yang sama dapat disimpan ke log keputusan
    dan merekam persis apa yang dilihat dokter saat mengambil keputusan.

    Selain ``snippet`` yang dipotong untuk tampilan, setiap baris membawa
    ``teks_lengkap`` berisi potongan dokumen UTUH sebagaimana dikirim ke LLM.
    Keduanya diperlukan: yang pertama agar UI terbaca, yang kedua agar penelusuran
    ke dokumen sumber dapat diverifikasi tanpa menebak bagian yang terpotong.
    """
    rows: List[Dict[str, Any]] = []

    for idx, doc in enumerate(retrieved_docs or [], start=1):
        meta = dict(doc.get("metadata") or {})
        fallback_name = str(doc.get("source") or "")
        similarity = doc.get("similarity")

        teks = " ".join(str(doc.get("text") or "").split())
        # Potongan tampilan berhenti di batas KALIMAT (Gao dkk. 2023 §V.A.1).
        potong = potong_batas_kalimat(teks, snippet_chars)
        snippet = potong["snippet"]

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
                "snippet": snippet,
                # Potongan utuh + penanda apakah tampilan memangkasnya. Tanpa penanda
                # ini pembaca tidak dapat membedakan "potongan memang sependek itu"
                # dari "tampilannya yang dipotong".
                "teks_lengkap": teks,
                "n_char": len(teks),
                "snippet_terpotong": potong["dipotong"],
                # Cara tampilan memotong: "utuh" | "batas_kalimat" | "batas_kata".
                "cara_potong": potong["cara_potong"],
                # Dua penanda di bawah menjawab pertanyaan "apakah ada konteks
                # yang tertinggal": keduanya menggambarkan POTONGAN ASLI hasil
                # indexing, bukan hasil pemotongan tampilan. Bila salah satunya
                # False, kalimat di tepi potongan memang mewarisi/menyisakan
                # separuh kalimat ke potongan tetangga.
                "mulai_kalimat_utuh": bermula_di_batas_kalimat(teks),
                "akhir_kalimat_utuh": berakhir_di_batas_kalimat(teks),
                "chunk_id": meta.get("chunk_id", ""),
            }
        )

    return rows
