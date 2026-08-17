"""Pembentukan sitasi dari metadata potongan — satu sumber kebenaran.

Nomor halaman yang ditampilkan kepada dokter selalu berasal dari metadata potongan,
tidak pernah dari teks yang dihasilkan model bahasa. Dipusatkan di sini karena tiga
konsumen memakainya: panel Sumber Rujukan, log keputusan, dan advisor_chain.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

PAGE_UNKNOWN = "Hal. tidak tercatat"

# Judul lengkap artikel jurnal memuat ekor sitasi yang panjang
# (mis. KB-05: "... Diabetologia 64(12):2609-2652"), dipotong agar UI terbaca.
_TITLE_MAX_CHARS = 90

# Panjang kutipan pada daftar Sumber Rujukan. Cukup panjang agar potongan dapat
# dicocokkan ke halaman dokumen aslinya, yang menjadi bukti keterlacakan sitasi.
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
    """Potong pada batas kata terdekat.

    Potongan di tengah kata menyulitkan pencarian teks pada PDF sumber, sehingga
    ini bukan sekadar soal kerapian.
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


# Pemotongan pada batas kalimat. Batas kata saja tidak cukup: ia menjamin kata
# terakhir utuh, bukan kalimatnya (Gao dkk. 2023). Kandidat batas disaring empat
# penjaga di bawah supaya "PERKENI 2021 hal. 12" tidak terpotong jadi "... hal.".

# Penjaga 1 — kata berakhir titik yang bukan akhir kalimat, dikumpulkan dari korpus.
_SINGKATAN_BUKAN_AKHIR = {
    # gelar dan sapaan
    "dr", "drg", "prof", "sp", "ns", "yth",
    # rujukan dan penomoran naskah
    "dkk", "dll", "dsb", "tsb", "hal", "no", "nomor", "tab", "gbr", "gambar",
    "bab", "ed", "vol", "cet", "jl", "et", "al", "fig", "pp", "vs",
    # satuan yang kerap ditulis bertitik
    "mg", "ml", "dl", "kg", "mmol", "mm", "cm", "jam", "thn", "min", "maks",
}

# Penjaga 2 — huruf tunggal berakhir titik: inisial nama atau penanda enumerasi.
_MAKS_HURUF_INISIAL = 1

# Penjaga 3 — angka pendek berakhir titik: penanda daftar bernomor.
_MAKS_DIGIT_ENUMERASI = 2

# Wajib diikuti spasi; lookahead ini sekaligus menyelamatkan bilangan desimal.
_POLA_KANDIDAT = re.compile(r'[.!?]["\'\)\]]?(?=\s)')

# Penjaga 4 — yang menyusul harus tampak seperti awal kalimat.
_POLA_LANJUTAN_SAH = re.compile(r'["\'\(\[]?[A-Z0-9•\-–]')

# Bila batas kalimat memangkas lebih dari 40% jatah tampilan, kutipannya terlalu
# pendek untuk dicocokkan ke PDF sumber; pada kasus itu jatuh ke batas kata.
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

    Menandai potongan yang sudah terpotong sejak pengindeksan, bukan oleh tampilan.
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
    """Ratakan hasil penelusuran menjadi baris siap tampil dan siap audit.

    Nilainya sudah diresolusi (termasuk ``page_label``) supaya baris yang sama dapat
    disimpan ke log keputusan dan merekam persis apa yang dilihat dokter. Tiap baris
    membawa ``snippet`` untuk tampilan dan ``teks_lengkap`` untuk verifikasi.
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
