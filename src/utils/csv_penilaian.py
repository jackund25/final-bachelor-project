"""Pembaca berkas penilaian manual (T2.2). Murni pustaka standar.

Dipisahkan dari ``scripts/verifikasi_relevansi.py`` karena skrip itu mengimpor
``torch`` di tingkat modul, sehingga tesnya tidak dapat mengimpornya setelah
pytest memuat numpy lewat berkas tes lain.
"""

from __future__ import annotations

from typing import Iterable, Iterator

# Karakter yang mungkin mendahului kata `id` pada baris header: BOM, tanda kutip yang
# ditambahkan csv.writer, dan spasi.
_AWALAN = "﻿\"' "


def mulai_dari_header(baris_berkas: Iterable[str]) -> Iterator[str]:
    """Buang blok komentar dengan mencari baris header, bukan menebak awalannya.

    Penyaring ``not x.startswith("#")`` tidak cukup: baris komentar yang memuat
    koma akan dikutip ``csv.writer``, lolos penyaring, lalu dijadikan header oleh
    ``csv.DictReader``. Mencari ``id,`` bersifat deterministik dan tidak peduli
    bagaimana blok komentar diformat ulang.

    Bila header tidak pernah ditemukan, generator tidak menghasilkan apa pun;
    pemanggil wajib memeriksa ``fieldnames``.
    """
    mulai = False
    for x in baris_berkas:
        if not mulai:
            if x.lstrip(_AWALAN).startswith("id,"):
                mulai = True
            else:
                continue
        yield x
