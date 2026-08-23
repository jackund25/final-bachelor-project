"""Pembaca berkas penilaian manual (T2.2). Murni pustaka standar — tanpa torch/numpy.

Modul ini dipisahkan dari `scripts/verifikasi_relevansi.py` karena skrip itu mengimpor
`torch` di tingkat modul (wajib, agar mendahului numpy pada Windows — lihat WinError 1114),
sehingga tesnya tidak dapat mengimpornya setelah pytest memuat numpy lewat berkas tes lain.
Fungsi di sini karena itu menjadi SATU sumber kebenaran yang dapat dipakai keduanya.
"""

from __future__ import annotations

from typing import Iterable, Iterator

# Karakter yang mungkin mendahului kata `id` pada baris header: BOM, tanda kutip yang
# ditambahkan csv.writer, dan spasi.
_AWALAN = "﻿\"' "


def mulai_dari_header(baris_berkas: Iterable[str]) -> Iterator[str]:
    """Buang blok komentar dengan MENCARI baris header, bukan menebak awalan komentar.

    Penyaring semula ``not x.startswith("#")`` tidak cukup, dan ini bukan kehati-hatian
    berlebihan. Salah satu baris komentar memuat koma, sehingga ``csv.writer`` mengutipnya
    dan barisnya menjadi ``"# 'lain' dipakai bila ...``. Baris itu **lolos** penyaring, lalu
    ``csv.DictReader`` menjadikannya HEADER. Seluruh baris data kehilangan kolom ``id`` dan
    penilaian yang sudah diisi manusia terbaca sebagai kosong.

    Cacat itu ada sejak berkas dibuat dan hanya muncul **setelah** pekerjaan manual selesai,
    yaitu waktu paling mahal untuk menemukannya.

    Menyimpan-ulang berkas dengan Excel juga menambahkan koma ekor pada baris komentar,
    sehingga penyaring apa pun yang bertumpu pada BENTUK komentar bersifat rapuh. Mencari
    ``id,`` bersifat deterministik dan tidak peduli bagaimana blok komentar diformat ulang.

    Bila baris header tidak pernah ditemukan, generator ini tidak menghasilkan apa pun —
    pemanggil wajib memeriksa ``fieldnames`` dan berhenti dengan pesan yang jelas.
    """
    mulai = False
    for x in baris_berkas:
        if not mulai:
            if x.lstrip(_AWALAN).startswith("id,"):
                mulai = True
            else:
                continue
        yield x
