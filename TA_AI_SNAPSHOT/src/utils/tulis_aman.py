"""Penulisan berkas hasil yang tidak menimpa diam-diam.

LATAR
-----
`run_ragas.py` menimpa `results/ragas/summary.json` dan `per_sample.csv` dengan angka jalan
kedua, dan bahkan menghilangkan medan `faithfulness`. Itu tertangkap **hanya karena**
kebetulan ada perbandingan dua jalan yang sedang berlangsung. Audit 13 Agustus 2026
menemukan **33 dari 41** skrip penulis di `scripts/` menimpa berkas hasil tanpa penjagaan
apa pun.

Merombak ketiga puluh tiga skrip itu di luar lingkup. Modul ini menyediakan penggantinya
yang murah, dipakai untuk skrip baru dan untuk skrip lama bila kebetulan disentuh.

Penjagaan sesungguhnya bagi berkas yang sudah ada dipasang di `buat_ringkasan_bab6.py`:
ia memeriksa status git tiap sumber dan menandai yang termodifikasi-tetapi-belum-dicommit.
Itu satu perubahan yang melindungi keduapuluh empat sumber sekaligus.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


def arsipkan(path: Path, arsip_dir: Path | None = None) -> Path | None:
    """Pindahkan berkas yang sudah ada ke nama bersufiks waktu. Kembalikan tujuannya.

    Mengembalikan ``None`` bila berkasnya memang belum ada.
    """
    path = Path(path)
    if not path.exists():
        return None
    arsip_dir = Path(arsip_dir) if arsip_dir else path.parent / "_arsip"
    arsip_dir.mkdir(parents=True, exist_ok=True)
    cap = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%dT%H%M%S")
    tujuan = arsip_dir / f"{path.stem}.{cap}{path.suffix}"
    n = 1
    while tujuan.exists():
        tujuan = arsip_dir / f"{path.stem}.{cap}_{n}{path.suffix}"
        n += 1
    shutil.copy2(path, tujuan)
    return tujuan


def tulis_json_aman(path, data, *, indent: int = 2, arsip: bool = True) -> Path | None:
    """Tulis JSON, setelah menyalin versi lama ke arsip bersufiks waktu.

    Mengembalikan path arsip bila ada versi lama, ``None`` bila berkasnya baru.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lama = arsipkan(path) if arsip else None
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    return lama
