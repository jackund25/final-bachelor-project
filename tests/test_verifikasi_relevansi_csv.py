"""Regresi untuk pembacaan berkas penilaian T2.2.

Cacat yang dijaga tes ini ditemukan 11 Agustus 2026, dan ia jenis cacat yang paling mahal:
ia hanya muncul **setelah** pekerjaan manual selesai.

`scripts/verifikasi_relevansi.py --buat` menulis blok komentar di kepala berkas. Salah satu
baris komentar memuat koma, sehingga `csv.writer` mengutipnya dan barisnya menjadi
`"# 'lain' dipakai bila ...`. Penyaring semula, `not x.startswith("#")`, **tidak menangkap
baris berkutip itu**, sehingga `csv.DictReader` menjadikannya HEADER. Seluruh baris data
kehilangan kolom `id`, dan `--nilai` berhenti dengan "Belum ada satu pun penilaian yang
terisi" **meskipun keempat puluh baris sudah diisi**.

Menyimpan-ulang berkas dengan Excel juga menambahkan koma ekor pada baris komentar, sehingga
penyaring apa pun yang bertumpu pada BENTUK komentar bersifat rapuh. Penggantinya mencari
baris header `id,` secara deterministik.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.utils.csv_penilaian import mulai_dari_header

KOLOM = ["id", "penilaian_manusia", "kueri", "sumber", "halaman_cetak", "peringkat",
         "teks_potongan"]


def _tulis(tmp_path, baris_komentar, n_data=3):
    p = tmp_path / "verifikasi.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for k in baris_komentar:
            w.writerow([k])
        w.writerow([])
        w.writerow(KOLOM)
        for i in range(1, n_data + 1):
            w.writerow([i, "", f"kueri {i}", "KB-01.pdf", 10, 1, f"teks {i}"])
    return p


def _baca(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(mulai_dari_header(f))
        return r.fieldnames, [x for x in r if x.get("id")]


def test_komentar_berkoma_yang_dikutip_tidak_menjadi_header(tmp_path):
    """Inilah cacat aslinya: komentar memuat koma sehingga csv.writer mengutipnya."""
    p = _tulis(tmp_path, [
        "# T2.2 - verifikasi relevansi.",
        "# 'lain' dipakai bila potongan tidak membahas ketiga kondisi itu "
        "(mis. definisi diabetes, alat suntik, gizi umum).",
    ])
    fieldnames, baris = _baca(p)

    assert fieldnames == KOLOM
    assert len(baris) == 3
    assert baris[0]["id"] == "1"


def test_koma_ekor_gaya_excel_tetap_terbaca(tmp_path):
    """Excel menyimpan-ulang berkas dengan koma ekor pada tiap baris komentar."""
    p = tmp_path / "excel.csv"
    isi = [
        "# T2.2 - verifikasi relevansi.,,,,,,",
        "\"# 'lain' dipakai bila potongan tidak membahas kondisi itu (mis. gizi umum).\",,,,,,",
        "# Label otomatis SENGAJA tidak ditampilkan.,,,,,,",
        ",,,,,,",
        ",".join(KOLOM),
        "1,,kueri 1,KB-01.pdf,10,1,teks 1",
        "2,hipoglikemia,kueri 2,KB-02.pdf,25,4,teks 2",
    ]
    p.write_text("﻿" + "\n".join(isi) + "\n", encoding="utf-8")

    fieldnames, baris = _baca(p)

    assert fieldnames == KOLOM
    assert len(baris) == 2
    assert baris[1]["penilaian_manusia"] == "hipoglikemia"


def test_berkas_tanpa_komentar_tetap_terbaca(tmp_path):
    p = _tulis(tmp_path, [])
    fieldnames, baris = _baca(p)

    assert fieldnames == KOLOM
    assert len(baris) == 3


def test_berkas_nyata_memuat_empat_puluh_baris(tmp_path):
    """Berkas T2.2 yang sesungguhnya harus terbaca utuh, apa pun tata letaknya."""
    nyata = ROOT / "evaluation/verifikasi_relevansi.csv"
    if not nyata.exists():
        return

    fieldnames, baris = _baca(nyata)

    assert fieldnames == KOLOM
    assert len(baris) == 40
    assert [b["id"] for b in baris] == [str(i) for i in range(1, 41)]
