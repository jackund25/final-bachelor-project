#!/usr/bin/env python3
"""Statistik evaluasi ahli putaran n=12 dari CSV tanggapan per responden.

Sumber tunggal: ekspor spreadsheet tanggapan Google Forms pada
``results/hasil_form/terbaru/``. Berbeda dengan putaran n=7 dan n=10 yang hanya
memiliki ekspor PDF ringkasan analytics (distribusi marginal per pertanyaan),
ekspor ini memuat SATU BARIS PER RESPONDEN. Konsekuensinya:

  * skor SUS per responden dapat dihitung dengan algoritma Brooke yang baku,
    sehingga simpangan baku, median, rentang, dan selang kepercayaan rerata
    SUS antar-responden kini teridentifikasi. Pada putaran sebelumnya ketiganya
    sengaja tidak dilaporkan karena tidak dapat direkonstruksi dari marginal;
  * pemasangan nilai antar-butir untuk responden yang sama terekam, sehingga
    analisis yang menuntut baris responden menjadi mungkin.

Butir SUS memakai skala 1--5 sesuai instrumen aslinya, sedangkan butir skenario
dan penilaian menyeluruh memakai skala 1--4 mengikuti rancangan formulir.

Identitas responden (nama, surel) tidak pernah dibaca ke keluaran mana pun.

Jalankan:
    conda run -n diabetes-ta python scripts/analisis_hasil_form_csv.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMBER_DIR = ROOT / "results/hasil_form/terbaru"
CSV_NAMA = ("Form Penggunaan dan Evaluasi Sistem Diabetes Clinical Decision "
            "Support  (Jawaban) - Form Responses 3 (1).csv")
PDF_NAMA = "hasil form update 2.pdf"

# Indeks kolom pada CSV ekspor Google Forms.
KOL_PROFESI = 4
KOL_OKUPASI = 5
KOL_CDSS = 6
KOL_CGM = 8
KOL_REKOMENDASI = {"3B": 19, "3C": 24}
KOL_BEBAS = [25, 28, 40, 45]

# Butir skala 1--4: (kolom, label ringkas untuk tabel).
BUTIR_SKALA4 = [
    (9, "Prediksi dan status kondisi pada CGM"),
    (10, "Prediksi finger-stick dan komunikasi keterbatasan"),
    (26, "Penolakan prediksi saat data tidak memadai"),
    (27, "Menghindari kesan kepastian berlebihan"),
    (41, "Menjaga otonomi dokter"),
    (42, "Consent dan keamanan data"),
    (43, "Dasar rekomendasi cukup jelas"),
    (44, "Alur realistis untuk praktik klinis"),
    (39, "Karakteristik populasi sumber data memadai"),
]

# Grid 1--4 per skenario: (skenario, kriteria, kolom).
GRID = [
    ("3A", "Relevansi", 11), ("3A", "Kejelasan", 12),
    ("3A", "Kecukupan konteks", 13), ("3A", "Batasan keputusan", 14),
    ("3B", "Relevansi", 15), ("3B", "Kejelasan", 16),
    ("3B", "Kecukupan konteks", 17), ("3B", "Batasan keputusan", 18),
    ("3C", "Relevansi", 20), ("3C", "Kejelasan", 21),
    ("3C", "Kecukupan konteks", 22), ("3C", "Batasan keputusan", 23),
]

# Sepuluh butir SUS berurutan sesuai instrumen Brooke; ganjil positif, genap negatif.
SUS_KOLOM = list(range(29, 39))
SUS_LABEL = [
    "1. Akan sering menggunakan",
    "2. Rumit tanpa alasan jelas",
    "3. Mudah digunakan",
    "4. Butuh bantuan teknis",
    "5. Fungsi terintegrasi",
    "6. Terlalu banyak inkonsistensi",
    "7. Cepat dipelajari orang lain",
    "8. Merepotkan digunakan",
    "9. Percaya diri saat memakai",
    "10. Perlu banyak belajar dahulu",
]


def baca_csv():
    with open(SUMBER_DIR / CSV_NAMA, encoding="utf-8") as f:
        baris = list(csv.reader(f))
    return baris[0], baris[1:]


def angka(sel):
    sel = (sel or "").strip()
    return int(sel) if sel.isdigit() else None


def ringkas(nilai, batas_baik=3):
    """Statistik satu butir ordinal."""
    n = len(nilai)
    return {
        "n": n,
        "rerata": round(st.mean(nilai), 3),
        "sb": round(st.stdev(nilai), 3) if n > 1 else None,
        "median": st.median(nilai),
        "modus": st.mode(nilai),
        "min": min(nilai),
        "maks": max(nilai),
        "n_min_baik": sum(1 for v in nilai if v >= batas_baik),
        "persen_min_baik": round(100.0 * sum(1 for v in nilai if v >= batas_baik) / n, 1),
    }


def skor_sus_satu(nilai):
    """Algoritma Brooke: ganjil (v-1), genap (5-v), dijumlah lalu dikali 2,5."""
    total = 0
    for i, v in enumerate(nilai):
        total += (v - 1) if i % 2 == 0 else (5 - v)
    return total * 2.5


def main() -> None:
    kepala, data = baca_csv()
    n = len(data)

    hasil = {
        "judul": "Statistik evaluasi ahli putaran n=%d" % n,
        "sumber": {
            "csv": str((SUMBER_DIR / CSV_NAMA).relative_to(ROOT)).replace("\\", "/"),
            "pdf_ringkasan": str((SUMBER_DIR / PDF_NAMA).relative_to(ROOT)).replace("\\", "/"),
            "bentuk": "satu baris per responden",
        },
        "n_responden": n,
        "catatan_skala": {
            "sus": "1-5 sesuai instrumen Brooke",
            "skenario_dan_penilaian": "1-4 sesuai rancangan formulir",
        },
    }

    # --- profil ---
    hasil["profil"] = {}
    for nama, kol in (("profesi", KOL_PROFESI), ("bidang_okupasi", KOL_OKUPASI),
                      ("pengalaman_cdss", KOL_CDSS), ("pengalaman_cgm", KOL_CGM)):
        isi = [r[kol].strip() for r in data if r[kol].strip()]
        hasil["profil"][nama] = {
            "n_menjawab": len(isi),
            "cacah": dict(Counter(isi).most_common()),
        }

    # --- butir skala 1-4 ---
    hasil["butir_skala4"] = {}
    for kol, label in BUTIR_SKALA4:
        nilai = [angka(r[kol]) for r in data]
        nilai = [v for v in nilai if v is not None]
        hasil["butir_skala4"][label] = ringkas(nilai)

    # --- grid per skenario ---
    hasil["grid"] = {}
    for skenario, kriteria, kol in GRID:
        nilai = [angka(r[kol]) for r in data]
        nilai = [v for v in nilai if v is not None]
        hasil["grid"].setdefault(skenario, {})[kriteria] = ringkas(nilai)

    # --- kategori rekomendasi ---
    hasil["rekomendasi"] = {}
    for skenario, kol in KOL_REKOMENDASI.items():
        isi = [r[kol].strip() for r in data if r[kol].strip()]
        hasil["rekomendasi"][skenario] = {
            "n_menjawab": len(isi),
            "cacah": dict(Counter(isi).most_common()),
        }

    # --- SUS per responden ---
    sus_baris, sus_valid = [], []
    for r in data:
        nilai = [angka(r[k]) for k in SUS_KOLOM]
        if all(v is not None for v in nilai):
            sus_valid.append(nilai)
            sus_baris.append(skor_sus_satu(nilai))
    ns = len(sus_baris)
    sb = st.stdev(sus_baris) if ns > 1 else None
    galat = (sb / math.sqrt(ns)) if sb else None
    # t kritis dua sisi 95% untuk derajat bebas kecil
    T95 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
           11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}
    t = T95.get(ns - 1, 1.96)
    hasil["sus"] = {
        "n": ns,
        "skor_per_responden": sorted(sus_baris),
        "rerata": round(st.mean(sus_baris), 2),
        "sb": round(sb, 2) if sb else None,
        "median": round(st.median(sus_baris), 2),
        "min": min(sus_baris),
        "maks": max(sus_baris),
        "galat_baku": round(galat, 2) if galat else None,
        "ki95": [round(st.mean(sus_baris) - t * galat, 2),
                 round(st.mean(sus_baris) + t * galat, 2)] if galat else None,
        "t_kritis": t,
    }
    hasil["sus"]["per_butir"] = {}
    for i, kol in enumerate(SUS_KOLOM):
        nilai = [angka(r[kol]) for r in data]
        nilai = [v for v in nilai if v is not None]
        hasil["sus"]["per_butir"][SUS_LABEL[i]] = {
            "arah": "positif" if i % 2 == 0 else "negatif",
            "n": len(nilai),
            "rerata": round(st.mean(nilai), 3),
            "sb": round(st.stdev(nilai), 3) if len(nilai) > 1 else None,
            "median": st.median(nilai),
            "min": min(nilai),
            "maks": max(nilai),
        }

    # --- masukan terbuka: hanya dihitung, isinya tidak disalin ---
    hasil["masukan_terbuka"] = {
        "n_mengisi": {str(k): sum(1 for r in data if r[k].strip()) for k in KOL_BEBAS}
    }

    (SUMBER_DIR / "statistik.json").write_text(
        json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(hasil, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
