#!/usr/bin/env python3
"""Terjemahkan statistik putaran n=12 ke skema JSON yang dipakai skrip gambar.

``scripts/buat_gambar_bab6_kuesioner.py`` dibangun untuk skema putaran n=10 yang
bersumber dari distribusi marginal. Putaran n=12 bersumber dari baris per
responden dengan skema berbeda. Alih-alih menulis ulang skrip gambarnya,
berkas ini memetakan skema baru ke skema lama sehingga rancangan gambar yang
sudah dipakai pada laporan tetap sama dan hanya datanya yang berganti.

Jalankan:
    conda run -n diabetes-ta python scripts/kompat_statistik_form.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BARU = ROOT / "results/hasil_form/terbaru/statistik.json"
KELUAR = ROOT / "results/hasil_form/terbaru/statistik_kompat.json"

# Label butir skala 1--4 pada skema baru -> kunci pada skema lama.
PETA_BUTIR = {
    "Prediksi dan status kondisi pada CGM": "s1_cgm",
    "Prediksi finger-stick dan komunikasi keterbatasan": "s2_fingerstick",
    "Penolakan prediksi saat data tidak memadai": "s4_penolakan",
    "Menghindari kesan kepastian berlebihan": "s4_kepastian",
    "Karakteristik populasi sumber data memadai": "b11_populasi",
    "Menjaga otonomi dokter": "b11_otonomi",
    "Consent dan keamanan data": "b11_consent",
    "Dasar rekomendasi cukup jelas": "b11_dasar",
    "Alur realistis untuk praktik klinis": "b11_alur",
}

TEKS_SUS = [
    "Saya berpikir akan sering menggunakan sistem ini.",
    "Saya merasa sistem ini rumit tanpa alasan yang jelas.",
    "Saya merasa sistem ini mudah digunakan.",
    "Saya merasa membutuhkan bantuan teknis untuk menggunakan sistem ini.",
    "Saya merasa fungsi-fungsi dalam sistem ini terintegrasi dengan baik.",
    "Saya merasa terdapat terlalu banyak inkonsistensi dalam sistem ini.",
    "Saya membayangkan kebanyakan orang akan cepat mempelajari sistem ini.",
    "Saya merasa sistem ini sangat merepotkan untuk digunakan.",
    "Saya merasa percaya diri saat menggunakan sistem ini.",
    "Saya perlu mempelajari banyak hal sebelum bisa menggunakan sistem ini dengan baik.",
]


def main() -> None:
    d = json.loads(BARU.read_text(encoding="utf-8"))
    s = d["sus"]

    butir = {}
    for label, v in d["butir_skala4"].items():
        kunci = PETA_BUTIR.get(label)
        if not kunci:
            continue
        butir[kunci] = {
            "n": v["n"],
            "rerata": v["rerata"],
            "simpangan_baku": v["sb"],
            "median": v["median"],
            "modus": [v["modus"]],
            "min": v["min"],
            "maks": v["maks"],
            "ambang_setuju": 3,
            "n_setuju": v["n_min_baik"],
            "persen_setuju": v["persen_min_baik"],
        }

    sus_butir = []
    for i, (_, v) in enumerate(s["per_butir"].items()):
        r = v["rerata"]
        sumbangan = (r - 1) * 2.5 if v["arah"] == "positif" else (5 - r) * 2.5
        sus_butir.append({
            "butir": i + 1,
            "teks": TEKS_SUS[i],
            "arah": v["arah"],
            "rerata": r,
            "simpangan_baku": v["sb"],
            "sumbangan_sus": round(sumbangan, 3),
        })

    kompat = {
        "meta": {
            "n_responden": d["n_responden"],
            "sumber": d["sumber"],
            "bentuk_sumber": "satu baris per responden",
            "catatan": ("Skema kompatibilitas untuk skrip gambar Bab VI. "
                        "Sumber sebenarnya adalah baris per responden."),
        },
        "butir": butir,
        "sus": {
            "rerata": s["rerata"],
            "simpangan_baku_antar_responden": s["sb"],
            "median": s["median"],
            "rentang_individu": [s["min"], s["maks"]],
            "ki95": s["ki95"],
            "catatan": ("Skor dihitung per responden dengan algoritma Brooke "
                        "dari %d baris tanggapan." % s["n"]),
            "butir": sus_butir,
        },
    }
    KELUAR.write_text(json.dumps(kompat, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("ditulis:", KELUAR.relative_to(ROOT))


if __name__ == "__main__":
    main()
