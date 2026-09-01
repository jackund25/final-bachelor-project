#!/usr/bin/env python3
"""Hasilkan Gambar VI.13 (skenario CDSS) dan VI.14 (profil butir SUS).

Sumber: ``results/hasil_form/terbaru/statistik_kompat.json`` (n = 12
responden), yaitu statistik putaran akhir yang dipetakan ke skema lama oleh
``scripts/kompat_statistik_form.py``. Putaran lama tidak dipakai lagi dan
berkasnya tidak dilacak git. Berkas sumber ini
menyatakan sendiri bahwa bentuk sumbernya adalah distribusi marginal per
pertanyaan, bukan baris per responden; konsekuensinya skor SUS per responden
tidak teridentifikasi dan hanya reratanya yang eksak.

MENGAPA GAMBAR VI.14 DUA PANEL.

Butir SUS berselang-seling arah: butir ganjil positif, butir genap negatif.
Pada butir negatif, rerata yang LEBIH TINGGI berarti usability yang LEBIH
BURUK, karena responden makin menyetujui pernyataan yang merugikan. Menggambar
kesepuluh rerata dalam satu deret akan membuat butir 4 dan 10 (rerata 4,286,
tertinggi di seluruh instrumen) tampak sebagai butir terbaik, padahal sumbangan
SUS-nya justru terendah, yaitu 1,79 dari maksimum 10.

Karena itu panel kiri memberi warna berbeda menurut arah butir dan panel kanan
menampilkan sumbangan SUS yang sudah dibalik arahnya. Panel kanan itulah yang
sebanding antar-butir; panel kiri tidak.

Jalankan:
    PYTHONPATH=. python scripts/buat_gambar_bab6_kuesioner.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMBER = ROOT / "results/hasil_form/terbaru/statistik_kompat.json"
OUT_DIR = ROOT / "docs/laporan_TA/TA-STI-template-1.0/images/bab6"

# (kunci di statistik.json, label sumbu) — urutan menentukan urutan batang.
SKENARIO = [
    ("s1_cgm", "Skenario CGM"),
    ("s2_fingerstick", "Skenario finger-stick"),
    ("s4_penolakan", "Penolakan prediksi saat\ndata tidak memadai"),
    ("s4_kepastian", "Menghindari kepastian\nberlebihan"),
]

BIRU, JINGGA, MERAH, ABU = "#3498db", "#e67e22", "#e74c3c", "#7f8c8d"
SUMBANGAN_MAKS = 10.0


def muat(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Berkas hasil tidak ditemukan: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def gambar_skenario(d: dict, out: Path):
    butir = d["butir"]
    baris = [(lab, butir[k]["rerata"], butir[k]["simpangan_baku"], butir[k]["n"])
             for k, lab in SKENARIO]

    x = np.arange(len(baris))
    rerata = np.array([b[1] for b in baris])
    sd = np.array([b[2] for b in baris])

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    batang = ax.bar(x, rerata, width=0.5, color=BIRU, alpha=0.85)
    ax.errorbar(x, rerata, yerr=sd, fmt="none", ecolor="#1f4e79",
                elinewidth=1.5, capsize=6, capthick=1.5)
    for xi, v, s in zip(x, rerata, sd):
        ax.text(xi, v + s + 0.06, f"{v:.3f}".replace(".", ","), ha="center",
                fontsize=10, fontweight="bold")

    # Titik tengah skala digambar supaya "cenderung positif" terbaca terhadap
    # acuan, bukan terhadap tinggi batang semata.
    ax.axhline(2.5, color=ABU, ls=":", lw=1.4, label="Titik tengah skala (2,5)")

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in baris], fontsize=10)
    ax.set_ylabel("Rerata skor (skala 1-4)", fontsize=11)
    ax.set_ylim(1, 4.6)
    ax.set_title("Penilaian skenario CDSS oleh responden",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)

    ax.text(0.5, -0.20,
            f"n = {baris[0][3]} responden · galat = simpangan baku antar-responden",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return baris


def gambar_sus(d: dict, out: Path):
    sus = d["sus"]
    butir = sus["butir"]
    y = np.arange(len(butir))[::-1]

    label = [f"{b['butir']}. {b['teks'][:44]}{'...' if len(b['teks']) > 44 else ''}"
             for b in butir]
    rerata = [b["rerata"] for b in butir]
    sd = [b["simpangan_baku"] for b in butir]
    sumbangan = [b["sumbangan_sus"] for b in butir]
    negatif = [b["arah"] == "negatif" for b in butir]
    warna = [JINGGA if n else BIRU for n in negatif]

    fig, (kiri, kanan) = plt.subplots(
        1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    # --- Panel kiri: rerata mentah, TIDAK sebanding antar-arah ---------------
    kiri.barh(y, rerata, height=0.62, color=warna, alpha=0.85)
    kiri.errorbar(rerata, y, xerr=sd, fmt="none", ecolor="#444444",
                  elinewidth=1.1, capsize=3, capthick=1.1)
    for yi, v in zip(y, rerata):
        kiri.text(v + 0.08, yi, f"{v:.3f}".replace(".", ","), va="center",
                  fontsize=8.8, fontweight="bold")

    kiri.set_yticks(y)
    kiri.set_yticklabels(label, fontsize=8.6)
    kiri.set_xlim(1, 5.6)
    kiri.set_xlabel("Rerata jawaban (skala 1-5)", fontsize=10.5)
    kiri.set_title("(a) Rerata mentah — TIDAK sebanding antar-arah butir",
                   fontsize=10.5, loc="left", fontweight="bold")
    kiri.grid(axis="x", alpha=0.25, ls=":")
    kiri.set_axisbelow(True)

    # Legenda diletakkan di level gambar, bukan di dalam panel (a): batang di sana
    # terentang dari x=1 sampai 4,3 pada setiap baris sehingga tidak ada sudut kosong,
    # dan kotak legenda menutupi butir 10.
    penanda = [plt.Line2D([], [], color=BIRU, lw=8, label="Butir positif (tinggi = baik)"),
               plt.Line2D([], [], color=JINGGA, lw=8, label="Butir negatif (tinggi = BURUK)")]

    # --- Panel kanan: sumbangan SUS, sebanding antar-butir -------------------
    kanan.barh(y, sumbangan, height=0.62, color=warna, alpha=0.85)
    for yi, v in zip(y, sumbangan):
        kanan.text(v + 0.15, yi, f"{v:.2f}".replace(".", ","), va="center",
                   fontsize=8.8, fontweight="bold")

    kanan.axvline(SUMBANGAN_MAKS, color=ABU, ls=":", lw=1.2)
    kanan.text(SUMBANGAN_MAKS - 0.15, len(butir) - 0.4, "maks 10",
               ha="right", fontsize=8.4, color=ABU)

    kanan.set_yticks(y)
    kanan.set_yticklabels([str(b["butir"]) for b in butir], fontsize=9)
    kanan.set_xlim(0, 11.4)
    kanan.set_xlabel("Sumbangan skor SUS (0-10)", fontsize=10.5)
    kanan.set_title("(b) Sumbangan SUS — sudah dibalik, sebanding",
                    fontsize=10.5, loc="left", fontweight="bold")
    kanan.grid(axis="x", alpha=0.25, ls=":")
    kanan.set_axisbelow(True)

    acuan = sus.get("acuan_sauro_lewis", {})
    fig.suptitle(
        f"Profil butir SUS (skor total {str(sus['rerata']).replace('.', ',')}"
        + (f"; acuan industri {acuan['rerata_industri']}, peringkat {acuan['peringkat_huruf']}"
           if acuan else "") + ")",
        fontsize=12.5, fontweight="bold")
    fig.legend(handles=penanda, fontsize=9, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.035), frameon=False)
    fig.text(0.5, -0.085, sus.get("catatan", ""), ha="center", fontsize=8.4,
             color="#555555", wrap=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return list(zip([b["butir"] for b in butir], rerata, sumbangan, negatif))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--sumber", type=Path, default=SUMBER,
                   help="statistik.json yang dipakai (skema lama, isi n=12)")
    args = p.parse_args()

    d = muat(args.sumber)
    meta = d["meta"]
    print(f"Sumber data:\n  {args.sumber.relative_to(ROOT)}")
    print(f"  n = {meta['n_responden']} · {meta['bentuk_sumber']}")

    out13 = args.out_dir / "fig6.13_ringkasan_skenario_cdss.png"
    out14 = args.out_dir / "fig6.14_profil_butir_sus.png"

    skenario = gambar_skenario(d, out13)
    sus = gambar_sus(d, out14)

    print("\nGambar VI.13 — skenario CDSS:")
    for lab, rer, sd, _ in skenario:
        print(f"  {lab.replace(chr(10), ' '):<40} {rer:.3f} (sd {sd})")
    print(f"  tersimpan : {out13.relative_to(ROOT)}  ({out13.stat().st_size / 1024:.0f} KB)")

    print("\nGambar VI.14 — profil butir SUS:")
    print(f"  {'butir':>5} {'arah':<8} {'rerata':>7} {'sumbangan':>10}")
    for no, rer, sumb, neg in sus:
        print(f"  {no:>5} {'negatif' if neg else 'positif':<8} {rer:>7} {sumb:>10}")
    print(f"  tersimpan : {out14.relative_to(ROOT)}  ({out14.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
