#!/usr/bin/env python3
"""Hasilkan Gambar VI.7 — sensitivitas dan PPV hipoglikemia menurut pendekatan kondisi.

PENTING — pasangan lengan harus berasal dari SATU berkas.

Gambar ini membandingkan dua pendekatan penentuan kondisi masa depan:
``regresi_lalu_ambang`` (regresi glukosa lalu diambang) dan
``pengklasifikasi_kondisi`` (pengklasifikasi tiga kelas langsung). Perbandingan
hanya sah bila kedua lengan dihitung pada himpunan uji dan set fitur yang sama,
karena itu skrip ini menolak menggabungkan angka dari dua berkas berbeda.

Sumber bawaan adalah ``condition_classifier_9features.json``, yaitu varian yang
benar-benar dimuat produksi (``models/gbm_condition_classifier_h6.pkl`` melalui
``backend/services/prediction_service.py``). Sampai 30 Agustus 2026 berkas itu
hanya memuat lengan pengklasifikasi, sehingga gambar ini terpaksa memakai
``condition_classifier.json`` (14 Agustus, 7 fitur) yang himpunan ujinya berbeda
26.445 lawan 26.388 jendela. Lengan regresi kini dihitung ulang di
``scripts/train_condition_classifier.py`` pada split dan skema fitur yang sama,
dan skrip ini mencetak peringatan bila sumber yang dipakai menyimpang lagi dari
varian produksi.

Jalankan:
    PYTHONPATH=. python scripts/buat_gambar_bab6_hipoglikemia.py
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
HASIL = ROOT / "results/eval_prediksi"
OUT_DIR = ROOT / "docs/laporan_TA/TA-STI-template-1.0/images/bab6"

PRODUKSI = HASIL / "condition_classifier_9features.json"
ARSIP_7FITUR = HASIL / "condition_classifier.json"

# Urutan lengan menentukan urutan batang. Label sengaja deskriptif dan tidak
# memuat penilaian ("lebih baik"), sesuai sifat trade-off yang digambarkan.
LENGAN = [
    ("regresi_lalu_ambang", "Regresi lalu ambang"),
    ("pengklasifikasi_kondisi", "Pengklasifikasi langsung"),
]
METRIK = [("sensitivitas_%", "Sensitivitas hipoglikemia"), ("PPV_%", "PPV hipoglikemia")]

WARNA = ["#3498db", "#e67e22"]


def muat(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Berkas hasil tidak ditemukan: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def periksa_produksi(dipakai: dict) -> None:
    """Bandingkan lengan pengklasifikasi terhadap varian yang dimuat produksi."""
    if not PRODUKSI.exists():
        print(f"  CATATAN: {PRODUKSI.name} tidak ada, pemeriksaan produksi dilewati.")
        return
    prod = muat(PRODUKSI)
    hipo_prod = prod["pengklasifikasi_kondisi"]["hipoglikemia"]
    hipo_pakai = dipakai["pengklasifikasi_kondisi"]["hipoglikemia"]
    if hipo_prod == hipo_pakai:
        print(f"  Lengan pengklasifikasi COCOK dengan varian produksi "
              f"({prod['n_features']} fitur, n_uji {prod['n_uji']}).")
    else:
        print("  PERINGATAN: lengan pengklasifikasi pada gambar BUKAN varian produksi.")
        print(f"    gambar   : sens {hipo_pakai['sensitivitas_%']}%  "
              f"PPV {hipo_pakai['PPV_%']}%  (n_uji {dipakai['n_uji']})")
        print(f"    produksi : sens {hipo_prod['sensitivitas_%']}%  "
              f"PPV {hipo_prod['PPV_%']}%  (n_uji {prod['n_uji']}, "
              f"{prod['n_features']} fitur)")
        print("    Lengan regresi pada set fitur produksi belum dihitung; lihat "
              "docstring skrip ini.")


def gambar(data: dict, out: Path) -> list[tuple]:
    x = np.arange(len(METRIK))
    lebar = 0.8 / len(LENGAN)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    baris = []
    for i, (kunci, label) in enumerate(LENGAN):
        hipo = data[kunci]["hipoglikemia"]
        nilai = [hipo[m] for m, _ in METRIK]
        baris.append((label, *nilai))
        offset = (i - (len(LENGAN) - 1) / 2) * lebar
        batang = ax.bar(x + offset, nilai, lebar, label=label,
                        color=WARNA[i % len(WARNA)], alpha=0.85)
        ax.bar_label(batang, fmt="%.1f", padding=3, fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in METRIK], fontsize=11)
    ax.set_ylabel("Persentase (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Sensitivitas dan PPV hipoglikemia menurut pendekatan kondisi",
                 fontsize=12.5, fontweight="bold")

    n_hipo = data[LENGAN[0][0]]["hipoglikemia"]["n"]
    ax.text(0.5, -0.13, f"Horizon +{data['horizon_menit']:.0f} menit · "
                        f"{n_hipo} kejadian hipoglikemia dari {data['n_uji']} jendela uji",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")

    ax.legend(fontsize=10, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return baris


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sumber", type=Path, default=PRODUKSI,
                   help=(f"berkas JSON yang memuat KEDUA lengan "
                         f"(arsip 7 fitur: {ARSIP_7FITUR.name})"))
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="direktori keluaran gambar")
    args = p.parse_args()

    data = muat(args.sumber)
    kurang = [k for k, _ in LENGAN if k not in data]
    if kurang:
        raise SystemExit(
            f"{args.sumber.name} tidak memuat lengan: {', '.join(kurang)}. "
            "Perbandingan lintas-berkas tidak sah karena himpunan uji dan set "
            "fiturnya berbeda."
        )

    print(f"Sumber data:\n  {args.sumber.relative_to(ROOT)}")
    periksa_produksi(data)

    out = args.out_dir / "fig6.7_perbandingan_sensitivitas_hipoglikemia.png"
    hasil = gambar(data, out)

    print("\nGambar VI.7 — deteksi hipoglikemia:")
    for label, sens, ppv in hasil:
        print(f"  {label:<26}: sensitivitas {sens:5.1f}%   PPV {ppv:5.1f}%")
    print(f"  tersimpan : {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
