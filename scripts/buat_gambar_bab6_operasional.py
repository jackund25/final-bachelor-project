#!/usr/bin/env python3
"""Hasilkan Gambar VI.11 (waktu respons) dan VI.12 (jejak memori backend).

Sumber:
- ``results/benchmark/latency_endtoend_gemini.json`` — 12 ulangan, pemanasan
  dibuang. Ini jalan 9 Agustus, yakni SETELAH model generasi dipindah ke
  gemini-3.5-flash-lite pada 7 Agustus. Berkasnya sendiri hanya merekam
  ``llm_provider: gemini`` tanpa id model, jadi kecocokan model disimpulkan dari
  tanggal, bukan dibaca dari berkas. Berkas ``latency_endtoend_gemini-2.5-flash-lite.json``
  (6 Agustus) sengaja TIDAK dipakai karena berasal dari era model sebelumnya.
- ``results/benchmark/jejak_memori_backend.json`` — enam tahap inisialisasi.
  Berkas itu memuat kunci PROVENANS yang menyatakan angkanya disalin dari prosa
  dokumen kerja, bukan keluaran skrip; skrip ini mencetak peringatan itu setiap
  kali dijalankan supaya statusnya tidak terlupakan.

Gambar VI.11 menandai p95 hanya pada tiga baris agregat (komputasi lokal,
generasi LLM, total). p95 tiap tahap kecil memang ada di berkas sumber, tetapi
menampilkannya akan mengesankan bahwa p95 dapat dijumlahkan antar-tahap,
padahal tidak.

Gambar VI.12 tidak menggambarkan 342 MB sebagai kebutuhan RAM seluruh sistem;
sumbu diberi label "jejak memori aplikasi" dan catatan kaki menyatakan angka itu
diukur sebelum overhead lingkungan.

Jalankan:
    PYTHONPATH=. python scripts/buat_gambar_bab6_operasional.py
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
BENCH = ROOT / "results/benchmark"
OUT_DIR = ROOT / "docs/laporan_TA/TA-STI-template-1.0/images/bab6"

LATENSI = BENCH / "latency_endtoend_gemini.json"
MEMORI = BENCH / "jejak_memori_backend.json"

# (kunci di per_tahap, label, agregat?) — agregat mendapat penanda p95.
TAHAP = [
    ("rekayasa_fitur", "Rekayasa fitur", False),
    ("prediksi", "Prediksi (2 horizon)", False),
    ("kalibrasi_interval", "Kalibrasi interval", False),
    ("retrieval", "Retrieval", False),
]
AGREGAT = [
    ("komputasi_lokal", "Komputasi lokal"),
    ("generasi_llm", "Generation LLM"),
    ("total", "Total end-to-end"),
]

BIRU, JINGGA, ABU, MERAH = "#3498db", "#e67e22", "#95a5a6", "#e74c3c"


def muat(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Berkas hasil tidak ditemukan: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def gambar_latensi(d: dict, out: Path):
    baris = []
    for kunci, label, _ in TAHAP:
        baris.append((label, d["per_tahap"][kunci]["median_dtk"], None, False))
    for kunci, label in AGREGAT:
        sumber = d["per_tahap"].get(kunci) or d.get(kunci)
        baris.append((label, sumber["median_dtk"], sumber["p95_dtk"], True))

    y = np.arange(len(baris))[::-1]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    for yi, (label, med, p95, agg) in zip(y, baris):
        warna = JINGGA if agg else BIRU
        ax.barh(yi, med, height=0.6, color=warna, alpha=0.85)
        # Latar putih: pada baris komputasi lokal penanda p95 jatuh tepat di belakang
        # angka median, dan garisnya memotong glifnya.
        ax.text(med + 0.045, yi, f"{med:.3f}".replace(".", ","), va="center",
                fontsize=9.5, fontweight="bold", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        if p95 is not None:
            ax.plot([p95, p95], [yi - 0.3, yi + 0.3], color=MERAH, lw=2.2, zorder=3)
            # Label p95 diletakkan di ATAS penanda, bukan di sampingnya: pada baris
            # komputasi lokal jarak median ke p95 hanya 0,19 detik sehingga label
            # samping bertabrakan dengan angka median.
            ax.text(p95, yi + 0.34, f"p95 {p95:.3f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=8.2, color=MERAH)

    # Garis pemisah antara tahap penyusun dan baris agregat: keduanya tidak boleh
    # dibaca sebagai satu daftar yang dapat dijumlahkan.
    ax.axhline(len(AGREGAT) - 0.5, color="#cccccc", lw=1.0, ls="--")

    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in baris], fontsize=10.5)
    ax.set_xlabel("Waktu (detik)", fontsize=11)
    ax.set_xlim(0, max(b[2] or b[1] for b in baris) * 1.22)
    ax.set_title("Waktu respons pipeline assessment", fontsize=12.5, fontweight="bold")

    penanda = [plt.Line2D([], [], color=BIRU, lw=8, label="Tahap penyusun (median)"),
               plt.Line2D([], [], color=JINGGA, lw=8, label="Agregat (median)"),
               plt.Line2D([], [], color=MERAH, lw=2.2, label="p95")]
    ax.legend(handles=penanda, fontsize=9, loc="upper right", framealpha=0.92)

    prop = d.get("proporsi_llm") or {}
    prop_teks = ""
    if isinstance(prop, dict) and "median" in prop:
        prop_teks = (f" · generasi LLM menempati {100 * prop['median']:.1f}% "
                     f"waktu total (median)").replace(".", ",")
    ax.text(0.5, -0.15,
            f"n = {d['n']} ulangan, pemanasan dibuang{prop_teks}",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")

    ax.grid(axis="x", alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return baris


def gambar_memori(d: dict, out: Path):
    tahap = d["tahap"]
    nama = [t["nama"] for t in tahap]
    mb = [t["memori_mb"] for t in tahap]
    batas = d["batas_render_gratis_mb"]

    x = np.arange(len(tahap))
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    ax.plot(x, mb, color=BIRU, lw=2.2, marker="o", ms=8, zorder=3,
            label="Jejak memori aplikasi")
    ax.fill_between(x, 0, mb, color=BIRU, alpha=0.12)
    for xi, v in zip(x, mb):
        ax.text(xi, v + 14, f"{v} MB", ha="center", fontsize=9.5, fontweight="bold")

    ax.axhline(batas, color=MERAH, ls="--", lw=1.8,
               label=f"Batas instans Render gratis ({batas} MB)")

    # Sisa ruang adalah inti argumennya: bukan "muat", melainkan "muat dengan sisa
    # yang habis oleh overhead lingkungan".
    sisa = batas - mb[-1]
    ax.annotate("", xy=(x[-1], batas), xytext=(x[-1], mb[-1]),
                arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1.4))
    ax.text(x[-1] - 0.12, (batas + mb[-1]) / 2, f"sisa {sisa} MB",
            ha="right", va="center", fontsize=9.5, color="#7f8c8d")

    ax.set_xticks(x)
    ax.set_xticklabels(nama, fontsize=10, rotation=18, ha="right")
    ax.set_ylabel("Jejak memori aplikasi (MB)", fontsize=11)
    ax.set_ylim(0, batas * 1.16)
    ax.set_title("Jejak memori backend menurut tahap inisialisasi",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)

    ax.text(0.5, -0.30,
            "Jejak aplikasi yang diukur sebelum overhead lingkungan, bukan total "
            "kebutuhan RAM seluruh proses sistem.",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return list(zip(nama, mb)), batas, sisa


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    lat, mem = muat(LATENSI), muat(MEMORI)

    print(f"Sumber data:\n  {LATENSI.relative_to(ROOT)}  "
          f"(provider {lat['llm_provider']}, n={lat['n']})")
    print(f"  {MEMORI.relative_to(ROOT)}")
    if "PROVENANS" in mem:
        print(f"  PERINGATAN PROVENANS: {mem['PROVENANS'][:140]}...")

    out11 = args.out_dir / "fig6.11_waktu_respons_pipeline.png"
    out12 = args.out_dir / "fig6.12_jejak_memori_backend.png"

    baris = gambar_latensi(lat, out11)
    tahap, batas, sisa = gambar_memori(mem, out12)

    print("\nGambar VI.11 — waktu respons (detik):")
    for label, med, p95, _ in baris:
        print(f"  {label:<22} median {med:6.3f}" + (f"   p95 {p95:6.3f}" if p95 else ""))
    print(f"  tersimpan : {out11.relative_to(ROOT)}  ({out11.stat().st_size / 1024:.0f} KB)")

    print("\nGambar VI.12 — jejak memori (MB):")
    for nama, v in tahap:
        print(f"  {nama:<22} {v:>5}")
    print(f"  batas {batas} MB, sisa {sisa} MB")
    print(f"  tersimpan : {out12.relative_to(ROOT)}  ({out12.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
