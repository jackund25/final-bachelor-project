#!/usr/bin/env python3
"""Hasilkan Gambar VI.5 dan VI.6 — cakupan dan lebar interval konformal.

PENTING — dari mana angkanya diambil, dan mengapa bukan dari berkas lain.

Gambar VI.5 (cakupan) memakai ``results/eval_prediksi/cakupan_conformal.json``,
yaitu percobaan T12b yang memisahkan pasien latih, pasien kalibrasi, dan pasien
pengukur. Ia TIDAK memakai ``coverage%`` di ``conformal_h{6,12}.json`` karena
kedua berkas itu memuat ``PERINGATAN_CAKUPAN``: himpunan kalibrasi dan himpunan
pelaporannya sama, sehingga cakupannya optimistis dan tidak boleh dikutip
sebagai cakupan yang tercapai. Yang sah dari berkas tersebut adalah faktor q
dan lebar interval, bukan cakupannya.

Gambar VI.6 (lebar) memakai ``conformal_h{6,12}.json``. Lebar interval tidak
terkena peringatan di atas: ia adalah konsekuensi langsung dari q, dan q sah
karena himpunan kalibrasi tidak pernah dilihat model saat pelatihan.

Kedua sumber saling konsisten: q rerata T12b (2,162 pada +30) praktis sama
dengan q produksi di ``conformal_h6.json`` (2,15), sehingga cakupan pada
Gambar VI.5 dan lebar pada Gambar VI.6 berasal dari kalibrasi yang setara.

JANGAN menyalin angka dari ``results/ringkasan_untuk_bab6.json``. Snapshot di
sana bertanggal 6 Agustus dan berasal dari era Random Forest
(``conformal_h{6,12}_RF_arsip.json``); angkanya berbeda dari model produksi.
Skrip ini sengaja tidak memuat satu pun angka hasil secara literal supaya
kekeliruan salin-angka itu tidak terulang.

Jalankan:
    PYTHONPATH=. python scripts/buat_gambar_bab6_konformal.py
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

CAKUPAN = HASIL / "cakupan_conformal.json"
KONFORMAL = {6: HASIL / "conformal_h6.json", 12: HASIL / "conformal_h12.json"}

# Varian interval yang dilaporkan. "conformal_normalized" menskalakan q dengan
# sebaran kuantil per titik, sehingga lebarnya menyesuaikan ketidakpastian lokal.
VARIAN = "conformal_normalized"
MODEL_PRODUKSI = "HistGradientBoostingRegressor"

WARNA_BATANG = "#3498db"
WARNA_TARGET = "#e74c3c"
WARNA_SEBARAN = "#7f8c8d"


def muat(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Berkas hasil tidak ditemukan: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def periksa_model(konformal: dict[int, dict]) -> None:
    """Peringatkan bila berkas kalibrasi bukan berasal dari model produksi.

    Pemeriksaan ini ada karena arsip Random Forest memakai skema JSON yang sama
    tanpa kunci ``model_family``. Tanpa penjagaan ini, menunjuk skrip ke berkas
    arsip akan menghasilkan gambar yang tampak benar tetapi salah modelnya.
    """
    for langkah, d in konformal.items():
        keluarga = d.get("model_family")
        if keluarga is None:
            print(f"  PERINGATAN h{langkah}: berkas tidak memuat 'model_family'. "
                  f"Kemungkinan arsip pra-produksi — periksa sebelum dipakai di laporan.")
        elif keluarga != MODEL_PRODUKSI:
            print(f"  PERINGATAN h{langkah}: model_family = {keluarga}, "
                  f"bukan {MODEL_PRODUKSI} yang dipakai model produksi.")


def gambar_cakupan(cakupan: dict, level: str, out: Path) -> list[tuple]:
    """Gambar VI.5 — cakupan penaksir jujur terhadap garis target nominal.

    Dipakai dot plot, bukan bar chart. Selisih yang digambarkan kecil (sekitar
    satu poin persen antar-horizon) sedangkan sumbunya harus dipotong agar
    simpangan antar-putaran terbaca; batang dengan sumbu terpotong akan
    melebih-lebihkan selisih itu lewat panjang batangnya. Penanda titik tidak
    memiliki panjang yang bisa disalahbaca, sehingga sumbu boleh dipotong.
    """
    horizon = sorted(cakupan["horizon"].items(), key=lambda kv: int(kv[0]))
    baris = []
    for langkah, isi in horizon:
        r = isi["ringkasan"][level]
        baris.append((isi["menit"], r["cakupan_rerata_%"], r["cakupan_sd_%"],
                      r["cakupan_min_%"], r["cakupan_maks_%"], r["n_putaran"]))

    menit = [b[0] for b in baris]
    rerata = np.array([b[1] for b in baris])
    sd = np.array([b[2] for b in baris])
    minimum = np.array([b[3] for b in baris])
    maksimum = np.array([b[4] for b in baris])

    x = np.arange(len(baris))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    target = float(level)
    ax.axhline(target, color=WARNA_TARGET, ls="--", lw=1.6, zorder=1,
               label=f"Target nominal {target:.0f}%")

    # Rentang min-maks digambar lebih dahulu dan lebih tipis agar simpangan baku
    # tetap menjadi elemen yang paling menonjol.
    for xi, lo, hi in zip(x, minimum, maksimum):
        ax.vlines(xi, lo, hi, color=WARNA_SEBARAN, lw=1.0, ls=":", zorder=2)
        ax.plot([xi - 0.05, xi + 0.05], [lo, lo], color=WARNA_SEBARAN, lw=1.0, zorder=2)
        ax.plot([xi - 0.05, xi + 0.05], [hi, hi], color=WARNA_SEBARAN, lw=1.0, zorder=2)
    ax.plot([], [], color=WARNA_SEBARAN, lw=1.0, ls=":",
            label="Rentang min-maks antar-putaran")

    ax.errorbar(x, rerata, yerr=sd, fmt="o", color=WARNA_BATANG,
                ecolor="#1f4e79", ms=10, mec="#1f4e79", mew=1.4,
                elinewidth=1.8, capsize=7, capthick=1.8, zorder=3,
                label=f"Cakupan empiris ± simpangan baku ({baris[0][5]} putaran)")

    for xi, nilai, s in zip(x, rerata, sd):
        ax.text(xi + 0.1, nilai, f"{nilai:.1f}%", ha="left", va="center",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"+{m:.0f} menit" for m in menit], fontsize=11)
    ax.set_xlim(-0.5, len(baris) - 0.5)
    ax.set_ylabel("Cakupan empiris (%)", fontsize=11)
    ax.set_xlabel("Horizon prediksi", fontsize=11)
    ax.set_title(f"Cakupan interval konformal pada level nominal {target:.0f}%",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(min(minimum.min(), target) - 6, max(maksimum.max(), target) + 4)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=8.8, loc="lower center", framealpha=0.92)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return baris


def gambar_lebar(konformal: dict[int, dict], level: str, out: Path) -> list[tuple]:
    """Gambar VI.6 — lebar rata-rata interval per horizon."""
    baris = []
    for langkah in sorted(konformal):
        d = konformal[langkah]
        v = d["levels"][level][VARIAN]
        baris.append((d["horizon_min"], v["mean_width"], v["q"]))

    menit = [b[0] for b in baris]
    lebar = np.array([b[1] for b in baris])

    x = np.arange(len(baris))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    batang = ax.bar(x, lebar, width=0.45, color=WARNA_BATANG, alpha=0.85)
    ax.bar_label(batang, fmt="%.1f", padding=3, fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"+{m:.0f} menit" for m in menit], fontsize=11)
    ax.set_ylabel("Lebar interval rata-rata (mg/dL)", fontsize=11)
    ax.set_xlabel("Horizon prediksi", fontsize=11)
    ax.set_title(f"Lebar rata-rata interval konformal (level {float(level):.0f}%)",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(0, lebar.max() * 1.22)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return baris


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--level", default="95", choices=["90", "95"],
                   help="level nominal interval (bawaan: 95)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="direktori keluaran gambar")
    args = p.parse_args()

    cakupan = muat(CAKUPAN)
    konformal = {langkah: muat(path) for langkah, path in KONFORMAL.items()}

    print("Sumber data:")
    print(f"  {CAKUPAN.relative_to(ROOT)}  ({cakupan['n_pasien']} pasien)")
    for langkah, path in KONFORMAL.items():
        print(f"  {path.relative_to(ROOT)}")
    periksa_model(konformal)

    out5 = args.out_dir / "fig6.5_coverage_interval_konformal.png"
    out6 = args.out_dir / "fig6.6_lebar_interval_per_horizon.png"

    hasil5 = gambar_cakupan(cakupan, args.level, out5)
    hasil6 = gambar_lebar(konformal, args.level, out6)

    print(f"\nGambar VI.5 — cakupan (level {args.level}%):")
    for menit, rerata, sd, lo, hi, n in hasil5:
        print(f"  +{menit:>2.0f} menit : {rerata:5.1f}%  sd {sd:.1f}  "
              f"rentang {lo:.1f}-{hi:.1f}  ({n} putaran)")
    print(f"  tersimpan : {out5.relative_to(ROOT)}  ({out5.stat().st_size / 1024:.0f} KB)")

    print(f"\nGambar VI.6 — lebar interval ({VARIAN}, level {args.level}%):")
    for menit, lebar, q in hasil6:
        print(f"  +{menit:>2.0f} menit : {lebar:6.1f} mg/dL  (q = {q})")
    print(f"  tersimpan : {out6.relative_to(ROOT)}  ({out6.stat().st_size / 1024:.0f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
