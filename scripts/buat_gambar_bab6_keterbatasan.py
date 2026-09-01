#!/usr/bin/env python3
"""Hasilkan Gambar VI.16 — peta keterbatasan sistem menurut lapisan.

Gambar ini adalah sintesis penulis, bukan metrik baru. Yang tetap dijaga sama
seperti Gambar VI.15: angka pendukung tiap kotak batas DIBACA dari berkas hasil,
tidak diketik ulang, sehingga peta ini ikut berubah bila hasilnya berubah.

DUA LAPISAN SENGAJA TANPA KOTAK BATAS. "Model / Generalisasi" dan "Clinical
State" dibiarkan kosong karena spesifikasi tidak menetapkan batas untuk
keduanya. Ruang kosong itu dipertahankan alih-alih diisi karangan, dan justru
memperlihatkan bahwa keterbatasan menumpuk pada lapisan ujung — data di hulu,
evaluasi dan deployment di hilir.

SATU KOTAK TANPA JANGKAR. Batas pada lapisan Generation ("konfigurasi eksperimen
dan produksi berbeda") adalah pernyataan penulis; skrip ini tidak menemukan
berkas yang mengukurnya, jadi kotaknya diberi penanda bahwa ia tidak berjangkar.

Jalankan:
    PYTHONPATH=. python scripts/buat_gambar_bab6_keterbatasan.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HASIL = ROOT / "results"
OUT_DIR = ROOT / "docs/laporan_TA/TA-STI-template-1.0/images/bab6"

BIRU_TAHAP = "#34638b"
AMBER = "#e67e22"
ABU = "#7f8c8d"


def _koma(v, d=2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def jangkar() -> tuple[dict, list[str]]:
    j, gagal = {}, []

    try:
        d = json.loads((HASIL / "eval_prediksi/hipoglikemia_h6.json").read_text(encoding="utf-8"))
        b = d["basis_kejadian"]
        j["n_jendela"] = f"{b['n_jendela']:,}".replace(",", ".")
        j["porsi_hipo"] = _koma(b["porsi_hipo_persen"])
        j["n_hipo"] = f"{b['n_hipo']:,}".replace(",", ".")
    except Exception:
        gagal.append("hipoglikemia_h6.json")

    try:
        baris = {}
        with open(HASIL / "eval_prediksi/summary_all_horizons.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["model"] == "Gradient Boosting":
                    baris[r["horizon_min"]] = r
        j["rmse30"] = _koma(float(baris["30"]["RMSE"]))
        j["rmse60"] = _koma(float(baris["60"]["RMSE"]))
        j["clarke30"] = _koma(float(baris["30"]["Clarke_A+B"]))
        j["clarke60"] = _koma(float(baris["60"]["Clarke_A+B"]))
    except Exception:
        gagal.append("summary_all_horizons.csv")

    # Ketiadaan kalibrasi finger-stick dibuktikan dari daftar berkas, bukan diasumsikan.
    # Diurutkan sebagai bilangan, bukan teks: sebagai teks "12" mendahului "6"
    # sehingga daftarnya terbaca "+60 menit, +30 menit".
    horizon_terkalibrasi = sorted(
        int(Path(p).stem.split("_h")[-1])
        for p in glob.glob(str(HASIL / "eval_prediksi/conformal_h*.json"))
        if "arsip" not in p and "per_rentang" not in p
    )
    j["horizon_konformal"] = ", ".join(f"+{h * 5} menit" for h in horizon_terkalibrasi)

    try:
        d = json.loads((HASIL / "retrieval_realcases_kb12/crossfold.json").read_text(encoding="utf-8"))
        r = d["ringkasan_lintas_fold"]
        div = r["divergen"]["pc_rag_classifier"]["mrr_rerata"] - r["divergen"]["standard"]["mrr_rerata"]
        nat = r["natural"]["pc_rag_classifier"]["mrr_rerata"] - r["natural"]["standard"]["mrr_rerata"]
        j["delta_div"] = ("+" if div >= 0 else "") + _koma(div, 3)
        j["delta_nat"] = ("+" if nat >= 0 else "") + _koma(nat, 3)
    except Exception:
        gagal.append("retrieval_realcases_kb12/crossfold.json")

    try:
        d = json.loads((HASIL / "hasil_form/statistik.json").read_text(encoding="utf-8"))
        j["n_responden"] = d["meta"]["n_responden"]
    except Exception:
        gagal.append("hasil_form/statistik.json")

    try:
        d = json.loads((HASIL / "benchmark/jejak_memori_backend.json").read_text(encoding="utf-8"))
        j["memori"] = d["tahap"][-1]["memori_mb"]
        j["batas_memori"] = d["batas_render_gratis_mb"]
    except Exception:
        gagal.append("jejak_memori_backend.json")

    return j, gagal


def susun(j: dict) -> list[tuple[str, list[tuple[str, str, bool]]]]:
    """(nama lapisan, [(judul batas, isi, berjangkar)])."""
    def a(kunci, teks):
        return teks if all(k in j for k in kunci) else None

    return [
        ("DATASET", [
            ("Dataset", a(["n_jendela", "porsi_hipo", "n_hipo"],
                          f"Populasi dan kejadian hipoglikemia terbatas: 12 pasien, "
                          f"hipoglikemia hanya {j.get('porsi_hipo')}% dari "
                          f"{j.get('n_jendela')} jendela ({j.get('n_hipo')} kejadian)."), True),
        ]),
        ("Model / Generalisasi", []),
        ("Prediction + Uncertainty", [
            ("Prediction", a(["rmse30", "rmse60", "clarke30", "clarke60"],
                             f"Galat meningkat pada horizon lebih panjang: RMSE "
                             f"{j.get('rmse30')} menjadi {j.get('rmse60')} mg/dL; "
                             f"Clarke A+B {j.get('clarke30')}% menjadi {j.get('clarke60')}%."), True),
            ("Uncertainty", a(["horizon_konformal"],
                              f"Kalibrasi konformal hanya tersedia untuk CGM "
                              f"{j.get('horizon_konformal')}; jalur finger-stick 240 menit "
                              f"belum terkalibrasi."), True),
        ]),
        ("Clinical State", []),
        ("RAG / Retrieval", [
            ("Retrieval", a(["delta_div", "delta_nat"],
                            f"Manfaat conditioning terpusat pada kasus divergen: MRR "
                            f"{j.get('delta_div')} pada divergen lawan {j.get('delta_nat')} "
                            f"pada natural."), True),
        ]),
        ("Generation + Citation", [
            ("Generation", "Konfigurasi eksperimen dan produksi berbeda.", False),
        ]),
        ("CDSS / Expert Evaluation", [
            ("Expert", a(["n_responden"],
                         f"Verifikasi ahli terbatas: n = {j.get('n_responden')} responden, "
                         f"dan kepatuhan generation diuji pada dua kasus."), True),
        ]),
        ("Deployment", [
            ("Deployment", a(["memori", "batas_memori"],
                             f"Backend dijalankan lokal melalui ngrok; jejak "
                             f"{j.get('memori')} MB terhadap batas {j.get('batas_memori')} MB "
                             f"instans gratis."), True),
        ]),
    ]


def gambar(lapisan: list, out: Path):
    n = len(lapisan)
    # Tinggi baris mengikuti jumlah kotak batas: lapisan dengan dua kotak diberi ruang
    # lebih, kalau tidak teks kotak pertama menyentuh judul kotak kedua.
    tinggi = [1.0 if len(b) <= 1 else 1.5 for _, b in lapisan]
    total = sum(tinggi)
    # Titik bawah tiap baris, dihitung dari lapisan terakhir ke atas.
    dasar, akum = [], 0.2
    for t in reversed(tinggi):
        dasar.append(akum)
        akum += t
    dasar.reverse()

    fig, ax = plt.subplots(figsize=(13.2, 1.15 * total + 1.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total + 0.35)
    ax.axis("off")

    for i, (nama, batas) in enumerate(lapisan):
        y = dasar[i]

        tinggi_baris = tinggi[i]
        tengah = y + tinggi_baris / 2

        # Kotak tahap, dipusatkan pada barisnya
        ax.add_patch(plt.Rectangle((0.15, tengah - 0.29), 3.1, 0.58,
                                   facecolor=BIRU_TAHAP, alpha=0.92, edgecolor="none"))
        ax.text(1.70, tengah, nama, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white")

        # Panah ke tahap berikutnya
        if i < n - 1:
            ax.annotate("", xy=(1.70, dasar[i + 1] + tinggi[i + 1] / 2 + 0.30),
                        xytext=(1.70, tengah - 0.30),
                        arrowprops=dict(arrowstyle="-|>", color=BIRU_TAHAP, lw=1.8))

        if not batas:
            continue

        tinggi_kotak = (tinggi_baris - 0.22) / len(batas)
        for k, (judul, isi, berjangkar) in enumerate(batas):
            if isi is None:
                continue
            yb = y + 0.11 + (len(batas) - 1 - k) * tinggi_kotak
            pusat = yb + (tinggi_kotak - 0.06) / 2
            ax.plot([3.25, 3.62], [pusat, pusat], color=AMBER, lw=1.3, ls=":")
            ax.add_patch(plt.Rectangle((3.62, yb), 6.2, tinggi_kotak - 0.06,
                                       facecolor=AMBER, alpha=0.13,
                                       edgecolor=AMBER, lw=1.1))
            ax.text(3.76, pusat + 0.15, judul.upper(),
                    ha="left", va="center", fontsize=8, fontweight="bold", color=AMBER)
            tanda = "" if berjangkar else "  (tanpa jangkar pengukuran)"
            ax.text(3.76, pusat - 0.09, textwrap.fill(isi + tanda, 108),
                    ha="left", va="center", fontsize=8.2, color="#222222",
                    linespacing=1.4)

    ax.set_title("Peta keterbatasan sistem menurut lapisan",
                 fontsize=13.5, fontweight="bold", pad=14)
    fig.text(0.5, 0.005,
             "Sintesis penulis atas hasil evaluasi, bukan metrik baru. Keterbatasan berasal "
             "dari beberapa lapisan sekaligus dan membatasi tafsir hasil secara keseluruhan.",
             ha="center", fontsize=8.6, color="#555555")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    j, gagal = jangkar()
    print(f"Jangkar termuat: {len(j)} nilai")
    if gagal:
        print("  GAGAL dimuat:")
        for g in gagal:
            print(f"    {g}")

    lapisan = susun(j)
    out = args.out_dir / "fig6.16_peta_keterbatasan_sistem.png"
    gambar(lapisan, out)

    print()
    for nama, batas in lapisan:
        if not batas:
            print(f"{nama}  — tanpa kotak batas")
            continue
        for judul, isi, berjangkar in batas:
            status = "" if berjangkar else "  [TANPA JANGKAR]"
            print(f"{nama}  -> {judul}{status}")
            print(f"     {textwrap.shorten(isi or '(jangkar tidak tersedia)', 100)}")
    print(f"\ntersimpan : {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
