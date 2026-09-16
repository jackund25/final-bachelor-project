#!/usr/bin/env python3
"""Hasilkan Gambar VI.8 — MRR retrieval pada himpunan natural dan divergen.

Sumber: ``results/retrieval_realcases_kb12/crossfold.json`` (22 Agustus 2026),
satu-satunya jalan cross-fold yang ``konfigurasi_efektif``-nya cocok dengan
config.yaml produksi: chunk 900/120, fetch_k 12, top_k 5, lambda_mult 0,0,
embedding all-MiniLM-L6-v2, prediktor HistGradientBoosting.

Varian lain (``retrieval_realcases_kb12_final``, ``kb12_gbm_bm25_v2``, dst.) telah
dipindahkan ke ``arsip/code_eksperimen/results/`` dan JANGAN dipakai: yang pertama
bertanggal 6 Agustus (sebelum migrasi ke Gradient Boosting, tanpa blok
``konfigurasi_efektif``), yang kedua memakai chunk_size 500 padahal produksi 900.

Oracle digambar sebagai kontrol, bukan metode: ia memakai kondisi masa depan
yang sudah diketahui benar, sehingga diberi warna netral dan arsiran serta
diberi label "kontrol" pada legenda.

Jalankan:
    PYTHONPATH=. python scripts/buat_gambar_bab6_retrieval.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMBER = ROOT / "results/retrieval_realcases_kb12/crossfold.json"
OUT_DIR = ROOT / "docs/laporan_TA/TA-STI-template-1.0/images/bab6"

# Urutan lengan menentukan urutan batang; oracle sengaja diletakkan terakhir.
LENGAN = [
    ("standard", "Standard", "#3498db", None),
    ("pc_rag", "Prediction-conditioned", "#e67e22", None),
    ("pc_rag_classifier", "Prediction-conditioned + classifier", "#2ecc71", None),
    ("oracle", "Oracle (kontrol, bukan metode)", "#95a5a6", "//"),
]
HIMPUNAN = [("divergen", "Kasus divergen"), ("natural", "Kasus natural")]

# Kunci config.yaml yang harus cocok dengan konfigurasi_efektif berkas hasil.
PERIKSA = {
    "chunk_size": ["rag", "chunk_size"],
    "chunk_overlap": ["rag", "chunk_overlap"],
    "fetch_k": ["rag", "fetch_k"],
    "embedding_model": ["rag", "embedding_model"],
}


def muat(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Berkas hasil tidak ditemukan: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def gali(d: dict, jalur: list[str]):
    for k in jalur:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def periksa_konfigurasi(data: dict) -> None:
    """Bandingkan konfigurasi_efektif berkas hasil terhadap config.yaml produksi."""
    eff = data.get("konfigurasi_efektif")
    if not eff:
        print("  PERINGATAN: berkas hasil tidak memuat 'konfigurasi_efektif', "
              "konfigurasinya tidak dapat diperiksa. Kemungkinan jalan pra-produksi.")
        return

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    beda = []
    for kunci, jalur in PERIKSA.items():
        produksi, hasil = gali(cfg, jalur), eff.get(kunci)
        if produksi is not None and hasil is not None and produksi != hasil:
            beda.append(f"{kunci}: hasil={hasil} config={produksi}")

    if beda:
        print("  PERINGATAN: konfigurasi berkas hasil menyimpang dari config.yaml:")
        for b in beda:
            print(f"    {b}")
    else:
        print(f"  Konfigurasi COCOK dengan config.yaml "
              f"(prediktor {eff.get('keluarga_prediktor')}, "
              f"corpus {eff.get('corpus_tag')}).")


def gambar(data: dict, out: Path) -> list[tuple]:
    ring = data["ringkasan_lintas_fold"]
    x = np.arange(len(HIMPUNAN))
    lebar = 0.8 / len(LENGAN)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    baris = []
    for i, (kunci, label, warna, arsir) in enumerate(LENGAN):
        nilai = [ring[him][kunci]["mrr_rerata"] for him, _ in HIMPUNAN]
        sd = [ring[him][kunci].get("mrr_sd", 0.0) for him, _ in HIMPUNAN]
        baris.append((label, *nilai))
        offset = (i - (len(LENGAN) - 1) / 2) * lebar
        batang = ax.bar(x + offset, nilai, lebar, label=label, color=warna,
                        alpha=0.85, hatch=arsir,
                        edgecolor="white" if arsir else "none")
        ax.errorbar(x + offset, nilai, yerr=sd, fmt="none", ecolor="#444444",
                    elinewidth=1.1, capsize=3, capthick=1.1)
        # Label diletakkan di atas ujung error bar, bukan di atas batang; dengan
        # bar_label biasa angkanya tertimpa garis galat pada lengan ber-sd besar.
        for xi, v, s in zip(x + offset, nilai, sd):
            ax.text(xi, v + s + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in HIMPUNAN], fontsize=11)
    ax.set_ylabel("MRR (rerata enam fold)", fontsize=11)
    ax.set_ylim(0, max(v for _, *vs in baris for v in vs) * 1.35)
    ax.set_title("MRR retrieval pada himpunan natural dan divergen",
                 fontsize=12.5, fontweight="bold")

    n_kasus = data.get("n_kasus_per_himpunan_per_fold")
    ax.text(0.5, -0.12,
            f"Validasi silang {data['n_fold']} fold lintas-pasien · top-k {data['top_k']}"
            + (f" · {n_kasus} kasus per himpunan per fold" if n_kasus else "")
            + " · galat = simpangan baku antar-fold",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")

    ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return baris


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sumber", type=Path, default=SUMBER)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    data = muat(args.sumber)
    print(f"Sumber data:\n  {args.sumber.relative_to(ROOT)}")
    periksa_konfigurasi(data)

    out = args.out_dir / "fig6.8_mrr_natural_vs_divergen.png"
    hasil = gambar(data, out)

    print("\nGambar VI.8 — MRR:")
    print(f"  {'lengan':<38}{'divergen':>10}{'natural':>10}")
    for label, div, nat in hasil:
        print(f"  {label:<38}{div:>10.3f}{nat:>10.3f}")
    print(f"  tersimpan : {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
