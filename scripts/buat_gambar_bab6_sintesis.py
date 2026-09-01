#!/usr/bin/env python3
"""Hasilkan Gambar VI.15 — sintesis hasil T1, T2, dan T3.

Gambar ini berbeda dari gambar Bab VI lainnya: strukturnya konseptual, bukan
plot atas satu berkas hasil. Yang tetap dijaga adalah jangkar buktinya — tiap
sel "Bukti" memuat angka yang DIBACA dari berkas hasil, bukan diketik ulang di
sini, sehingga sintesis ini ikut berubah bila hasilnya berubah.

TIDAK ADA SKOR AGREGAT. Ketiga tujuan sengaja dilaporkan berdampingan tanpa
diringkas menjadi satu angka: model dapat memiliki galat prediksi yang baik
tetapi lemah pada hipoglikemia, dan retrieval dapat menemukan dokumen relevan
tanpa generation yang terikat evidence. Menjumlahkannya akan menyembunyikan
justru bagian yang harus terbaca.

Bila sebuah berkas sumber tidak ada, jangkarnya dihilangkan dan barisnya tetap
tergambar; skrip mencetak jangkar mana yang gagal dimuat.

Jalankan:
    PYTHONPATH=. python scripts/buat_gambar_bab6_sintesis.py
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HASIL = ROOT / "results"
OUT_DIR = ROOT / "docs/laporan_TA/TA-STI-template-1.0/images/bab6"

WARNA = {"T1": "#3498db", "T2": "#e67e22", "T3": "#2ecc71"}
BARIS = [("bukti", "Bukti"), ("kesimpulan", "Kesimpulan"), ("batas", "Batas")]


def _koma(v, desimal=1) -> str:
    return f"{v:.{desimal}f}".replace(".", ",")


def jangkar() -> tuple[dict, list[str]]:
    """Baca angka jangkar dari berkas hasil. Mengembalikan (jangkar, kegagalan)."""
    j, gagal = {}, []

    try:
        with open(HASIL / "eval_prediksi/summary_all_horizons.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["horizon_min"] == "30" and r["model"] == "Gradient Boosting":
                    j["rmse30"] = _koma(float(r["RMSE"]), 2)
                    j["clarke30"] = _koma(float(r["Clarke_A+B"]), 2)
    except Exception:
        gagal.append("summary_all_horizons.csv")

    try:
        d = json.loads((HASIL / "eval_prediksi/cakupan_conformal.json").read_text(encoding="utf-8"))
        r = d["horizon"]["6"]["ringkasan"]["95"]
        j["cakupan30"] = _koma(r["cakupan_rerata_%"])
        j["cakupan_sd30"] = _koma(r["cakupan_sd_%"])
    except Exception:
        gagal.append("cakupan_conformal.json")

    try:
        d = json.loads((HASIL / "eval_prediksi/condition_classifier_9features.json").read_text(encoding="utf-8"))
        j["sens_clf"] = _koma(d["pengklasifikasi_kondisi"]["hipoglikemia"]["sensitivitas_%"])
        j["ppv_clf"] = _koma(d["pengklasifikasi_kondisi"]["hipoglikemia"]["PPV_%"])
        j["divergen_clf"] = _koma(d["pada_kasus_divergen"]["akurasi_kondisi_pengklasifikasi_%"])
        if "regresi_lalu_ambang" in d:
            j["sens_reg"] = _koma(d["regresi_lalu_ambang"]["hipoglikemia"]["sensitivitas_%"])
    except Exception:
        gagal.append("condition_classifier_9features.json")

    try:
        d = json.loads((HASIL / "retrieval_realcases_kb12/crossfold.json").read_text(encoding="utf-8"))
        r = d["ringkasan_lintas_fold"]
        j["mrr_div_std"] = _koma(r["divergen"]["standard"]["mrr_rerata"], 3)
        j["mrr_div_clf"] = _koma(r["divergen"]["pc_rag_classifier"]["mrr_rerata"], 3)
        j["mrr_div_ora"] = _koma(r["divergen"]["oracle"]["mrr_rerata"], 3)
        j["mrr_nat_std"] = _koma(r["natural"]["standard"]["mrr_rerata"], 3)
        j["n_fold"] = d["n_fold"]
    except Exception:
        gagal.append("retrieval_realcases_kb12/crossfold.json")

    try:
        d = json.loads((HASIL / "hasil_form/statistik.json").read_text(encoding="utf-8"))
        j["sus"] = _koma(d["sus"]["rerata"], 2)
        j["sus_acuan"] = d["sus"]["acuan_sauro_lewis"]["rerata_industri"]
        j["sus_peringkat"] = d["sus"]["acuan_sauro_lewis"]["peringkat_huruf"]
        j["n_responden"] = d["meta"]["n_responden"]
    except Exception:
        gagal.append("hasil_form/statistik.json")

    try:
        d = json.loads((HASIL / "benchmark/latency_endtoend_gemini.json").read_text(encoding="utf-8"))
        j["latensi"] = _koma(d["total"]["median_dtk"], 3)
    except Exception:
        gagal.append("latency_endtoend_gemini.json")

    return j, gagal


def susun(j: dict) -> list[dict]:
    def a(kunci, teks, cadangan=""):
        return teks if all(k in j for k in kunci) else cadangan

    return [
        {
            "kode": "T1",
            "judul": "Prediksi kadar glukosa",
            "bukti": a(["rmse30", "clarke30", "cakupan30"],
                       f"RMSE +30 menit {j.get('rmse30')} mg/dL, Clarke A+B "
                       f"{j.get('clarke30')}%. Interval konformal mencakup "
                       f"{j.get('cakupan30')}% (sd {j.get('cakupan_sd30')}) pada "
                       f"penaksir lintas-pasien."),
            "kesimpulan": "Terpenuhi pada ruang uji yang ditetapkan.",
            "batas": a(["sens_reg", "sens_clf", "ppv_clf"],
                       f"Hipoglikemia tetap sulit: sensitivitas {j.get('sens_reg')}% "
                       f"lewat regresi, naik ke {j.get('sens_clf')}% lewat "
                       f"pengklasifikasi tetapi PPV turun ke {j.get('ppv_clf')}%. "
                       f"Generalisasi di luar OhioT1DM belum diuji."),
        },
        {
            "kode": "T2",
            "judul": "RAG terkondisi-prediksi",
            "bukti": a(["mrr_div_std", "mrr_div_clf", "mrr_div_ora", "n_fold"],
                       f"Validasi silang {j.get('n_fold')} fold lintas-pasien. MRR kasus "
                       f"divergen {j.get('mrr_div_std')} menjadi {j.get('mrr_div_clf')}; "
                       f"oracle {j.get('mrr_div_ora')} sebagai kontrol. Kasus natural "
                       f"praktis datar ({j.get('mrr_nat_std')})."),
            "kesimpulan": "Manfaat jelas pada kasus divergen, bukan pada kasus natural.",
            "batas": a(["divergen_clf"],
                       f"Bergantung pada ketepatan prediksi kondisi: pada kasus divergen "
                       f"kondisi hanya benar {j.get('divergen_clf')}%. Selisih terhadap "
                       f"oracle adalah kelemahan prediksi, bukan kelemahan mekanismenya."),
        },
        {
            "kode": "T3",
            "judul": "CDSS doctor-mediated",
            "bukti": a(["latensi", "sus", "n_responden"],
                       f"Alur end-to-end median {j.get('latensi')} detik. Lima aturan "
                       f"keluaran dipatuhi pada dua kasus uji; fallback meneruskan "
                       f"keterbatasan tanpa evidence sintetis. SUS {j.get('sus')} "
                       f"dari {j.get('n_responden')} responden."),
            "kesimpulan": "Integrasi berjalan pada alur yang divalidasi dokter.",
            "batas": a(["sus", "sus_acuan", "sus_peringkat", "n_responden"],
                       f"Usability di bawah acuan industri {j.get('sus_acuan')} "
                       f"(peringkat {j.get('sus_peringkat')}). Verifikasi ahli terbatas "
                       f"pada {j.get('n_responden')} responden dan dua kasus generation."),
        },
    ]


def gambar(kolom: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(13.6, 7.6))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3.62)
    ax.axis("off")

    lebar_bungkus = 44

    for i, kol in enumerate(kolom):
        warna = WARNA[kol["kode"]]

        # Kepala kolom
        ax.add_patch(plt.Rectangle((i + 0.03, 3.06), 0.94, 0.5, facecolor=warna,
                                   alpha=0.9, edgecolor="none"))
        ax.text(i + 0.5, 3.40, kol["kode"], ha="center", va="center",
                fontsize=15, fontweight="bold", color="white")
        ax.text(i + 0.5, 3.18, kol["judul"], ha="center", va="center",
                fontsize=10.5, color="white")

        for r, (kunci, label) in enumerate(BARIS):
            y = 2.02 - r * 1.02
            ax.add_patch(plt.Rectangle((i + 0.03, y), 0.94, 0.96, facecolor=warna,
                                       alpha=0.10 + 0.03 * r, edgecolor=warna, lw=1.1))
            ax.text(i + 0.08, y + 0.84, label.upper(), ha="left", va="center",
                    fontsize=9, fontweight="bold", color=warna)
            teks = textwrap.fill(kol[kunci], lebar_bungkus)
            ax.text(i + 0.08, y + 0.42, teks, ha="left", va="center",
                    fontsize=8.6, color="#222222", linespacing=1.45)

    ax.set_title("Sintesis hasil evaluasi menurut tujuan artefak",
                 fontsize=13.5, fontweight="bold", pad=16)
    fig.text(0.5, 0.015,
             "Ketiga tujuan dilaporkan berdampingan dan sengaja TIDAK diringkas menjadi "
             "satu skor agregat: kekuatan pada satu tujuan tidak menutup keterbatasan "
             "pada tujuan lain.",
             ha="center", fontsize=8.8, color="#555555")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    j, gagal = jangkar()
    print(f"Jangkar bukti termuat: {len(j)} nilai")
    if gagal:
        print("  GAGAL dimuat (jangkarnya dihilangkan dari gambar):")
        for g in gagal:
            print(f"    {g}")

    kolom = susun(j)
    out = args.out_dir / "fig6.15_sintesis_hasil_t1_t2_t3.png"
    gambar(kolom, out)

    for kol in kolom:
        print(f"\n{kol['kode']} — {kol['judul']}")
        for kunci, label in BARIS:
            isi = kol[kunci] or "(jangkar tidak tersedia)"
            print(f"  {label:<11}: {textwrap.shorten(isi, 96)}")
    print(f"\ntersimpan : {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
