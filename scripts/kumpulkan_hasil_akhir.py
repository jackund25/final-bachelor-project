#!/usr/bin/env python3
"""Kumpulkan hasil akhir yang dipakai laporan ke ``results/__Hasil_Akhir__``.

Direktori ``results/`` memuat sebelas varian ``retrieval_realcases_*``, lima
varian ``baseline_ablation_*``, serta banyak berkas bertanda ``_arsip``,
``_PARSIAL``, dan ``_TIDAK_LAYAK``. Sebagian besar merupakan jejak eksperimen
dan bukan angka yang dilaporkan. Percampuran itu pernah menimbulkan kekeliruan
provenans, maka berkas yang benar-benar menopang laporan dikumpulkan terpisah.

Kriteria masuk: angka utama berkas itu muncul pada laporan yang dikumpulkan.
Berkas hasil yang benar tetapi tidak dikutip laporan (misalnya kalibrasi
konformal dengan pasien pengukur terpisah, atau validasi silang +60 menit)
sengaja TIDAK disertakan, agar direktori ini tidak memuat angka yang
bertentangan dengan laporan.

Skrip ini menyalin, bukan memindahkan. Berkas asli tetap menjadi sumber
kebenaran, dan salinannya diberi SHA-256 sehingga penyimpangan dapat dideteksi
dengan menjalankan ulang skrip ini. Salinan yang tidak lagi ada di daftar
dihapus agar direktori selalu sama persis dengan daftar.

Jalankan:
    conda run -n diabetes-ta python scripts/kumpulkan_hasil_akhir.py
    conda run -n diabetes-ta python scripts/kumpulkan_hasil_akhir.py --periksa
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUJUAN = ROOT / "results/__Hasil_Akhir__"

# (jalur sumber relatif terhadap ROOT, nama di dalam __Hasil_Akhir__,
#  bagian laporan yang ditopang)
BERKAS: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------- T1 prediksi --
    ("results/eval_prediksi/summary_all_horizons.csv",
     "T1_prediksi/holdout_semua_horizon.csv",
     "Tabel VI.3, kinerja hold-out ketiga model pada kedua horizon"),
    ("results/eval_prediksi/gradient_boosting_h6.json",
     "T1_prediksi/crossfold_h30m.json",
     "Tabel VI.4, validasi silang lintas-pasien horizon +30 menit"),
    ("results/eval_prediksi/smbg_sweep.json",
     "T1_prediksi/holdout_fingerstick.json",
     "Tabel VI.5, baris hold-out skenario finger-stick"),
    ("results/eval_prediksi/crossval_smbg.json",
     "T1_prediksi/crossfold_fingerstick.json",
     "Tabel VI.5, baris cross-fold skenario finger-stick"),
    ("results/eval_prediksi/condition_classifier.json",
     "T1_prediksi/pengklasifikasi_kondisi.json",
     "Tabel VI.8, regresi-lalu-ambang vs pengklasifikasi kondisi"),
    ("results/eval_prediksi/hypo_improve.json",
     "T1_prediksi/sapuan_ambang_hipoglikemia.json",
     "Sapuan ambang peringatan 70, 85, dan 90 mg/dL pada Bab VI"),
    ("results/eval_prediksi/h30m/clarke_grid_gbm.png",
     "T1_prediksi/clarke_grid_h30m.png",
     "Gambar VI.1a, Clarke Error Grid horizon +30 menit"),
    ("results/eval_prediksi/h60m/clarke_grid_gbm.png",
     "T1_prediksi/clarke_grid_h60m.png",
     "Gambar VI.1b, Clarke Error Grid horizon +60 menit"),

    # --------------------------------------------------------- T2 retrieval --
    ("results/retrieval_realcases_kb12/crossfold.json",
     "T2_retrieval/crossfold.json",
     "Tabel VI.10, Lampiran B.1, abstrak, dan ablasi horizon conditioning"),
    ("results/retrieval_realcases_kb12/summary.json",
     "T2_retrieval/summary.json",
     "Kasus nyata dua pasien hold-out (120 kasus) dan kontrol lengan acak"),
    ("results/retrieval_realcases_kb12/per_case_divergen.csv",
     "T2_retrieval/per_kasus_divergen.csv",
     "Rincian per kasus divergen di balik Tabel VI.10"),
    ("results/retrieval_realcases_kb12/per_case_natural.csv",
     "T2_retrieval/per_kasus_natural.csv",
     "Rincian per kasus natural di balik Tabel VI.10"),
    ("results/reranker_ablation.json",
     "T2_retrieval/ablasi_reranker.json",
     "Ablasi reranker pada Bab VI"),

    # -------------------------------------------------------- T3 generation --
    ("results/eval_prediksi/generation_novelty.json",
     "T3_generation/generation_novelty.json",
     "sim_ref dan action coverage pada enam kasus, RAG standar vs terkondisi"),
    ("results/ragas/kestabilan.json",
     "T3_generation/kestabilan_ragas.json",
     "Kestabilan penilai RAGAS antar-run pada Bab VI"),

    # ------------------------------------------------------- operasional ----
    ("results/benchmark/latency_endtoend_gemini.json",
     "operasional/latensi_ujung_ke_ujung.json",
     "Waktu tanggap ujung-ke-ujung dan komputasi lokal pada Bab VI"),

    # ----------------------------------------------------- evaluasi ahli ----
    ("results/hasil_form/terbaru/statistik.json",
     "evaluasi_ahli/statistik_n12.json",
     "Tabel VI.16 sampai VI.19"),
    ("results/hasil_form/terbaru/statistik_kompat.json",
     "evaluasi_ahli/statistik_n12_skema_lama.json",
     "Gambar VI.14"),

    # ------------------------------------------------------------ korpus ----
    ("data/knowledge_base/manifest.csv",
     "korpus/manifest.csv",
     "Tabel D.1, korpus pedoman klinis"),

    # ----------------------------------------------------------- lain-lain --
    ("results/ringkasan_untuk_bab6.json",
     "ringkasan_untuk_bab6.json",
     "Tabel kalibrasi konformal (faktor q, cakupan, lebar) dan ringkasan "
     "lintas-tujuan"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blok in iter(lambda: f.read(65536), b""):
            h.update(blok)
    return h.hexdigest()


def tulis_manifest(baris: list[tuple[str, str, str, str]]) -> None:
    lines = [
        "# results/__Hasil\\_Akhir__",
        "",
        "Seluruh hasil akhir yang dipakai pada laporan tugas akhir, terpisah",
        "dari jejak eksperimen di `results/`. Isi direktori ini **salinan**;",
        "sumber kebenarannya tetap berkas asal pada kolom Sumber.",
        "",
        "Kriteria masuk: angka utama berkas muncul pada laporan yang",
        "dikumpulkan. Berkas hasil yang benar tetapi tidak dikutip laporan",
        "sengaja tidak disertakan, agar direktori ini tidak memuat angka yang",
        "bertentangan dengan laporan.",
        "",
        "Direktori ini dibangkitkan skrip. Jangan menyunting isinya dengan",
        "tangan, karena perubahan akan tertimpa. Untuk memperbarui:",
        "",
        "```bash",
        "conda run -n diabetes-ta python scripts/kumpulkan_hasil_akhir.py",
        "```",
        "",
        "Memeriksa apakah salinan masih sama dengan sumbernya:",
        "",
        "```bash",
        "conda run -n diabetes-ta python scripts/kumpulkan_hasil_akhir.py --periksa",
        "```",
        "",
        "## Isi",
        "",
        "| Berkas | Menopang | Sumber | SHA-256 |",
        "| --- | --- | --- | --- |",
    ]
    for tujuan, menopang, sumber, h in baris:
        lines.append("| `%s` | %s | `%s` | `%s` |"
                     % (tujuan, menopang, sumber, h[:16]))
    lines += [
        "",
        "## Yang sengaja tidak ada di sini",
        "",
        "- Varian `retrieval_realcases_*` selain `kb12`. Angka laporan berasal",
        "  dari `kb12` lengan BM25; varian lain adalah eksperimen.",
        "- Seluruh `baseline_ablation_*`, `tuning_log.json`, dan berkas",
        "  bertanda `_arsip`, `_PARSIAL`, `_TIDAK_LAYAK`, `_kuota_habis`.",
        "- Hasil yang benar tetapi tidak dikutip laporan: validasi silang",
        "  horizon +60 menit, kalibrasi konformal dengan pasien pengukur",
        "  terpisah (`cakupan_conformal.json`), pemeriksaan keterlacakan angka",
        "  (`generation_safety.json`), dan tolok ukur keterterapan",
        "  (`deployability.json`).",
        "- Berkas mentah kuesioner. CSV dan PDF sumbernya memuat nama lengkap",
        "  dan alamat surel responden, jadi tidak pernah dilacak git. Hanya",
        "  statistik agregatnya yang disertakan.",
        "",
    ]
    (TUJUAN / "README.md").write_text(
        "\n".join(lines).replace("\n", "\r\n"), encoding="utf-8")


def bersihkan(dipakai: set[Path]) -> list[str]:
    """Hapus salinan yang tidak lagi ada di daftar. Kembalikan yang gagal."""
    gagal = []
    if not TUJUAN.exists():
        return gagal
    for p in TUJUAN.rglob("*"):
        if p.is_file() and p not in dipakai and p.name != "README.md":
            try:
                p.unlink()
                print("dibuang (tidak lagi di daftar):",
                      p.relative_to(TUJUAN).as_posix())
            except OSError as e:
                gagal.append("%s (%s)" % (p.relative_to(TUJUAN).as_posix(),
                                          e.strerror))
    for d in sorted(TUJUAN.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return gagal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--periksa", action="store_true",
                    help="hanya bandingkan salinan dengan sumber, jangan menyalin")
    args = ap.parse_args()

    baris: list[tuple[str, str, str, str]] = []
    hilang: list[str] = []
    beda: list[str] = []
    dipakai: set[Path] = set()

    for rel_sumber, rel_tujuan, menopang in BERKAS:
        src = ROOT / rel_sumber
        dst = TUJUAN / rel_tujuan
        dipakai.add(dst)
        if not src.exists():
            hilang.append(rel_sumber)
            continue
        h = sha256(src)
        if args.periksa:
            if not dst.exists():
                beda.append("%s (salinan belum ada)" % rel_tujuan)
            elif sha256(dst) != h:
                beda.append("%s (isi berbeda dari sumber)" % rel_tujuan)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        baris.append((rel_tujuan, menopang, rel_sumber, h))

    if hilang:
        print("SUMBER TIDAK DITEMUKAN:")
        for x in hilang:
            print("  ", x)

    if args.periksa:
        sisa = [p.relative_to(TUJUAN).as_posix() for p in TUJUAN.rglob("*")
                if p.is_file() and p not in dipakai and p.name != "README.md"]
        if sisa:
            beda.extend("%s (tidak ada di daftar)" % s for s in sisa)
        if beda:
            print("MENYIMPANG:")
            for x in beda:
                print("  ", x)
            return 1
        print("%d berkas sama dengan sumbernya, tidak ada salinan asing"
              % len(baris))
        return 1 if hilang else 0

    gagal = bersihkan(dipakai)
    tulis_manifest(baris)
    print("%d berkas disalin ke %s" % (len(baris), TUJUAN.relative_to(ROOT)))
    print("manifest: results/__Hasil_Akhir__/README.md")
    if gagal:
        print("TIDAK BISA DIHAPUS, hapus manual:")
        for x in gagal:
            print("  ", x)
    return 1 if (hilang or gagal) else 0


if __name__ == "__main__":
    sys.exit(main())
