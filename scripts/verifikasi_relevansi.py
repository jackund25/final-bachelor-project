"""T2.2 — verifikasi manusia atas pelabel relevansi otomatis `classify_chunk()`.

MENGAPA INI MENENTUKAN NASIB SETENGAH BAB VI
--------------------------------------------
Seluruh metrik retrieval di laporan — Hit@1, Hit@5, MRR, nDCG@5, dan hasil T3.1/T3.2 —
memakai relevansi yang ditetapkan `classify_chunk()`: pencocokan kata kunci berbobot.
Itu keterbatasan K1. Selama ketepatannya tidak pernah diukur, setiap angka retrieval
berdiri di atas asumsi yang tidak diperiksa, dan penguji berhak menanyakannya.

Skrip ini tidak MEMPERBAIKI pelabel. Ia mengukur seberapa sering pelabel itu sepakat
dengan penilaian manusia, sehingga angka-angka retrieval dapat dilaporkan dengan
kualifikasi yang jujur — atau ditarik bila kesepakatannya rendah.

RANCANGAN YANG MENJAGA PENILAIAN TETAP BUTA
-------------------------------------------
Tiga hal yang membuat penilaian dapat dipercaya, dan semuanya mudah dilanggar tanpa
sengaja:

1. **Label otomatis tidak ada di berkas penilaian.** Disimpan terpisah pada berkas kunci.
   Kalau label terlihat, penilai akan cenderung menyetujuinya (anchoring).
2. **Urutan diacak** memakai benih tetap, sehingga label otomatis juga tidak dapat
   disimpulkan dari urutan atau pengelompokan.
3. **Pasangan diambil dari retrieval SUNGGUHAN**, bukan dari contoh yang dipilih tangan.
   Memilih contoh yang "jelas" akan melebih-lebihkan kesepakatan.

CARA PAKAI
----------
    python scripts/verifikasi_relevansi.py --buat        # menghasilkan berkas penilaian
    # isi kolom `penilaian_manusia` pada evaluation/verifikasi_relevansi.csv
    python scripts/verifikasi_relevansi.py --nilai       # menghitung kesepakatan

Nilai yang boleh diisi pada kolom `penilaian_manusia`:
    hipoglikemia | hiperglikemia | normal | lain

Definisi yang dipakai penilai HARUS sama dengan yang dipakai pelabel, kalau tidak yang
terukur adalah beda definisi, bukan beda ketepatan. Definisinya dicetak di kepala berkas:
*topik utama yang dibahas potongan teks ini*, bukan *apakah potongan ini berguna bagi
pasien tertentu*.

Keluaran: evaluation/verifikasi_relevansi.csv (untuk diisi),
          evaluation/verifikasi_relevansi_KUNCI.json (label otomatis, jangan dibuka dulu),
          results/eval_rag/verifikasi_relevansi.json (hasil, setelah --nilai)
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

BERKAS_NILAI = ROOT / "evaluation/verifikasi_relevansi.csv"
BERKAS_KUNCI = ROOT / "evaluation/verifikasi_relevansi_KUNCI.json"
BERKAS_HASIL = ROOT / "results/eval_rag/verifikasi_relevansi.json"

N_PASANGAN = 40          # di dalam rentang 30-50 yang diminta
TOP_K = 5
SEED = 42
KELAS = ["hipoglikemia", "hiperglikemia", "normal", "lain"]
MAKS_KARAKTER = 1200     # potongan ditampilkan utuh sampai batas ini

# Glukosa yang mewakili tiap kondisi. Sama dengan GLUKOSA_PER_KONDISI pada run_ragas.py
# supaya kueri di sini sepadan dengan kueri yang dipakai evaluasi RAGAS.
GLUKOSA = {"hipoglikemia": 58.0, "normal": 120.0, "hiperglikemia": 230.0}


def buat() -> int:
    from src.rag.ablation_query import build_ablation_query
    from src.rag.retriever import MMRRetriever
    from ablation_rag_fullkb import classify_chunk
    from eval_susunan_kueri import VARIAN

    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")

    # Kueri divariasikan di sekitar tiap kondisi supaya potongan yang terambil tidak
    # hanya lima chunk yang itu-itu saja.
    #
    # Grid diperlebar setelah percobaan pertama: lima nilai per kondisi hanya
    # menghasilkan 20 potongan unik dari 75 pengambilan, di bawah rentang 30-50 yang
    # dituntut. Penyebabnya `lambda_mult` 0,0 memaksimalkan keberagaman, sehingga kueri
    # berdekatan pada kondisi yang sama mengembalikan himpunan yang mirip. Yang dilebarkan
    # adalah CACAH KUERI, bukan `top_k` — menaikkan top_k akan mengambil sampel potongan
    # yang tidak pernah terambil pada konfigurasi produksi, sehingga pelabel diuji pada
    # sebaran yang salah.
    glukosa = [(kond, float(g + d)) for kond, g in GLUKOSA.items()
               for d in (-14, -11, -8, -6, -3, -1, 0, 2, 4, 6, 9, 12)]

    pasangan, terlihat = [], set()

    def kumpulkan(bentuk_nama, fn):
        """Ambil potongan unik untuk satu BENTUK kueri. Mengembalikan cacah yang baru."""
        sebelum = len(pasangan)
        for kond, g in glukosa:
            q = fn(g, kond)
            for peringkat, d in enumerate(r.retrieve(q, top_k=TOP_K), start=1):
                teks = (d.get("text") or "").strip()
                kunci_unik = teks[:200]
                if not teks or kunci_unik in terlihat:
                    continue
                terlihat.add(kunci_unik)
                meta = d.get("metadata") or {}
                pasangan.append({
                    "kueri": q, "bentuk_kueri": bentuk_nama,
                    "kondisi_kueri": kond, "glukosa_kueri": g, "peringkat": peringkat,
                    "sumber": meta.get("source") or meta.get("nama_berkas") or "?",
                    "halaman_cetak": meta.get("halaman_cetak"),
                    "teks": teks[:MAKS_KARAKTER],
                    "teks_dipotong": len(teks) > MAKS_KARAKTER,
                    "label_otomatis": classify_chunk(teks),
                })
        return len(pasangan) - sebelum

    # Bentuk PRODUKSI lebih dulu, dan cacahnya dicatat tersendiri.
    n_produksi = kumpulkan("produksi", lambda g, kond: build_ablation_query(g))
    print(f"bentuk produksi   : {n_produksi} potongan unik dari "
          f"{len(glukosa) * TOP_K} pengambilan")

    # KONSENTRASI PENELUSURAN. Bentuk produksi sendirian tidak sampai 30 potongan unik
    # betapa pun kuerinya divariasikan, karena `lambda_mult` 0,0 memilih sebaran yang
    # nyaris sama setiap kali. Itu temuan tentang sistem, bukan kekurangan skrip ini,
    # dan dicatat di berkas kunci.
    #
    # Kerangka sampel diperlebar memakai bentuk kueri LAIN yang dipakai T3.1 — bukan
    # dengan menaikkan top_k. Alasannya: pelabel `classify_chunk()` diterapkan pada
    # setiap potongan yang diambil evaluasi mana pun, sehingga bentuk kueri T3.1 memang
    # bagian dari populasi yang harus ia tangani. Menaikkan top_k justru akan menguji
    # pelabel pada potongan yang tidak pernah terambil di konfigurasi produksi.
    per_bentuk = {"produksi": n_produksi}
    if len(pasangan) < N_PASANGAN:
        for nama, fn in VARIAN:
            if nama == "produksi" or len(pasangan) >= N_PASANGAN:
                continue
            per_bentuk[nama] = kumpulkan(nama, fn)
            print(f"bentuk {nama:<11}: +{per_bentuk[nama]} potongan unik "
                  f"(total {len(pasangan)})")

    if len(pasangan) < N_PASANGAN:
        print(f"PERINGATAN: hanya {len(pasangan)} potongan unik tersedia dari SELURUH "
              f"bentuk kueri, kurang dari {N_PASANGAN} yang diminta. Seluruhnya dipakai.")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pasangan))[:min(N_PASANGAN, len(pasangan))]
    # Diacak SETELAH pemilihan pula: urutan akhir tidak boleh mencerminkan pengelompokan
    # kondisi kueri maupun peringkat retrieval.
    terpilih = [pasangan[i] for i in idx]

    sebaran = Counter(p["label_otomatis"] for p in terpilih)
    BERKAS_NILAI.parent.mkdir(parents=True, exist_ok=True)
    with BERKAS_NILAI.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# T2.2 — verifikasi relevansi. Isi HANYA kolom penilaian_manusia."])
        w.writerow([f"# Nilai yang sah: {' | '.join(KELAS)}"])
        w.writerow(["# Pertanyaannya: APA TOPIK UTAMA yang dibahas potongan ini?"])
        w.writerow(["# BUKAN: apakah potongan ini berguna bagi pasien pada kueri tersebut."])
        w.writerow(["# 'lain' dipakai bila potongan tidak membahas ketiga kondisi itu "
                    "(mis. definisi diabetes, alat suntik, gizi umum)."])
        w.writerow(["# Label otomatis SENGAJA tidak ditampilkan; ada di berkas KUNCI."])
        w.writerow([])
        w.writerow(["id", "penilaian_manusia", "kueri", "sumber", "halaman_cetak",
                    "peringkat", "teks_potongan"])
        for i, p in enumerate(terpilih, start=1):
            teks = p["teks"] + (" […dipotong]" if p["teks_dipotong"] else "")
            w.writerow([i, "", p["kueri"], p["sumber"], p["halaman_cetak"],
                        p["peringkat"], teks])

    BERKAS_KUNCI.write_text(json.dumps({
        "peringatan": ("Berkas KUNCI. Jangan dibuka sebelum penilaian manusia selesai — "
                       "melihatnya membatalkan sifat buta penilaian."),
        "dibuat": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED, "n": len(terpilih),
        "sebaran_label_otomatis": dict(sebaran),
        "potongan_unik_per_bentuk_kueri": per_bentuk,
        "konsentrasi_penelusuran": (
            f"Bentuk kueri PRODUKSI hanya menghasilkan {n_produksi} potongan unik dari "
            f"{len(glukosa) * TOP_K} pengambilan ({len(glukosa)} kueri x top_k {TOP_K}), "
            f"atas korpus 2.186 chunk. Kerangka sampel diperlebar dengan bentuk kueri lain "
            f"yang dipakai T3.1, BUKAN dengan menaikkan top_k."),
        "label": {str(i): p["label_otomatis"] for i, p in enumerate(terpilih, start=1)},
        "rincian": {str(i): {k: p[k] for k in
                             ("kondisi_kueri", "glukosa_kueri", "peringkat", "sumber",
                              "halaman_cetak", "label_otomatis")}
                    for i, p in enumerate(terpilih, start=1)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"T2.2 — berkas penilaian dibuat: {len(terpilih)} pasangan")
    print(f"  untuk diisi : {BERKAS_NILAI}")
    print(f"  kunci       : {BERKAS_KUNCI}  <- JANGAN dibuka dulu")
    print(f"  sebaran label otomatis (hanya untuk memastikan tidak timpang): "
          f"{dict(sebaran)}")
    if len(sebaran) < 2:
        print("  PERINGATAN: label otomatis nyaris seragam; kesepakatan akan sulit "
              "ditafsirkan karena tebakan mayoritas sudah hampir selalu benar.")
    print("\nLangkah berikutnya: isi kolom penilaian_manusia, lalu jalankan "
          "`python scripts/verifikasi_relevansi.py --nilai`")
    return 0


def _kappa(a: list[str], b: list[str], kelas: list[str]) -> float:
    """Cohen's kappa. Dihitung sendiri agar tidak menambah dependensi."""
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in kelas)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def nilai() -> int:
    if not BERKAS_NILAI.exists():
        raise SystemExit(f"{BERKAS_NILAI} tidak ada. Jalankan --buat lebih dulu.")
    if not BERKAS_KUNCI.exists():
        raise SystemExit(f"{BERKAS_KUNCI} tidak ada.")

    kunci = json.loads(BERKAS_KUNCI.read_text(encoding="utf-8"))
    baris = []
    with BERKAS_NILAI.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(x for x in f if not x.startswith("#") and x.strip()):
            if row.get("id"):
                baris.append(row)

    manusia, otomatis, tak_sah, kosong = [], [], [], 0
    for row in baris:
        v = (row.get("penilaian_manusia") or "").strip().lower()
        if not v:
            kosong += 1
            continue
        if v not in KELAS:
            tak_sah.append((row["id"], v))
            continue
        manusia.append(v)
        otomatis.append(kunci["label"][str(row["id"])])

    if tak_sah:
        raise SystemExit(f"Nilai tidak sah pada id {[i for i, _ in tak_sah]}: "
                         f"{sorted({v for _, v in tak_sah})}. Sah: {KELAS}")
    if not manusia:
        raise SystemExit("Belum ada satu pun penilaian yang terisi.")
    if kosong:
        print(f"PERINGATAN: {kosong} baris belum diisi dan tidak ikut dihitung. "
              f"Kesepakatan atas sebagian data dapat menyesatkan bila yang dilewati "
              f"justru kasus yang sulit.")

    n = len(manusia)
    sepakat = sum(1 for a, b in zip(manusia, otomatis) if a == b)
    kappa = _kappa(otomatis, manusia, KELAS)

    # Matriks kebingungan: baris = label otomatis, kolom = penilaian manusia.
    matriks = {a: {b: 0 for b in KELAS} for a in KELAS}
    for o, m in zip(otomatis, manusia):
        matriks[o][m] += 1

    per_kelas = {}
    for k in KELAS:
        tp = matriks[k][k]
        n_otomatis = sum(matriks[k].values())
        n_manusia = sum(matriks[a][k] for a in KELAS)
        per_kelas[k] = {
            "n_menurut_otomatis": n_otomatis, "n_menurut_manusia": n_manusia,
            "presisi_otomatis": round(tp / n_otomatis, 3) if n_otomatis else None,
            "recall_otomatis": round(tp / n_manusia, 3) if n_manusia else None,
        }

    if kappa != kappa:
        tafsir = "tidak terdefinisi"
    elif kappa < 0.20:
        tafsir = "sangat lemah"
    elif kappa < 0.40:
        tafsir = "lemah"
    elif kappa < 0.60:
        tafsir = "sedang"
    elif kappa < 0.80:
        tafsir = "kuat"
    else:
        tafsir = "sangat kuat"

    out = {
        "percobaan": "T2.2 — verifikasi manusia atas pelabel relevansi otomatis",
        "waktu": datetime.now().isoformat(timespec="seconds"),
        "n_dinilai": n, "n_belum_diisi": kosong,
        "kesepakatan_mentah_persen": round(100 * sepakat / n, 2),
        "cohen_kappa": None if kappa != kappa else round(float(kappa), 4),
        "tafsir_kappa": tafsir,
        "matriks_kebingungan_baris_otomatis_kolom_manusia": matriks,
        "per_kelas": per_kelas,
        "akibat_bagi_metrik_retrieval": (
            "Hit@1, Hit@5, MRR, nDCG@5, T3.1, dan T3.2 seluruhnya memakai label otomatis "
            "ini. Kesepakatan yang diukur di sini adalah batas atas seberapa jauh "
            "angka-angka tersebut mencerminkan relevansi yang sebenarnya."),
        "aturan_pelaporan": (
            "kappa < 0,40: angka retrieval tidak boleh dilaporkan tanpa kualifikasi "
            "eksplisit di setiap penyebutannya. kappa 0,40-0,60: dilaporkan dengan "
            "catatan. kappa > 0,60: pelabel dapat dianggap memadai untuk tujuan ini."),
    }
    BERKAS_HASIL.parent.mkdir(parents=True, exist_ok=True)
    BERKAS_HASIL.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"T2.2 — hasil verifikasi ({n} pasangan dinilai)")
    print(f"  kesepakatan mentah : {out['kesepakatan_mentah_persen']}%")
    print(f"  Cohen's kappa      : {out['cohen_kappa']} ({tafsir})")
    print(f"\n  matriks (baris = otomatis, kolom = manusia):")
    print(f"    {'':<16}" + "".join(f"{k[:6]:>9}" for k in KELAS))
    for a in KELAS:
        print(f"    {a:<16}" + "".join(f"{matriks[a][b]:>9}" for b in KELAS))
    print(f"\n  presisi/recall pelabel otomatis per kelas:")
    for k, v in per_kelas.items():
        print(f"    {k:<16} presisi {v['presisi_otomatis']} | recall {v['recall_otomatis']} "
              f"(otomatis {v['n_menurut_otomatis']}, manusia {v['n_menurut_manusia']})")
    print(f"\nDisimpan ke {BERKAS_HASIL}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--buat", action="store_true", help="hasilkan berkas penilaian")
    g.add_argument("--nilai", action="store_true", help="hitung kesepakatan")
    args = ap.parse_args()
    return buat() if args.buat else nilai()


if __name__ == "__main__":
    raise SystemExit(main())
