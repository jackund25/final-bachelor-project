"""Tahap 0.4 — angka GABUNGAN B4 (chunk_size) dan B5 (lambda_mult) pada set PELAPORAN.

B4 dan B5 diuji terpisah, masing-masing dengan parameter lain tetap pada nilai lama.
Perbaikannya (+0,040 dan +0,057) **tidak dapat dijumlahkan**: keduanya bekerja pada
mekanisme yang sama, yaitu urutan dokumen yang keluar dari MMR atas kolam kandidat.
Mengubah ukuran potongan mengubah isi kolam; mengubah lambda mengubah cara memilih dari
kolam. Efeknya bisa saling menguatkan, saling meniadakan, atau tumpang tindih.

Skrip ini mengukur keempat kombinasi pada set PELAPORAN yang sama — himpunan yang tidak
pernah dipakai menyetel — sehingga angka gabungannya sah dilaporkan:

    lama     : chunk_size 900, lambda 0,5
    hanya B4 : chunk_size 500, lambda 0,5
    hanya B5 : chunk_size 900, lambda 0,0
    gabungan : chunk_size 500, lambda 0,0

Pembagian set memakai `tuning_protocol.bagi_dua()` dengan benih yang sama seperti B4 dan
B5, sehingga set pelaporannya benar-benar himpunan yang sama.

Keluaran: results/baseline_ablation_fullkb_<CORPUS_TAG>/kombinasi_b4b5.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_rag_fullkb import TOP_K, classify_chunk  # noqa: E402
from src.rag.ablation_query import build_ablation_query  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402
from tuning_protocol import bagi_dua  # noqa: E402

CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12_sym")
OUT = ROOT / f"results/baseline_ablation_fullkb_{CORPUS_TAG}/kombinasi_b4b5.json"

# Indeks yang sudah dibangun B4. Dipakai apa adanya supaya tidak ada perbedaan lain
# selain ukuran potongan.
INDEKS = {900: "models/chroma_db", 500: "models/_chunk_sweep/cs500"}

KONFIGURASI = [
    ("lama",     900, 0.5),
    ("hanya_B4", 500, 0.5),
    ("hanya_B5", 900, 0.0),
    ("gabungan", 500, 0.0),
]


def muat_skenario() -> dict:
    out = {}
    for berkas, nama in (("per_case_divergen.csv", "divergen"),
                         ("per_case_natural.csv", "natural")):
        p = ROOT / f"results/retrieval_realcases_{CORPUS_TAG}/{berkas}"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("mode") == "pc_rag"]
        out[f"{nama}_prediction_conditioned"] = [
            (build_ablation_query(float(r["query_glucose"])), r["expected"]) for r in rows]
        out[f"{nama}_standard"] = [
            (build_ablation_query(float(r["current"])), r["expected"]) for r in rows]
    return out


def evaluasi(pasangan: list, persist: str, lam: float) -> dict:
    r = MMRRetriever(persist_dir=persist, collection_name="diabetes_kb",
                     embed_provider="sentence-transformers", lambda_mult=lam)
    ranks, n_dok = [], []
    for q, expected in pasangan:
        docs = r.retrieve(q, top_k=TOP_K)
        topics = [classify_chunk(d["text"]) for d in docs]
        ranks.append((topics.index(expected) + 1) if expected in topics else 0)
        n_dok.append(len({str((d.get("metadata") or {}).get("kb_id") or d.get("source"))
                          for d in docs}))
    n = max(len(pasangan), 1)
    return {
        "n": len(pasangan),
        "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
        "hit@3(%)": round(100 * sum(1 for x in ranks if 0 < x <= 3) / n, 1),
        f"hit@{TOP_K}(%)": round(100 * sum(1 for x in ranks if 0 < x <= TOP_K) / n, 1),
        "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
        "rerata_dokumen_unik_di_top_k": round(sum(n_dok) / n, 2),
    }


def main() -> None:
    skenario = muat_skenario()
    if not skenario:
        raise SystemExit(f"Skenario tidak ditemukan untuk CORPUS_TAG={CORPUS_TAG}")

    lapor = []
    for nama, pasangan in skenario.items():
        bagi = bagi_dua(pasangan)
        if bagi is None:
            print(f"  {nama} (n={len(pasangan)}) terlalu kecil untuk dibagi — dilewati")
            continue
        lapor += bagi[1]          # BAGIAN KEDUA = set pelaporan, sama seperti B4 dan B5

    for _, ukuran, _ in KONFIGURASI:
        p = ROOT / INDEKS[ukuran]
        if not p.exists():
            raise SystemExit(f"Indeks untuk chunk_size={ukuran} tidak ada di {p}. "
                             f"Jalankan scripts/eval_chunk_size.py lebih dulu.")

    print(f"Set pelaporan n={len(lapor)} (identik dengan yang dipakai B4 dan B5)\n")
    hasil = {}
    for nama, ukuran, lam in KONFIGURASI:
        t0 = time.time()
        m = evaluasi(lapor, str(ROOT / INDEKS[ukuran]), lam)
        m.update({"chunk_size": ukuran, "lambda_mult": lam})
        hasil[nama] = m
        print(f"  {nama:<10} chunk={ukuran:<4} lambda={lam:<4} | MRR {m['mrr']:.3f}  "
              f"hit@1 {m['hit@1(%)']:5.1f}%  dok unik {m['rerata_dokumen_unik_di_top_k']:.2f}"
              f"  ({time.time() - t0:.0f} dtk)", flush=True)

    dasar = hasil["lama"]["mrr"]
    for nama in hasil:
        hasil[nama]["delta_mrr_vs_lama"] = round(hasil[nama]["mrr"] - dasar, 3)

    # Apakah efeknya aditif? Bandingkan gabungan terhadap jumlah kedua efek tunggal.
    d4 = hasil["hanya_B4"]["delta_mrr_vs_lama"]
    d5 = hasil["hanya_B5"]["delta_mrr_vs_lama"]
    dg = hasil["gabungan"]["delta_mrr_vs_lama"]
    hasil["_analisis"] = {
        "delta_B4": d4, "delta_B5": d5, "delta_gabungan": dg,
        "jumlah_naif_B4_plus_B5": round(d4 + d5, 3),
        "selisih_gabungan_vs_jumlah_naif": round(dg - (d4 + d5), 3),
        "gabungan_terbaik": dg >= max(d4, d5),
        "konfigurasi_terbaik": max(hasil, key=lambda k: hasil[k]["mrr"]
                                   if not k.startswith("_") else -9),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2), encoding="utf-8")

    a = hasil["_analisis"]
    print(f"\n=== Apakah efeknya dapat dijumlahkan? ===")
    print(f"  delta B4 sendirian      : {d4:+.3f}")
    print(f"  delta B5 sendirian      : {d5:+.3f}")
    print(f"  jumlah naif (SALAH)     : {d4 + d5:+.3f}")
    print(f"  delta gabungan SEBENARNYA: {dg:+.3f}")
    print(f"  selisih                 : {a['selisih_gabungan_vs_jumlah_naif']:+.3f}")
    print(f"\n  Konfigurasi terbaik: {a['konfigurasi_terbaik']}")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
