"""B5 — Setel lambda_mult MMR menurut protokol Bagian C.

lambda_mult saat ini 0,5 (seimbang relevansi vs keragaman). Pada korpus 12 dokumen dengan
tumpang tindih topik tinggi, keragaman mungkin kurang bernilai dibandingkan relevansi.

PROTOKOL (Bagian C) yang dipatuhi skrip ini:
  - set penyetelan dan set pelaporan dibagi dengan benih tetap SEBELUM hasil dilihat
  - kriteria MRR ditetapkan di muka (scripts/tuning_protocol.py, konstanta KRITERIA)
  - SELURUH nilai yang diuji dicatat ke results/tuning_log.json, termasuk yang kalah
  - angka yang dilaporkan diambil dari set pelaporan dengan parameter yang sudah dikunci

Selain metrik, skrip melaporkan KERAGAMAN SUMBER: rerata jumlah dokumen berbeda di dalam
top-k. Menampilkan lima potongan dari satu dokumen yang sama kurang berguna bagi dokter,
sehingga keragaman punya nilai praktis yang tidak tertangkap MRR.

Keluaran: results/baseline_ablation_fullkb_<CORPUS_TAG>/mmr_lambda.json
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
from tuning_protocol import KRITERIA, bagi_dua, catat, pilih_terbaik  # noqa: E402

CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12_sym")
OUT = ROOT / f"results/baseline_ablation_fullkb_{CORPUS_TAG}/mmr_lambda.json"

# Nilai yang diuji. Ditetapkan di muka, mencakup kedua ujung dan nilai saat ini.
NILAI_LAMBDA = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMBDA_SEKARANG = 0.5


def muat_skenario() -> dict:
    """Kasus nyata pasien hold-out, KEDUA mode, per himpunan."""
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


def evaluasi(pasangan: list, lam: float) -> dict:
    """Jalankan retrieval dengan lambda tertentu dan hitung metrik + keragaman sumber."""
    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers", lambda_mult=lam)
    ranks, n_dok_unik = [], []
    for q, expected in pasangan:
        docs = r.retrieve(q, top_k=TOP_K)
        topics = [classify_chunk(d["text"]) for d in docs]
        ranks.append((topics.index(expected) + 1) if expected in topics else 0)
        sumber = {str((d.get("metadata") or {}).get("kb_id") or d.get("source")) for d in docs}
        n_dok_unik.append(len(sumber))
    n = max(len(pasangan), 1)
    return {
        "n": len(pasangan),
        "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
        "hit@3(%)": round(100 * sum(1 for x in ranks if 0 < x <= 3) / n, 1),
        f"hit@{TOP_K}(%)": round(100 * sum(1 for x in ranks if 0 < x <= TOP_K) / n, 1),
        "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
        "rerata_dokumen_unik_di_top_k": round(sum(n_dok_unik) / n, 2),
    }


def main() -> None:
    skenario = muat_skenario()
    if not skenario:
        raise SystemExit(f"Skenario tidak ditemukan untuk CORPUS_TAG={CORPUS_TAG}")

    # Pembagian dilakukan PER HIMPUNAN supaya kedua bagian punya komposisi yang sama,
    # lalu digabung. Dilakukan SEBELUM satu pun hasil dihitung.
    setel, lapor, tak_bisa = [], [], []
    for nama, pasangan in skenario.items():
        bagi = bagi_dua(pasangan)
        if bagi is None:
            tak_bisa.append({"himpunan": nama, "n": len(pasangan)})
            continue
        setel += bagi[0]
        lapor += bagi[1]

    if tak_bisa:
        print("Himpunan yang TIDAK dapat dibagi dua (terlalu sedikit skenario):")
        for t in tak_bisa:
            print(f"  {t['himpunan']} (n={t['n']}) — penyetelan tidak dilakukan di sini")

    if not setel:
        raise SystemExit("Tidak ada himpunan yang cukup besar untuk dibagi. "
                         "Penyetelan TIDAK dilakukan; jangan laporkan angka apa pun.")

    print(f"\nSet penyetelan n={len(setel)} | set pelaporan n={len(lapor)} "
          f"(tidak beririsan, benih tetap)")
    print(f"Kriteria ditetapkan di muka: {KRITERIA}")
    print(f"Nilai yang diuji: {NILAI_LAMBDA} ({len(NILAI_LAMBDA)} konfigurasi)\n")

    # ── Tahap 1: sapuan HANYA pada set penyetelan ──
    hasil_setel = {}
    for lam in NILAI_LAMBDA:
        t0 = time.time()
        m = evaluasi(setel, lam)
        hasil_setel[lam] = m
        catat("lambda_mult", lam, m,
              catatan=f"set penyetelan n={len(setel)}, korpus={CORPUS_TAG}")
        print(f"  lambda={lam:<5} MRR {m['mrr']:.3f}  hit@1 {m['hit@1(%)']:5.1f}%  "
              f"dok unik {m['rerata_dokumen_unik_di_top_k']:.2f}  ({time.time() - t0:.0f} dtk)",
              flush=True)

    terbaik = pilih_terbaik(hasil_setel)
    print(f"\nTerpilih pada set penyetelan: lambda_mult={terbaik} "
          f"(MRR {hasil_setel[terbaik]['mrr']:.3f})")

    # ── Tahap 2: kunci parameter, ukur pada set pelaporan ──
    print("\nMengukur pada set pelaporan dengan parameter terkunci ...")
    lapor_terbaik = evaluasi(lapor, terbaik)
    lapor_sekarang = evaluasi(lapor, LAMBDA_SEKARANG)

    hasil = {
        "protokol": ("Bagian C: set penyetelan dan pelaporan tidak beririsan, dibagi dengan "
                     "benih tetap sebelum hasil dilihat. Kriteria MRR ditetapkan di muka. "
                     "Seluruh konfigurasi tercatat di results/tuning_log.json."),
        "kriteria": KRITERIA,
        "n_konfigurasi_diuji": len(NILAI_LAMBDA),
        "nilai_diuji": NILAI_LAMBDA,
        "n_set_penyetelan": len(setel),
        "n_set_pelaporan": len(lapor),
        "himpunan_tidak_dapat_disetel": tak_bisa,
        "hasil_set_penyetelan": {str(k): v for k, v in hasil_setel.items()},
        "terpilih": terbaik,
        "set_pelaporan": {
            f"lambda_terpilih_{terbaik}": lapor_terbaik,
            f"lambda_sekarang_{LAMBDA_SEKARANG}": lapor_sekarang,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2), encoding="utf-8")

    print(f"\n=== SET PELAPORAN (n={len(lapor)}, parameter terkunci) ===")
    print(f"  lambda={LAMBDA_SEKARANG} (sekarang) : MRR {lapor_sekarang['mrr']:.3f}  "
          f"hit@1 {lapor_sekarang['hit@1(%)']:.1f}%  "
          f"dok unik {lapor_sekarang['rerata_dokumen_unik_di_top_k']:.2f}")
    print(f"  lambda={terbaik} (terpilih)  : MRR {lapor_terbaik['mrr']:.3f}  "
          f"hit@1 {lapor_terbaik['hit@1(%)']:.1f}%  "
          f"dok unik {lapor_terbaik['rerata_dokumen_unik_di_top_k']:.2f}")
    print(f"\n{len(NILAI_LAMBDA)} konfigurasi diuji. Disimpan ke {OUT}")


if __name__ == "__main__":
    main()
