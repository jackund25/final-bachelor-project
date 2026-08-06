"""B4 — Setel ukuran potongan dokumen menurut protokol Bagian C.

chunk_size 900 karakter dengan tumpang tindih 120 ditetapkan tanpa pengujian. Gao dkk.
(2023) menyebut ukuran lazim 100, 256, dan 512 token dan menjelaskan pertukarannya.
Karena pemecahan kini dilakukan PER HALAMAN (Tugas 1), ukuran lebih kecil tidak lagi
berisiko memutus konteks lintas halaman.

PROTOKOL (Bagian C):
  - set penyetelan dan set pelaporan dibagi dengan benih tetap SEBELUM hasil dilihat
  - kriteria MRR ditetapkan di muka (scripts/tuning_protocol.py)
  - SELURUH ukuran yang diuji dicatat ke results/tuning_log.json, termasuk yang kalah
  - angka yang dilaporkan diambil dari set pelaporan dengan ukuran yang sudah dikunci

Tumpang tindih dijaga proporsional (~13% dari ukuran potongan, mengikuti rasio 120/900
yang berlaku sekarang) supaya yang diuji benar-benar UKURAN, bukan campuran ukuran dan
rasio tumpang tindih.

Keluaran: results/baseline_ablation_fullkb_<CORPUS_TAG>/chunk_size.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_rag_fullkb import TOP_K, classify_chunk  # noqa: E402
from src.rag.ablation_query import build_ablation_query  # noqa: E402
from tuning_protocol import KRITERIA, bagi_dua, catat, pilih_terbaik  # noqa: E402

CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12_sym")
OUT = ROOT / f"results/baseline_ablation_fullkb_{CORPUS_TAG}/chunk_size.json"
TMP = ROOT / "models/_chunk_sweep"

UKURAN_SEKARANG = 900
RASIO_TUMPANG_TINDIH = 120 / 900  # dipertahankan agar hanya UKURAN yang berubah

# Ditetapkan di muka. Mencakup rentang 100-512 token yang disebut Gao dkk. (~400-2048
# karakter) dan nilai yang berlaku sekarang.
UKURAN_DIUJI = [300, 500, 900, 1400, 2000]


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


def indeks_ulang(ukuran: int, persist: Path) -> int:
    """Indeks ulang korpus dengan satu ukuran potongan. Mengembalikan jumlah chunk."""
    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
    overlap = max(int(round(ukuran * RASIO_TUMPANG_TINDIH)), 1)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/reingest_kb.py"),
         "--persist", str(persist), "--collection", "diabetes_kb",
         "--chunk-size", str(ukuran), "--chunk-overlap", str(overlap)],
        cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT)),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"reingest gagal (ukuran={ukuran}):\n"
                           f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    import chromadb
    n = chromadb.PersistentClient(path=str(persist)).get_collection("diabetes_kb").count()
    if n == 0:
        raise RuntimeError(f"Indeks kosong untuk ukuran={ukuran} — hasil tidak sah.")
    return n


def evaluasi(pasangan: list, persist: Path) -> dict:
    from src.rag.retriever import MMRRetriever
    r = MMRRetriever(persist_dir=str(persist), collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")
    ranks = []
    for q, expected in pasangan:
        topics = [classify_chunk(d["text"]) for d in r.retrieve(q, top_k=TOP_K)]
        ranks.append((topics.index(expected) + 1) if expected in topics else 0)
    n = max(len(pasangan), 1)
    return {
        "n": len(pasangan),
        "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
        "hit@3(%)": round(100 * sum(1 for x in ranks if 0 < x <= 3) / n, 1),
        f"hit@{TOP_K}(%)": round(100 * sum(1 for x in ranks if 0 < x <= TOP_K) / n, 1),
        "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
    }


def main() -> None:
    skenario = muat_skenario()
    if not skenario:
        raise SystemExit(f"Skenario tidak ditemukan untuk CORPUS_TAG={CORPUS_TAG}")

    setel, lapor, tak_bisa = [], [], []
    for nama, pasangan in skenario.items():
        bagi = bagi_dua(pasangan)
        if bagi is None:
            tak_bisa.append({"himpunan": nama, "n": len(pasangan)})
            continue
        setel += bagi[0]
        lapor += bagi[1]

    for t in tak_bisa:
        print(f"  {t['himpunan']} (n={t['n']}) terlalu kecil untuk dibagi — dilewati")
    if not setel:
        raise SystemExit("Tidak ada himpunan yang cukup besar. Penyetelan TIDAK dilakukan.")

    print(f"Set penyetelan n={len(setel)} | set pelaporan n={len(lapor)} (tidak beririsan)")
    print(f"Kriteria ditetapkan di muka: {KRITERIA}")
    print(f"Ukuran diuji: {UKURAN_DIUJI} ({len(UKURAN_DIUJI)} konfigurasi)\n")

    hasil_setel, jumlah_chunk, waktu = {}, {}, {}
    for ukuran in UKURAN_DIUJI:
        t0 = time.time()
        persist = TMP / f"cs{ukuran}"
        n_chunk = indeks_ulang(ukuran, persist)
        jumlah_chunk[ukuran] = n_chunk
        m = evaluasi(setel, persist)
        hasil_setel[ukuran] = m
        waktu[ukuran] = round(time.time() - t0, 1)
        catat("chunk_size", ukuran, {**m, "n_chunk": n_chunk},
              catatan=(f"overlap={int(round(ukuran * RASIO_TUMPANG_TINDIH))}, "
                       f"set penyetelan n={len(setel)}, korpus={CORPUS_TAG}"))
        print(f"  chunk_size={ukuran:<5} {n_chunk:>5} chunk | MRR {m['mrr']:.3f}  "
              f"hit@1 {m['hit@1(%)']:5.1f}%  ({waktu[ukuran]:.0f} dtk)", flush=True)

    terbaik = pilih_terbaik(hasil_setel)
    print(f"\nTerpilih pada set penyetelan: chunk_size={terbaik} "
          f"(MRR {hasil_setel[terbaik]['mrr']:.3f})")

    print("\nMengukur pada set pelaporan dengan ukuran terkunci ...")
    lapor_terbaik = evaluasi(lapor, TMP / f"cs{terbaik}")
    lapor_sekarang = evaluasi(lapor, TMP / f"cs{UKURAN_SEKARANG}")

    hasil = {
        "protokol": ("Bagian C: set penyetelan dan pelaporan tidak beririsan, dibagi dengan "
                     "benih tetap sebelum hasil dilihat. Kriteria MRR ditetapkan di muka. "
                     "Seluruh konfigurasi tercatat di results/tuning_log.json."),
        "kriteria": KRITERIA,
        "n_konfigurasi_diuji": len(UKURAN_DIUJI),
        "ukuran_diuji": UKURAN_DIUJI,
        "rasio_tumpang_tindih": round(RASIO_TUMPANG_TINDIH, 4),
        "n_set_penyetelan": len(setel), "n_set_pelaporan": len(lapor),
        "himpunan_tidak_dapat_disetel": tak_bisa,
        "jumlah_chunk": jumlah_chunk,
        "waktu_per_konfigurasi_dtk": waktu,
        "hasil_set_penyetelan": {str(k): v for k, v in hasil_setel.items()},
        "terpilih": terbaik,
        "set_pelaporan": {
            f"chunk_size_terpilih_{terbaik}": lapor_terbaik,
            f"chunk_size_sekarang_{UKURAN_SEKARANG}": lapor_sekarang,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2), encoding="utf-8")

    print(f"\n=== SET PELAPORAN (n={len(lapor)}, ukuran terkunci) ===")
    print(f"  chunk_size={UKURAN_SEKARANG} (sekarang) : MRR {lapor_sekarang['mrr']:.3f}  "
          f"hit@1 {lapor_sekarang['hit@1(%)']:.1f}%  {jumlah_chunk[UKURAN_SEKARANG]} chunk")
    print(f"  chunk_size={terbaik} (terpilih)  : MRR {lapor_terbaik['mrr']:.3f}  "
          f"hit@1 {lapor_terbaik['hit@1(%)']:.1f}%  {jumlah_chunk[terbaik]} chunk")
    print(f"\n{len(UKURAN_DIUJI)} konfigurasi diuji. Disimpan ke {OUT}")
    print(f"Indeks sementara ada di {TMP} — hapus manual bila tidak diperlukan lagi.")


if __name__ == "__main__":
    main()
