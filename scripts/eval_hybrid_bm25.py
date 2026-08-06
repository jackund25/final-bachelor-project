"""B3 — Ukur penelusuran hibrida: BM25 (jarang) + vektor (padat), digabung dengan RRF.

Alasan: penelusuran saat ini sepenuhnya bertumpu pada kemiripan vektor. Istilah klinis
seperti nama obat, angka ambang ("70 mg/dL", "15-15"), dan singkatan (KAD, HHS, TIR)
sering lebih baik ditangani pencocokan leksikal (Gao dkk. 2023, Subbab II.6.2).

Penggabungan memakai Reciprocal Rank Fusion:

    skor(d) = SIGMA_r  1 / (k_rrf + peringkat_r(d))

RRF dipilih karena bekerja pada PERINGKAT, bukan skor mentah. Skor kosinus ChromaDB dan
skor BM25 berada pada skala yang sama sekali berbeda dan tidak dapat dijumlahkan langsung;
menormalkannya lebih dulu akan menambah satu parameter bebas yang harus disetel, dan
protokol Bagian C menuntut setiap parameter yang disetel dicatat. RRF menghindari itu.

WAJIB: diterapkan pada KEDUA mode ablasi, sama seperti B2.

Skrip ini TIDAK mengubah jalur produksi. Ia hanya mengukur.

Keluaran: results/baseline_ablation_fullkb_<CORPUS_TAG>/hybrid_bm25.json
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_rag_fullkb import TEST_CASES, TOP_K, build_query, classify_chunk  # noqa: E402
from src.rag.ablation_query import build_ablation_query  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402
from src.timing import percentile  # noqa: E402

CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12_sym")
OUT = ROOT / f"results/baseline_ablation_fullkb_{CORPUS_TAG}/hybrid_bm25.json"

K_RRF = 60          # konstanta baku RRF (Cormack dkk. 2009); TIDAK disetel di sini
N_KANDIDAT = 50     # kedalaman daftar tiap penelusur sebelum digabung


def tokenisasi(teks: str) -> list[str]:
    """Tokenisasi sederhana yang mempertahankan angka dan satuan.

    Angka ambang ("70", "180", "15") justru sinyal yang diharapkan ditangkap BM25, jadi
    angka TIDAK dibuang. Tanda hubung dipertahankan sebagai pemisah supaya "15-15" menjadi
    dua token "15", yang tetap cocok dengan penulisan "15 gram".
    """
    return re.findall(r"[a-z0-9]+", teks.lower())


def muat_kueri() -> dict:
    """Himpunan uji identik dengan B1 dan B2 agar ketiganya dapat disandingkan."""
    kueri = {}
    for mode in ("standard", "prediction_conditioned"):
        kueri[f"ablasi_{mode}"] = [(build_query(c, mode), c["expected"]) for c in TEST_CASES]

    for berkas, nama in (("per_case_divergen.csv", "divergen"),
                         ("per_case_natural.csv", "natural")):
        p = ROOT / f"results/retrieval_realcases_{CORPUS_TAG}/{berkas}"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("mode") == "pc_rag"]
        kueri[f"{nama}_prediction_conditioned"] = [
            (build_ablation_query(float(r["query_glucose"])), r["expected"]) for r in rows]
        kueri[f"{nama}_standard"] = [
            (build_ablation_query(float(r["current"])), r["expected"]) for r in rows]
    return kueri


def metrik(ranks: list[int], n: int, top_k: int) -> dict:
    return {
        "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
        "hit@3(%)": round(100 * sum(1 for x in ranks if 0 < x <= 3) / n, 1),
        f"hit@{top_k}(%)": round(100 * sum(1 for x in ranks if 0 < x <= top_k) / n, 1),
        "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
    }


def rank_dari_topik(topics: list[str], expected: str) -> int:
    return (topics.index(expected) + 1) if expected in topics else 0


def rrf(daftar_peringkat: list[list[int]], k_rrf: int = K_RRF) -> list[int]:
    """Gabungkan beberapa daftar indeks terurut menjadi satu, menurut RRF."""
    skor: dict[int, float] = {}
    for daftar in daftar_peringkat:
        for pos, idx in enumerate(daftar):
            skor[idx] = skor.get(idx, 0.0) + 1.0 / (k_rrf + pos + 1)
    return sorted(skor, key=lambda i: -skor[i])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-kandidat", type=int, default=N_KANDIDAT)
    args = ap.parse_args()
    top_k = TOP_K

    print("Memuat korpus dari ChromaDB ...")
    import chromadb
    col = chromadb.PersistentClient(path="models/chroma_db").get_collection("diabetes_kb")
    got = col.get(limit=10000, include=["documents", "metadatas"])
    teks_korpus = got["documents"]
    meta_korpus = got["metadatas"]
    print(f"  {len(teks_korpus)} chunk")

    # PENTING: id koleksi Chroma adalah UUID, BUKAN chunk_id di metadata (diperiksa
    # langsung: 0 dari 2.061 cocok). Peta posisi karena itu harus dikunci pada chunk_id,
    # bukan pada got["ids"] — kalau salah, setiap pencarian meleset dan jatuh ke
    # pencocokan teks O(n) yang lambat sekaligus rawan salah pada chunk kembar.
    pos_by_chunk_id = {}
    for i, m in enumerate(meta_korpus):
        cid = (m or {}).get("chunk_id")
        if cid is not None:
            pos_by_chunk_id.setdefault(str(cid), i)
    print(f"  {len(pos_by_chunk_id)} chunk_id unik terpetakan")

    print("Membangun indeks BM25 ...")
    from rank_bm25 import BM25Okapi
    t0 = time.time()
    bm25 = BM25Okapi([tokenisasi(t) for t in teks_korpus])
    t_bangun = time.time() - t0
    print(f"  selesai dalam {t_bangun:.1f} dtk")

    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")
    kueri = muat_kueri()
    tak_terpeta = 0
    hasil = {
        "catatan": ("BM25 + vektor digabung dengan Reciprocal Rank Fusion. k_rrf=60 (nilai "
                    "baku, tidak disetel). Diterapkan pada KEDUA mode ablasi."),
        "k_rrf": K_RRF, "n_kandidat": args.n_kandidat, "top_k": top_k,
        "n_chunk": len(teks_korpus), "waktu_bangun_bm25_dtk": round(t_bangun, 1),
        "himpunan": {}, "latensi_ms": {},
    }
    lat_bm25, lat_total = [], []

    for nama_himpunan, pasangan in kueri.items():
        n = len(pasangan)
        ranks = {"baseline_mmr": [], "bm25_saja": [], "hibrida_rrf": []}

        for q, expected in pasangan:
            # (a) baseline: jalur produksi
            docs_base = r.retrieve(q, top_k=top_k)
            ranks["baseline_mmr"].append(
                rank_dari_topik([classify_chunk(d["text"]) for d in docs_base], expected))

            # (b) BM25 saja
            t0 = time.perf_counter()
            skor_bm = bm25.get_scores(tokenisasi(q))
            urut_bm = sorted(range(len(skor_bm)), key=lambda i: -skor_bm[i])[:args.n_kandidat]
            lat_bm25.append((time.perf_counter() - t0) * 1000)
            ranks["bm25_saja"].append(rank_dari_topik(
                [classify_chunk(teks_korpus[i]) for i in urut_bm[:top_k]], expected))

            # (c) hibrida: daftar padat + daftar BM25, digabung RRF
            t0 = time.perf_counter()
            qvec = r._embeddings.embed_query(q)
            pool = r._vector_store.similarity_search_by_vector_with_relevance_scores(
                embedding=qvec, k=args.n_kandidat, filter=None)
            urut_padat = []
            for d, _ in pool:
                cid = (d.metadata or {}).get("chunk_id")
                i = pos_by_chunk_id.get(str(cid)) if cid is not None else None
                if i is None:
                    tak_terpeta += 1
                    continue
                urut_padat.append(i)
            gabung = rrf([urut_padat, urut_bm])[:top_k]
            lat_total.append((time.perf_counter() - t0) * 1000)
            ranks["hibrida_rrf"].append(rank_dari_topik(
                [classify_chunk(teks_korpus[i]) for i in gabung], expected))

        hasil["himpunan"][nama_himpunan] = {
            "n": n, **{k: metrik(v, n, top_k) for k, v in ranks.items()}}
        h = hasil["himpunan"][nama_himpunan]
        print(f"  {nama_himpunan:<38} baseline {h['baseline_mmr']['mrr']:.3f} | "
              f"bm25 {h['bm25_saja']['mrr']:.3f} | hibrida {h['hibrida_rrf']['mrr']:.3f}",
              flush=True)

    # Penjagaan: bila banyak hasil padat gagal dipetakan ke indeks korpus, lengan hibrida
    # diam-diam kehilangan kandidat dan angkanya tidak sah.
    hasil["dokumen_padat_tak_terpeta"] = tak_terpeta
    if tak_terpeta:
        print(f"\nPERINGATAN: {tak_terpeta} dokumen hasil pencarian padat tidak dapat "
              f"dipetakan ke indeks korpus. Lengan hibrida kehilangan kandidat; "
              f"angkanya TIDAK sah.")

    hasil["latensi_ms"] = {
        "bm25_saja": {"median": round(statistics.median(lat_bm25), 1),
                      "p95": round(percentile(lat_bm25, 95), 1)},
        "hibrida_padat_plus_gabung": {"median": round(statistics.median(lat_total), 1),
                                      "p95": round(percentile(lat_total, 95), 1)},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2), encoding="utf-8")

    print("\n=== Latensi ===")
    for nama, v in hasil["latensi_ms"].items():
        print(f"  {nama:<32} median {v['median']:>7.1f} ms  p95 {v['p95']:>7.1f} ms")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
