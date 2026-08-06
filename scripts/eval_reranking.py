"""B2 — Ukur pemeringkatan ulang (cross-encoder) di atas kolam kandidat fetch_k.

Masalah yang ditangani: Hit@1 jauh di bawah Hit@5 pada himpunan natural (23,3% vs 86,7%)
berarti dokumen relevan SERING ditemukan tetapi jarang menempati peringkat pertama. Itu
persis persoalan yang ditangani pemeringkatan ulang (Gao dkk. 2023, Subbab II.6.2).

Rancangan pengukuran:

  baseline : ambil top_k lewat MMR dari kolam fetch_k  (persis jalur produksi)
  reranked : ambil kolam fetch_k, skor ulang tiap pasangan (kueri, chunk) dengan
             cross-encoder, lalu ambil top_k menurut skor itu

Kolam kandidatnya SAMA untuk kedua lengan, sehingga yang diuji murni cara memilih k dari
kolam, bukan kolamnya. Dua model cross-encoder dibandingkan: satu dilatih untuk Bahasa
Inggris, satu multilingual — B1 menunjukkan pemilihan model peka terhadap bahasa, jadi
asumsi bahwa model Inggris cukup TIDAK boleh dipakai tanpa diuji.

WAJIB: pemeringkatan ulang diterapkan pada KEDUA mode ablasi. Menerapkannya hanya pada
mode prediction-conditioned akan membuat perbandingannya terhadap mode standard tidak sah.

Skrip ini TIDAK mengubah jalur produksi. Ia hanya mengukur.

Keluaran: results/baseline_ablation_fullkb_<CORPUS_TAG>/reranking.json
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")   # cross-encoder mungkin belum ter-cache
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_rag_fullkb import TEST_CASES, TOP_K, build_query, classify_chunk  # noqa: E402
from src.config import cfg_get  # noqa: E402
from src.rag.ablation_query import build_ablation_query  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402
from src.timing import percentile  # noqa: E402

CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12_sym")
OUT = ROOT / f"results/baseline_ablation_fullkb_{CORPUS_TAG}/reranking.json"

# Kandidat cross-encoder yang berjalan lokal di CPU.
MODEL_EN = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_ML = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def muat_kueri() -> dict:
    """Himpunan uji identik dengan B1 agar hasil kedua butir dapat disandingkan."""
    kueri = {}

    # Enam kasus divergen, KEDUA mode.
    for mode in ("standard", "prediction_conditioned"):
        kueri[f"ablasi_{mode}"] = [
            (build_query(c, mode), c["expected"]) for c in TEST_CASES
        ]

    # Kasus nyata, kedua mode. `current` untuk standard, `query_glucose` untuk PC-RAG.
    for berkas, nama in (("per_case_divergen.csv", "divergen"),
                         ("per_case_natural.csv", "natural")):
        p = ROOT / f"results/retrieval_realcases_{CORPUS_TAG}/{berkas}"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("mode") == "pc_rag"]
        kueri[f"{nama}_prediction_conditioned"] = [
            (build_ablation_query(float(r["query_glucose"])), r["expected"]) for r in rows
        ]
        kueri[f"{nama}_standard"] = [
            (build_ablation_query(float(r["current"])), r["expected"]) for r in rows
        ]
    return kueri


def kolam_kandidat(retriever, query: str, fetch_k: int):
    """Kolam fetch_k menurut kemiripan vektor — sama dengan yang dienumerasi MMR."""
    qvec = retriever._embeddings.embed_query(query)
    pool = retriever._vector_store.similarity_search_by_vector_with_relevance_scores(
        embedding=qvec, k=fetch_k, filter=None)
    return [d for d, _ in pool]


def metrik(ranks: list[int], n: int, top_k: int) -> dict:
    return {
        "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
        "hit@3(%)": round(100 * sum(1 for x in ranks if 0 < x <= 3) / n, 1),
        f"hit@{top_k}(%)": round(100 * sum(1 for x in ranks if 0 < x <= top_k) / n, 1),
        "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
    }


def rank_dari_topik(topics: list[str], expected: str) -> int:
    return (topics.index(expected) + 1) if expected in topics else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=[MODEL_EN, MODEL_ML])
    args = ap.parse_args()

    fetch_k = int(cfg_get("rag.fetch_k", 12))
    top_k = TOP_K
    print(f"fetch_k={fetch_k}  top_k={top_k}  korpus_tag={CORPUS_TAG}")

    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")
    kueri = muat_kueri()

    from sentence_transformers import CrossEncoder

    hasil = {
        "catatan": ("Kolam kandidat fetch_k IDENTIK untuk semua lengan; yang berbeda hanya "
                    "cara memilih top_k dari kolam itu. Diterapkan pada KEDUA mode ablasi."),
        "fetch_k": fetch_k, "top_k": top_k,
        "himpunan": {}, "latensi_rerank_ms": {},
    }

    # Muat cross-encoder sekali di luar loop himpunan.
    encoders = {}
    for nama_model in args.models:
        t = time.time()
        encoders[nama_model] = CrossEncoder(nama_model, max_length=512)
        print(f"  dimuat {nama_model} ({time.time() - t:.1f} dtk)")

    lat = {m: [] for m in args.models}

    for nama_himpunan, pasangan in kueri.items():
        n = len(pasangan)
        ranks = {"baseline_mmr": []}
        for m in args.models:
            ranks[m] = []

        for q, expected in pasangan:
            # Lengan baseline: persis jalur produksi (MMR atas kolam fetch_k).
            docs_base = r.retrieve(q, top_k=top_k)
            ranks["baseline_mmr"].append(
                rank_dari_topik([classify_chunk(d["text"]) for d in docs_base], expected))

            # Lengan rerank: kolam yang sama, dipilih ulang oleh cross-encoder.
            pool = kolam_kandidat(r, q, fetch_k)
            teks = [d.page_content for d in pool]
            for m in args.models:
                t0 = time.perf_counter()
                skor = encoders[m].predict([(q, t) for t in teks])
                lat[m].append((time.perf_counter() - t0) * 1000)
                urut = sorted(range(len(teks)), key=lambda i: -skor[i])[:top_k]
                ranks[m].append(
                    rank_dari_topik([classify_chunk(teks[i]) for i in urut], expected))

        hasil["himpunan"][nama_himpunan] = {
            "n": n,
            **{k: metrik(v, n, top_k) for k, v in ranks.items()},
        }
        b = hasil["himpunan"][nama_himpunan]["baseline_mmr"]["mrr"]
        garis = "  ".join(
            f"{m.split('/')[-1][:22]} {hasil['himpunan'][nama_himpunan][m]['mrr']:.3f}"
            for m in args.models)
        print(f"  {nama_himpunan:<38} baseline {b:.3f} | {garis}", flush=True)

    for m in args.models:
        hasil["latensi_rerank_ms"][m] = {
            "median": round(statistics.median(lat[m]), 1),
            "p95": round(percentile(lat[m], 95), 1),
            "n_kueri": len(lat[m]),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2), encoding="utf-8")

    print("\n=== Latensi pemeringkatan ulang (per kueri, kolam fetch_k) ===")
    for m, v in hasil["latensi_rerank_ms"].items():
        print(f"  {m:<48} median {v['median']:>7.1f} ms  p95 {v['p95']:>7.1f} ms")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
