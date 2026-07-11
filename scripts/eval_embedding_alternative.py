"""Uji model embedding alternatif (multilingual) terhadap all-MiniLM-L6-v2.

Hipotesis awal: karena korpus pedoman klinis berbahasa Indonesia, model embedding
multilingual (paraphrase-multilingual-MiniLM-L12-v2) semestinya me-retrieve lebih baik
daripada all-MiniLM-L6-v2 yang dilatih untuk Bahasa Inggris.

Skrip ini menguji hipotesis tersebut secara langsung: korpus yang sama diindeks ulang
memakai model alternatif ke koleksi ChromaDB terpisah, lalu ablation korpus-penuh
(RAG standar vs PC-RAG) dijalankan ulang di atasnya dengan kasus uji, metrik, dan
top_k yang identik. Yang berubah HANYA model embedding.

Hasil pada laporan: hipotesis TIDAK terbukti — Hit@1 PC-RAG jatuh dari 83,3% ke 0% dan
MRR dari 0,889 ke 0,319, sehingga all-MiniLM-L6-v2 dipertahankan.

Keluaran: results/baseline_ablation_fullkb/embedding_alternative.json

Catatan: model alternatif (~470 MB) diunduh sekali oleh sentence-transformers.
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BASE_MODEL = "all-MiniLM-L6-v2"
ALT_PERSIST = "models/chroma_db_multilingual"
COLLECTION = "diabetes_kb"
OUT = ROOT / "results/baseline_ablation_fullkb/embedding_alternative.json"


def ingest_with(model_name: str, persist_dir: str) -> int:
    """Indeks ulang korpus yang sama memakai model embedding tertentu.

    Memanggil scripts/reingest_kb.py — jalur ingest produksi — agar korpus (manual_kb.json
    + PDF pedoman) dan parameter chunking benar-benar identik; yang berbeda hanya embedding.
    """
    import subprocess

    env = dict(os.environ, HF_EMBED_MODEL=model_name, PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/reingest_kb.py"),
         "--persist", persist_dir, "--collection", COLLECTION],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"reingest gagal:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")

    import chromadb
    count = chromadb.PersistentClient(path=persist_dir).get_collection(COLLECTION).count()
    if count == 0:
        raise RuntimeError("Indeks alternatif kosong — perbandingan tidak sah.")
    return count


def ablate_on(persist_dir: str, model_name: str) -> dict:
    """Jalankan ablation korpus-penuh (identik dengan ablation_rag_fullkb) pada indeks tertentu."""
    os.environ["HF_EMBED_MODEL"] = model_name
    for mod in [m for m in list(sys.modules) if m.startswith("src.rag")]:
        del sys.modules[mod]
    from src.rag.retriever import MMRRetriever

    import importlib
    ab = importlib.import_module("ablation_rag_fullkb")

    r = MMRRetriever(persist_dir=persist_dir, collection_name=COLLECTION,
                     embed_provider="sentence-transformers")

    agg = {}
    for mode in ("standard", "prediction_conditioned"):
        hit1 = hitk = mrr = 0.0
        for case in ab.TEST_CASES:
            q = ab.build_query(case, mode)
            docs = r.retrieve(q, top_k=ab.TOP_K)
            topics = [ab.classify_chunk(d["text"]) for d in docs]
            expected = case["expected"]
            rank = (topics.index(expected) + 1) if expected in topics else 0
            hit1 += int(bool(topics) and topics[0] == expected)
            hitk += int(expected in topics)
            mrr += (1.0 / rank) if rank else 0.0
        n = len(ab.TEST_CASES)
        agg[mode] = {
            "hit@1(%)": round(100 * hit1 / n, 1),
            f"hit@{ab.TOP_K}(%)": round(100 * hitk / n, 1),
            "mrr": round(mrr / n, 3),
        }
    return agg


def main() -> None:
    print(f"[1/2] Indeks ulang korpus dengan {ALT_MODEL} ...")
    n_chunks = ingest_with(ALT_MODEL, ALT_PERSIST)
    print(f"      {n_chunks} chunk terindeks ke {ALT_PERSIST}")

    print(f"[2/2] Ablation pada indeks multilingual ...")
    alt = ablate_on(ALT_PERSIST, ALT_MODEL)

    print("      Ablation pada indeks produksi (all-MiniLM) ...")
    base = ablate_on("models/chroma_db", BASE_MODEL)

    out = {
        "catatan": "Korpus, kasus uji, metrik, dan top_k identik. Hanya model embedding yang berbeda.",
        "n_chunks_alternatif": n_chunks,
        BASE_MODEL: base,
        ALT_MODEL: alt,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Perbandingan model embedding (PC-RAG) ===")
    for m, res in ((BASE_MODEL, base), (ALT_MODEL, alt)):
        pc = res["prediction_conditioned"]
        print(f"  {m:45s} Hit@1 {pc['hit@1(%)']:5.1f}%  MRR {pc['mrr']:.3f}")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
