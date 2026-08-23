"""B2 (diagnostik) — apakah reranker kalah karena BENTUK kueri, bukan karena reranking?

Cross-encoder MS MARCO dilatih pada pasangan (PERTANYAAN, paragraf). Kueri sistem ini
berbentuk PERNYATAAN topik:

    "Kadar glukosa darah 58 mg/dL. Hipoglikemia, gula darah rendah di bawah 70 mg/dL.
     Penyebab, gejala, dan penanganan segera (aturan 15-15)."

Kalau reranker kalah dari MMR hanya karena ketidakcocokan bentuk ini, kesimpulannya
berbeda jauh dari "pemeringkatan ulang tidak berguna pada korpus ini". Yang pertama dapat
diperbaiki; yang kedua tidak.

Uji: kolam kandidat dan dokumen SAMA persis, hanya teks kueri yang diberikan kepada
cross-encoder yang diubah menjadi bentuk pertanyaan. Kueri untuk pencarian padat tetap
bentuk pernyataan, supaya kolam kandidatnya identik dan yang diuji murni masukan reranker.

Keluaran: results/baseline_ablation_fullkb_<CORPUS_TAG>/reranking_bentuk_kueri.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_rag_fullkb import TOP_K, classify_chunk  # noqa: E402
from src.config import cfg_get  # noqa: E402
from src.constants import classify_glucose_3class  # noqa: E402
from src.rag.ablation_query import build_ablation_query  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402

CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12_sym")
OUT = ROOT / f"results/baseline_ablation_fullkb_{CORPUS_TAG}/reranking_bentuk_kueri.json"

MODEL_EN = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_ML = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Bentuk pertanyaan, sejajar isinya dengan CONDITION_PHRASE supaya yang berubah hanya
# BENTUK kalimat, bukan informasi yang dikandungnya.
TANYA = {
    "hipoglikemia": "Apa penyebab, gejala, dan penanganan segera hipoglikemia "
                    "(gula darah rendah di bawah 70 mg/dL)?",
    "hiperglikemia": "Apa penyebab, gejala, dan penanganan hiperglikemia "
                     "(gula darah tinggi di atas 180 mg/dL)?",
    "normal": "Apa target kontrol glikemik dan bagaimana pemantauan rutin diabetes "
              "pada gula darah dalam rentang normal?",
}


def bentuk_pertanyaan(glukosa: float) -> str:
    return f"Kadar glukosa darah {glukosa:.0f} mg/dL. {TANYA[classify_glucose_3class(glukosa)]}"


def muat_glukosa() -> dict:
    """Nilai glukosa per skenario; kedua bentuk kueri dibangun dari nilai yang SAMA."""
    out = {}
    for berkas, nama in (("per_case_divergen.csv", "divergen"),
                         ("per_case_natural.csv", "natural")):
        p = ROOT / f"results/retrieval_realcases_{CORPUS_TAG}/{berkas}"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("mode") == "pc_rag"]
        out[nama] = [(float(r["query_glucose"]), r["expected"]) for r in rows]
    return out


def metrik(ranks: list[int], n: int) -> dict:
    return {
        "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
        "hit@3(%)": round(100 * sum(1 for x in ranks if 0 < x <= 3) / n, 1),
        "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
    }


def main() -> None:
    fetch_k = int(cfg_get("rag.fetch_k", 12))
    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")
    from sentence_transformers import CrossEncoder
    enc = {m: CrossEncoder(m, max_length=512) for m in (MODEL_EN, MODEL_ML)}

    hasil = {"catatan": ("Kolam kandidat dan dokumen identik; hanya teks kueri yang "
                         "diberikan kepada cross-encoder yang diubah bentuknya."),
             "fetch_k": fetch_k, "top_k": TOP_K, "himpunan": {}}

    for nama, kasus in muat_glukosa().items():
        n = len(kasus)
        ranks = {"baseline_mmr": []}
        for m in enc:
            ranks[f"{m}__pernyataan"] = []
            ranks[f"{m}__pertanyaan"] = []

        for g, expected in kasus:
            q_padat = build_ablation_query(g)          # kueri pencarian: TETAP pernyataan
            q_tanya = bentuk_pertanyaan(g)             # hanya untuk reranker

            docs_base = r.retrieve(q_padat, top_k=TOP_K)
            t = [classify_chunk(d["text"]) for d in docs_base]
            ranks["baseline_mmr"].append((t.index(expected) + 1) if expected in t else 0)

            qvec = r._embeddings.embed_query(q_padat)
            pool = r._vector_store.similarity_search_by_vector_with_relevance_scores(
                embedding=qvec, k=fetch_k, filter=None)
            teks = [d.page_content for d, _ in pool]

            for m, model in enc.items():
                for label, qq in (("pernyataan", q_padat), ("pertanyaan", q_tanya)):
                    skor = model.predict([(qq, t_) for t_ in teks])
                    urut = sorted(range(len(teks)), key=lambda i: -skor[i])[:TOP_K]
                    tp = [classify_chunk(teks[i]) for i in urut]
                    ranks[f"{m}__{label}"].append(
                        (tp.index(expected) + 1) if expected in tp else 0)

        hasil["himpunan"][nama] = {"n": n, **{k: metrik(v, n) for k, v in ranks.items()}}
        h = hasil["himpunan"][nama]
        print(f"\n{nama} (n={n})  baseline MRR {h['baseline_mmr']['mrr']:.3f}")
        for m in enc:
            print(f"  {m.split('/')[-1]:<34} pernyataan {h[f'{m}__pernyataan']['mrr']:.3f}"
                  f"  ->  pertanyaan {h[f'{m}__pertanyaan']['mrr']:.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2), encoding="utf-8")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
