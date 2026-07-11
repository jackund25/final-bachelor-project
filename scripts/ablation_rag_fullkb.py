"""Ablation KORPUS-PENUH: standard vs prediction-conditioned atas KB deployment nyata.

Melengkapi T5 (`ablation_rag.py`, korpus terkontrol `manual_kb` 7 topik) dengan uji pada
SELURUH ChromaDB (~2.673 chunk PERKENI/ADA). Retrieval memakai MMRRetriever (identik dengan
sistem); topik tiap chunk diklasifikasi via kata kunci berbobot (deterministik, tanpa LLM/kuota).
Query & kasus divergen IDENTIK dengan T5 → apple-to-apple, tetapi pada korpus penuh.

Tujuan: membuktikan keunggulan prediction-conditioning bertahan di luar 7 topik berlabel —
menjawab pertanyaan "apakah novelty ini nyata di korpus deployment, bukan hanya set kecil?".

Output: results/baseline_ablation_fullkb/
"""
import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import math
from pathlib import Path
import pandas as pd
from src.rag.retriever import MMRRetriever


def ndcg_at_k(rels, k):
    """nDCG@k (metrik IR baku; Jarvelin & Kekalainen 2002). Relevansi biner: 1 bila topik chunk
    == kondisi diharapkan. DCG didiskon posisi (1/log2(pos+1)); IDCG = ideal k slot relevan
    (korpus punya >>k chunk per kondisi). Skor 0..1: menghargai relevansi DAN posisi tinggi."""
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rels))))
    return dcg / idcg if idcg > 0 else 0.0

TOP_K = 5
OUT = Path("results/baseline_ablation_fullkb")

# Kasus divergen IDENTIK dengan T5 (current normal/near, prediksi hipo/hiper).
TEST_CASES = [
    {"id": "D1", "scenario": "Insulin menumpuk pasca-bolus", "current": 112.0, "predicted": 58.0, "expected": "hipoglikemia"},
    {"id": "D2", "scenario": "Olahraga tanpa penyesuaian insulin", "current": 98.0, "predicted": 64.0, "expected": "hipoglikemia"},
    {"id": "D3", "scenario": "Tren turun cepat sebelum tidur", "current": 128.0, "predicted": 66.0, "expected": "hipoglikemia"},
    {"id": "D4", "scenario": "Makan tinggi karbohidrat", "current": 150.0, "predicted": 214.0, "expected": "hiperglikemia"},
    {"id": "D5", "scenario": "Karbohidrat + stres tinggi", "current": 162.0, "predicted": 205.0, "expected": "hiperglikemia"},
    {"id": "D6", "scenario": "Dosis insulin kurang", "current": 140.0, "predicted": 238.0, "expected": "hiperglikemia"},
]
CONDITION_PHRASE = {
    "hipoglikemia": "Hipoglikemia, gula darah rendah di bawah 70 mg/dL. Penyebab, gejala, dan penanganan segera (aturan 15-15).",
    "hiperglikemia": "Hiperglikemia, gula darah tinggi di atas 180 mg/dL. Penyebab, gejala, dan penanganan.",
    "normal": "Gula darah dalam rentang normal/target. Target kontrol glikemik dan pemantauan rutin diabetes.",
}


def classify_glucose(g: float) -> str:
    return "hipoglikemia" if g < 70 else "hiperglikemia" if g > 180 else "normal"


def build_query(case, mode: str) -> str:
    g = case["current"] if mode == "standard" else case["predicted"]
    cond = classify_glucose(g)
    horizon = "" if mode == "standard" else " (prediksi 60 menit ke depan)"
    return f"Kadar glukosa darah {g:.0f} mg/dL{horizon}. {CONDITION_PHRASE[cond]}"


# Klasifikasi topik chunk via kata kunci berbobot (nama kondisi berbobot > angka ambang).
KW = {
    "hipoglikemia": [("hipoglikemi", 3), ("hypoglycemi", 3), ("gula darah rendah", 2), ("glukosa darah rendah", 2),
                     ("15-15", 2), ("glukagon", 2), ("dekstrosa", 1), ("< 70 mg", 1), ("<70 mg", 1), ("< 54", 1)],
    "hiperglikemia": [("hiperglikemi", 3), ("hyperglycemi", 3), ("ketoasidosis", 3), ("ketoacidosis", 3),
                      ("gula darah tinggi", 2), ("keton", 2), ("hiperosmolar", 2), ("hyperosmolar", 2),
                      ("krisis hiperglikemia", 2), ("> 180 mg", 1), (">180 mg", 1), ("> 250", 1), ("poliuria", 1)],
    "normal": [("target kontrol glikemik", 3), ("kontrol glikemik", 2), ("time in range", 2),
               ("pemantauan glukosa", 1), ("hba1c", 1), ("target glikemik", 2)],
}


def classify_chunk(text: str) -> str:
    t = text.lower()
    score = {k: sum(w for kw, w in kws if kw in t) for k, kws in KW.items()}
    best = max(score, key=score.get)
    return best if score[best] > 0 else "lain"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")
    rows = []
    for case in TEST_CASES:
        expected = case["expected"]
        for mode in ("standard", "prediction_conditioned"):
            q = build_query(case, mode)
            tgt = classify_glucose(case["current"] if mode == "standard" else case["predicted"])
            docs = r.retrieve(q, top_k=TOP_K)
            topics = [classify_chunk(d["text"]) for d in docs]
            rank = (topics.index(expected) + 1) if expected in topics else 0
            rels = [1 if t == expected else 0 for t in topics]
            rows.append({
                "case": case["id"], "scenario": case["scenario"], "mode": mode,
                "predicted": case["predicted"], "expected": expected, "targeted": tgt,
                "targets_needed": int(tgt == expected),
                "top1_topic": topics[0] if topics else "-", "retrieved_topics": "|".join(topics),
                "hit@1": int(bool(topics) and topics[0] == expected),
                f"hit@{TOP_K}": int(expected in topics), "rank": rank,
                "mrr": round(1.0 / rank, 3) if rank else 0.0,
                f"precision@{TOP_K}": round(sum(t == expected for t in topics) / max(len(topics), 1), 3),
                f"ndcg@{TOP_K}": round(ndcg_at_k(rels, TOP_K), 3),
                "top_sources": " | ".join(d["source"] for d in docs),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation_fullkb_per_case.csv", index=False)
    hk, pk, nk = f"hit@{TOP_K}", f"precision@{TOP_K}", f"ndcg@{TOP_K}"
    summ = df.groupby("mode").agg(n=("case", "count"), targets_needed=("targets_needed", "mean"),
                                  hit1=("hit@1", "mean"), hitk=(hk, "mean"),
                                  mrr=("mrr", "mean"), prec=(pk, "mean"), ndcg=(nk, "mean")).reset_index()
    for c in ["targets_needed", "hit1", "hitk", "prec"]:
        summ[c] = (summ[c] * 100).round(1)
    summ["mrr"] = summ["mrr"].round(3)
    summ["ndcg"] = summ["ndcg"].round(3)
    summ = summ.rename(columns={"targets_needed": "targets_needed(%)", "hit1": "hit@1(%)",
                                "hitk": f"hit@{TOP_K}(%)", "prec": f"precision@{TOP_K}(%)",
                                "ndcg": f"ndcg@{TOP_K}"})
    summ.to_csv(OUT / "ablation_fullkb_summary.csv", index=False)

    print(f"=== Ablation KORPUS-PENUH ({r.collection.count() if hasattr(r, 'collection') else '~2673'} chunk, top_k={TOP_K}) ===")
    print(summ.to_string(index=False))
    print(f"\nOutput -> {OUT}/")


if __name__ == "__main__":
    main()
