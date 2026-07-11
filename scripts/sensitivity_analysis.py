"""Analisis sensitivitas retrieval: top_k & chunk_size (menutup celah "tanpa ablation" di §4).

A. top_k  : sweep {3,4,5,8,10} pada KB produksi — apakah metrik novelty stabil thd top_k?
B. chunk_size: sweep {500,900,1500} via KB SEMENTARA (persist di scratchpad) — KB produksi
   tidak disentuh. Ukur ablation korpus-penuh (conditioned vs standard) tiap konfigurasi.

Deterministik, lokal, tanpa kuota. Output: results/eval_prediksi/sensitivity.json
"""
import torch  # noqa: F401
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import re
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reingest_kb import _read_pdf, _clean_text, _topic_from_name  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402
from src.rag.knowledge_base import MedicalKnowledgeBase  # noqa: E402

OUT = Path("results/eval_prediksi/sensitivity.json")
TMP = Path(os.environ.get("TMP", ".")) / "chroma_sens"
PDF_DIR = Path("data/knowledge_base/additional_docs")

TEST_CASES = [
    {"id": "D1", "current": 112.0, "predicted": 58.0, "expected": "hipoglikemia"},
    {"id": "D2", "current": 98.0, "predicted": 64.0, "expected": "hipoglikemia"},
    {"id": "D3", "current": 128.0, "predicted": 66.0, "expected": "hipoglikemia"},
    {"id": "D4", "current": 150.0, "predicted": 214.0, "expected": "hiperglikemia"},
    {"id": "D5", "current": 162.0, "predicted": 205.0, "expected": "hiperglikemia"},
    {"id": "D6", "current": 140.0, "predicted": 238.0, "expected": "hiperglikemia"},
]
PHRASE = {
    "hipoglikemia": "Hipoglikemia, gula darah rendah di bawah 70 mg/dL. Penyebab, gejala, dan penanganan segera (aturan 15-15).",
    "hiperglikemia": "Hiperglikemia, gula darah tinggi di atas 180 mg/dL. Penyebab, gejala, dan penanganan.",
    "normal": "Gula darah dalam rentang normal/target. Target kontrol glikemik dan pemantauan rutin diabetes.",
}
KW = {
    "hipoglikemia": [("hipoglikemi", 3), ("hypoglycemi", 3), ("gula darah rendah", 2), ("15-15", 2),
                     ("glukagon", 2), ("dekstrosa", 1), ("< 70 mg", 1), ("<70 mg", 1), ("< 54", 1)],
    "hiperglikemia": [("hiperglikemi", 3), ("hyperglycemi", 3), ("ketoasidosis", 3), ("keton", 2),
                      ("gula darah tinggi", 2), ("hiperosmolar", 2), ("> 180 mg", 1), ("> 250", 1), ("poliuria", 1)],
    "normal": [("target kontrol glikemik", 3), ("kontrol glikemik", 2), ("time in range", 2), ("hba1c", 1)],
}


def cls_g(g):
    return "hipoglikemia" if g < 70 else "hiperglikemia" if g > 180 else "normal"


def build_query(case, mode):
    g = case["current"] if mode == "standard" else case["predicted"]
    h = "" if mode == "standard" else " (prediksi 60 menit ke depan)"
    return f"Kadar glukosa darah {g:.0f} mg/dL{h}. {PHRASE[cls_g(g)]}"


def cls_chunk(text):
    t = text.lower()
    sc = {k: sum(w for kw, w in kws if kw in t) for k, kws in KW.items()}
    b = max(sc, key=sc.get)
    return b if sc[b] > 0 else "lain"


def eval_ablation(retriever, top_k):
    agg = {}
    for mode in ("standard", "prediction_conditioned"):
        h1 = hk = tn = mrr = 0
        for case in TEST_CASES:
            exp = case["expected"]
            topics = [cls_chunk(d["text"]) for d in retriever.retrieve(build_query(case, mode), top_k=top_k)]
            tn += int(cls_g(case["current"] if mode == "standard" else case["predicted"]) == exp)
            h1 += int(bool(topics) and topics[0] == exp)
            hk += int(exp in topics)
            rank = (topics.index(exp) + 1) if exp in topics else 0
            mrr += (1.0 / rank) if rank else 0.0
        n = len(TEST_CASES)
        agg[mode] = {"targets%": round(tn / n * 100, 1), "hit@1%": round(h1 / n * 100, 1),
                     f"hit@{top_k}%": round(hk / n * 100, 1), "mrr": round(mrr / n, 3)}
    return agg


def load_docs(kb):
    docs = list(kb.load_manual_kb("manual_kb.json") or [])
    for p in sorted(PDF_DIR.glob("*.pdf")):
        txt = _clean_text(_read_pdf(p))
        if len(txt) >= 300:
            docs.append({"text": txt, "source": p.name, "topic": _topic_from_name(p),
                         "metadata": {"doc_id": re.sub(r'[^a-z0-9]+', '_', p.stem.lower())}})
    return docs


def main():
    res = {"A_top_k_sweep": {}, "B_chunk_size_sweep": {}}

    # ---- A. top_k pada KB produksi ----
    prod = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                        embed_provider="sentence-transformers")
    print("=== A. SWEEP top_k (KB produksi, chunk 900) ===")
    print(f"  {'top_k':>5} | {'cond hit@1':>10} {'cond mrr':>9} | {'std hit@1':>9} {'std mrr':>8}")
    for k in [3, 4, 5, 8, 10]:
        a = eval_ablation(prod, k)
        res["A_top_k_sweep"][k] = a
        c, s = a["prediction_conditioned"], a["standard"]
        print(f"  {k:>5} | {c['hit@1%']:>10} {c['mrr']:>9} | {s['hit@1%']:>9} {s['mrr']:>8}")

    # ---- B. chunk_size via KB sementara ----
    print("\n=== B. SWEEP chunk_size (KB sementara, top_k=5) ===")
    print(f"  {'chunk':>6} {'#chunk':>7} | {'cond hit@1':>10} {'cond mrr':>9} | {'std hit@1':>9} {'std mrr':>8}")
    for cs in [500, 900, 1500]:
        persist = str(TMP / f"cs{cs}")
        shutil.rmtree(persist, ignore_errors=True)
        kb = MedicalKnowledgeBase(kb_dir="data/knowledge_base", persist_dir=persist,
                                  collection_name="diabetes_kb", embed_provider="sentence-transformers")
        docs = load_docs(kb)
        chunks = kb.chunk_documents(documents=docs, chunk_size=cs, chunk_overlap=int(cs * 0.13))
        kb.save_to_chroma(chunks=chunks, reset_collection=True)
        rt = MMRRetriever(persist_dir=persist, collection_name="diabetes_kb", embed_provider="sentence-transformers")
        a = eval_ablation(rt, 5)
        a["n_chunks"] = len(chunks)
        res["B_chunk_size_sweep"][cs] = a
        c, s = a["prediction_conditioned"], a["standard"]
        print(f"  {cs:>6} {len(chunks):>7} | {c['hit@1%']:>10} {c['mrr']:>9} | {s['hit@1%']:>9} {s['mrr']:>8}")
        shutil.rmtree(persist, ignore_errors=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nOutput -> {OUT}")


if __name__ == "__main__":
    main()
